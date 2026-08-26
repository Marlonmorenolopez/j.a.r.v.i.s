import asyncio
import threading
import json
import sys
import time
import traceback
from pathlib import Path

# ============================================================================
# CRITICAL: Patch subprocess BEFORE any other imports that might use it
# This fixes UnicodeDecodeError on Windows with Spanish locale (cp1252)
# BUT we must be careful not to break asyncio's internal use of Popen
# ============================================================================
import subprocess
import locale

# Only patch subprocess.run (which is commonly called with text=True)
# Patching Popen breaks asyncio on Windows
_orig_run = subprocess.run
def _patched_run(*args, **kwargs):
    if "encoding" not in kwargs and kwargs.get("text", False):
        kwargs["encoding"] = locale.getpreferredencoding(False)
    if "errors" not in kwargs and kwargs.get("text", False):
        kwargs["errors"] = "replace"
    return _orig_run(*args, **kwargs)
subprocess.run = _patched_run

# ============================================================================
# UNICODE OUTPUT FIX — Windows console (cp1252) no puede codificar emojis.
# Reconfiguramos stdout/stderr a UTF-8 en cualquiera de estas situaciones:
#   1. El stream no tiene codificación explícita asignada
#   2. El entorno está configurado para UTF-8 (PYTHONIOENCODING=utf-8)
#   3. Windows moderno con VT100/Jupyter/IDE que soporta UTF-8
# ============================================================================
def _configure_unicode_output() -> None:
    """Reconfigure stdout/stderr para UTF-8 cuando sea seguro."""
    import locale
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        try:
            current_enc = stream.encoding
        except Exception:
            current_enc = None

        # Detectar si el stream puede manejar UTF-8
        can_utf8 = False
        if current_enc and current_enc.lower() in ("utf-8", "utf8", "utf_8"):
            can_utf8 = True
        elif current_enc is None:
            # Stream sin codificación explícita — reconfigurar es seguro
            can_utf8 = True
        elif hasattr(stream, "buffer") and hasattr(stream.buffer, "raw"):
            # Segunda opción: verificar si el raw stream es un ConsoleBytesIO
            # o puede manejar UTF-8
            try:
                raw_name = type(stream.buffer.raw).__name__
                if "Console" not in raw_name:
                    can_utf8 = True
            except Exception:
                pass

        if can_utf8:
            try:
                stream.reconfigure(encoding="utf-8")
            except (ValueError, TypeError, OSError):
                # No se pudo reconfigurar — continuar con la codificación actual
                pass


_configure_unicode_output()

import sounddevice as sd
from google import genai
from google.genai import types
from ui import PipeUIWrapper as PipeUI
from memory.memory_manager import (
    load_memory, update_memory, format_memory_for_prompt,
    should_extract_memory, extract_memory
)

from actions.file_processor import file_processor
from actions.flight_finder     import flight_finder
from actions.open_app          import open_app
from actions.weather_report    import weather_action
from actions.send_message      import send_message
from actions.reminder          import reminder
from actions.computer_settings import computer_settings
from actions.screen_processor  import screen_process
from actions.youtube_video     import youtube_video
from actions.desktop           import desktop_control
from actions.browser_control   import browser_control
from actions.file_controller   import file_controller
from actions.code_helper       import code_helper
from actions.dev_agent         import dev_agent
from actions.web_search        import web_search as web_search_action
from actions.computer_control  import computer_control
from actions.game_updater      import game_updater

from core.tool_registry import TOOL_DECLARATIONS
from core.config_loader import get_gemini_api_key, get_env, BASE_DIR


def get_base_dir():
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent


PROMPT_PATH     = BASE_DIR / "core" / "prompt.txt"
LIVE_MODEL      = "models/gemini-2.5-flash-native-audio-preview-12-2025"
CHANNELS        = 1
SEND_SAMPLE_RATE    = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE          = 1024

# Hybrid VAD: Gemini still detects speech server-side, while this local signal
# finalizes a completed phrase promptly when the microphone goes quiet.
VAD_SPEECH_RMS_THRESHOLD = 500
VAD_END_SILENCE_SECONDS  = 0.85


def _load_system_prompt() -> str:
    try:
        return PROMPT_PATH.read_text(encoding="utf-8")
    except Exception:
        return (
            "You are P.I.P.E, a Personal Intelligent Processing Entity. "
            "Be concise, direct, and always use the provided tools to complete tasks. "
            "Never simulate or guess results — always call the appropriate tool."
        )
   
_last_memory_input = ""

def _update_memory_async(user_text: str, jarvis_text: str) -> None:
    global _last_memory_input

    user_text   = (user_text   or "").strip()
    jarvis_text = (jarvis_text or "").strip()

    if len(user_text) < 5 or user_text == _last_memory_input:
        return
    _last_memory_input = user_text

    try:
        api_key = get_gemini_api_key()
        if not should_extract_memory(user_text, jarvis_text, api_key):
            return
        data = extract_memory(user_text, jarvis_text, api_key)
        if data:
            update_memory(data)
            print(f"[Memory] ✅ {list(data.keys())}")
    except Exception as e:
        if "429" not in str(e):
            print(f"[Memory] ⚠️ {e}")


class PipeLive:

    def __init__(self, ui: PipeUI):
        self.ui             = ui
        self.session        = None
        self.audio_in_queue = None
        self.out_queue      = None
        self._loop          = None
        self._is_speaking   = False
        self._speaking_lock = threading.Lock()
        self._voice_metrics = None
        self._vad_speech_active = False
        self._vad_last_voice_at = None
        self.ui.on_text_command = self._on_text_command

    def _start_voice_metrics(self, source: str) -> None:
        self._voice_metrics = {
            "source": source,
            "started": time.perf_counter(),
            "speech_ended": None,
            "first_transcription": None,
            "first_audio": None,
            "tool_seconds": 0.0,
        }

    def _log_voice_metrics(self) -> None:
        metrics = self._voice_metrics
        if not metrics:
            return
        now = time.perf_counter()
        start = metrics["started"]
        first_audio = metrics["first_audio"]
        stt = metrics["first_transcription"]
        speech_end = metrics["speech_ended"]
        stt_text = f"{stt - start:.2f}s" if stt else "n/a"
        end_text = f"{speech_end - start:.2f}s" if speech_end else "n/a"
        audio_text = f"{first_audio - start:.2f}s" if first_audio else "n/a"
        print(
            f"[VOICE] Source: {metrics['source']} End speech: {end_text} STT: {stt_text} "
            f"First audio: {audio_text} Tool: {metrics['tool_seconds']:.2f}s "
            f"Total: {now - start:.2f}s"
        )
        self._voice_metrics = None

    def _mark_voice_started(self) -> None:
        if not self._voice_metrics:
            self._start_voice_metrics("voice")

    def _mark_voice_ended(self) -> None:
        if self._voice_metrics and self._voice_metrics["speech_ended"] is None:
            self._voice_metrics["speech_ended"] = time.perf_counter()

    def _enqueue_realtime(self, message) -> None:
        if not self.out_queue:
            return
        try:
            self.out_queue.put_nowait(message)
        except asyncio.QueueFull:
            print("[P.I.P.E] ⚠️ Audio queue full; dropping one input block.")

    def _on_text_command(self, text: str):
        if not self._loop or not self.session:
            return
        self._start_voice_metrics("text")
        asyncio.run_coroutine_threadsafe(
            self.session.send_client_content(
                turns={"parts": [{"text": text}]},
                turn_complete=True
            ),
            self._loop
        )

    def set_speaking(self, value: bool):
        with self._speaking_lock:
            self._is_speaking = value
        if value:
            self.ui.set_state("SPEAKING")
        elif not self.ui.muted:
            self.ui.set_state("LISTENING")

    def speak(self, text: str):
        if not self._loop or not self.session:
            return
        asyncio.run_coroutine_threadsafe(
            self.session.send_client_content(
                turns={"parts": [{"text": text}]},
                turn_complete=True
            ),
            self._loop
        )

    def speak_error(self, tool_name: str, error: str):
        short = str(error)[:120]
        self.ui.write_log(f"ERR: {tool_name} — {short}")
        self.speak(f"Sir, {tool_name} encountered an error. {short}")

    def _build_config(self) -> types.LiveConnectConfig:
        from datetime import datetime

        memory     = load_memory()
        mem_str    = format_memory_for_prompt(memory)
        sys_prompt = _load_system_prompt()

        now      = datetime.now()
        time_str = now.strftime("%A, %B %d, %Y — %I:%M %p")
        time_ctx = (
            f"[CURRENT DATE & TIME]\n"
            f"Right now it is: {time_str}\n"
            f"Use this to calculate exact times for reminders.\n\n"
        )

        parts = [time_ctx]
        if mem_str:
            parts.append(mem_str)
        parts.append(sys_prompt)

        return types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            output_audio_transcription={},
            input_audio_transcription={},
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(
                    disabled=False,
                    start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_LOW,
                    end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_LOW,
                    prefix_padding_ms=20,
                    silence_duration_ms=700,
                )
            ),
            system_instruction="\n".join(parts),
            tools=[{"function_declarations": TOOL_DECLARATIONS}],
            session_resumption=types.SessionResumptionConfig(),
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Charon"
                    )
                )
            ),
        )

    async def _execute_tool(self, fc) -> types.FunctionResponse:
        name = fc.name
        args = dict(fc.args or {})
        tool_started = time.perf_counter()

        print(f"[P.I.P.E] 🔧 {name}  {args}")
        self.ui.set_state("THINKING")
        if name == "save_memory":
            category = args.get("category", "notes")
            key      = args.get("key", "")
            value    = args.get("value", "")
            if key and value:
                update_memory({category: {key: {"value": value}}})
                print(f"[Memory] 💾 save_memory: {category}/{key} = {value}")
            if not self.ui.muted:
                self.ui.set_state("LISTENING")
            return types.FunctionResponse(
                id=fc.id, name=name,
                response={"result": "ok", "silent": True}
            )

        loop   = asyncio.get_event_loop()
        result = "Done."

        try:
            if name == "open_app":
                r = await loop.run_in_executor(None, lambda: open_app(parameters=args, response=None, player=self.ui))
                result = r or f"Opened {args.get('app_name')}."

            elif name == "weather_report":
                r = await loop.run_in_executor(None, lambda: weather_action(parameters=args, player=self.ui))
                result = r or "Weather delivered."

            elif name == "browser_control":
                r = await loop.run_in_executor(None, lambda: browser_control(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "file_controller":
                r = await loop.run_in_executor(None, lambda: file_controller(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "send_message":
                r = await loop.run_in_executor(None, lambda: send_message(parameters=args, response=None, player=self.ui, session_memory=None))
                result = r or f"Message sent to {args.get('receiver')}."

            elif name == "reminder":
                r = await loop.run_in_executor(None, lambda: reminder(parameters=args, response=None, player=self.ui))
                result = r or "Reminder set."

            elif name == "youtube_video":
                r = await loop.run_in_executor(None, lambda: youtube_video(parameters=args, response=None, player=self.ui))
                result = r or "Done."

            elif name == "file_processor":
                if not args.get("file_path") and self.ui.current_file:
                    args["file_path"] = self.ui.current_file
                r = await loop.run_in_executor(
                    None,
                    lambda: file_processor(parameters=args, player=self.ui, speak=self.speak)
                )
                result = r or "Done."


            elif name == "screen_process":
                threading.Thread(
                    target=screen_process,
                    kwargs={"parameters": args, "response": None,
                            "player": self.ui, "session_memory": None},
                    daemon=True
                ).start()
                result = "Vision module activated. Stay completely silent — vision module will speak directly."

            elif name == "computer_settings":
                r = await loop.run_in_executor(None, lambda: computer_settings(parameters=args, response=None, player=self.ui))
                result = r or "Done."

            elif name == "desktop_control":
                r = await loop.run_in_executor(None, lambda: desktop_control(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "code_helper":
                r = await loop.run_in_executor(None, lambda: code_helper(parameters=args, player=self.ui, speak=self.speak))
                result = r or "Done."

            elif name == "dev_agent":
                r = await loop.run_in_executor(None, lambda: dev_agent(parameters=args, player=self.ui, speak=self.speak))
                result = r or "Done."

            elif name == "agent_task":
                from agent.task_queue import get_queue, TaskPriority
                priority_map = {"low": TaskPriority.LOW, "normal": TaskPriority.NORMAL, "high": TaskPriority.HIGH}
                priority = priority_map.get(args.get("priority", "normal").lower(), TaskPriority.NORMAL)
                task_id  = get_queue().submit(goal=args.get("goal", ""), priority=priority, speak=self.speak)
                result   = f"Task started (ID: {task_id})."

            elif name == "web_search":
                r = await loop.run_in_executor(None, lambda: web_search_action(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "computer_control":
                r = await loop.run_in_executor(None, lambda: computer_control(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "game_updater":
                r = await loop.run_in_executor(None, lambda: game_updater(parameters=args, player=self.ui, speak=self.speak))
                result = r or "Done."

            elif name == "flight_finder":
                r = await loop.run_in_executor(None, lambda: flight_finder(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "shutdown_pipe":
                self.ui.write_log("SYS: Shutdown requested.")
                self.speak("Goodbye.")

                def _shutdown():
                    import time, sys, os
                    time.sleep(1)
                    os._exit(0)

                threading.Thread(target=_shutdown, daemon=True).start()
            else:
                result = f"Unknown tool: {name}"

        except Exception as e:
            result = f"Tool '{name}' failed: {e}"
            traceback.print_exc()
            self.speak_error(name, e)

        if not self.ui.muted:
            self.ui.set_state("LISTENING")

        print(f"[P.I.P.E] 📤 {name} → {str(result)[:80]}")

        if self._voice_metrics:
            self._voice_metrics["tool_seconds"] += time.perf_counter() - tool_started

        return types.FunctionResponse(
            id=fc.id, name=name,
            response={"result": result}
        )

    async def _send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            if msg == "__audio_stream_end__":
                await self.session.send_realtime_input(audio_stream_end=True)
            else:
                await self.session.send_realtime_input(media=msg)

    async def _listen_audio(self):
        print("[P.I.P.E] 🎤 Mic started")
        loop = asyncio.get_event_loop()

        def callback(indata, frames, time_info, status):
            with self._speaking_lock:
                jarvis_speaking = self._is_speaking
            if jarvis_speaking or self.ui.muted:
                self._vad_speech_active = False
                self._vad_last_voice_at = None
                return

            data = indata.tobytes()
            loop.call_soon_threadsafe(
                self._enqueue_realtime,
                {"data": data, "mime_type": "audio/pcm"}
            )

            samples = memoryview(data).cast("h")
            rms = (sum(sample * sample for sample in samples) / max(len(samples), 1)) ** 0.5
            now = time.perf_counter()
            if rms >= VAD_SPEECH_RMS_THRESHOLD:
                if not self._vad_speech_active:
                    self._vad_speech_active = True
                    loop.call_soon_threadsafe(self._mark_voice_started)
                self._vad_last_voice_at = now
            elif (
                self._vad_speech_active
                and self._vad_last_voice_at is not None
                and now - self._vad_last_voice_at >= VAD_END_SILENCE_SECONDS
            ):
                self._vad_speech_active = False
                self._vad_last_voice_at = None
                loop.call_soon_threadsafe(self._mark_voice_ended)
                loop.call_soon_threadsafe(self._enqueue_realtime, "__audio_stream_end__")

        try:
            with sd.InputStream(
                samplerate=SEND_SAMPLE_RATE,
                channels=CHANNELS,
                dtype="int16",
                blocksize=CHUNK_SIZE,
                callback=callback,
            ):
                print("[P.I.P.E] 🎤 Mic stream open")
                while True:
                    await asyncio.sleep(0.1)
        except Exception as e:
            print(f"[P.I.P.E] ❌ Mic: {e}")
            raise

    async def _receive_audio(self):
        print("[P.I.P.E] 👂 Recv started")
        out_buf, in_buf = [], []

        try:
            while True:
                async for response in self.session.receive():

                    if response.data:
                        if self._voice_metrics and self._voice_metrics["first_audio"] is None:
                            self._voice_metrics["first_audio"] = time.perf_counter()
                        self.audio_in_queue.put_nowait(response.data)

                    if response.server_content:
                        sc = response.server_content

                        if sc.output_transcription and sc.output_transcription.text:
                            self.set_speaking(True)
                            txt = sc.output_transcription.text.strip()
                            if txt:
                                out_buf.append(txt)

                        if sc.input_transcription and sc.input_transcription.text:
                            if self._voice_metrics is None:
                                self._start_voice_metrics("voice")
                            if self._voice_metrics["first_transcription"] is None:
                                self._voice_metrics["first_transcription"] = time.perf_counter()
                            txt = sc.input_transcription.text.strip()
                            if txt:
                                in_buf.append(txt)

                        if sc.turn_complete:
                            self.set_speaking(False)

                            full_in = " ".join(in_buf).strip()
                            if full_in:
                                self.ui.write_log(f"You: {full_in}")
                            in_buf = []

                            full_out = " ".join(out_buf).strip()
                            if full_out:
                                self.ui.write_log(f"Pipe: {full_out}")
                            out_buf = []

                            if full_in and len(full_in) > 5:
                                threading.Thread(
                                    target=_update_memory_async,
                                    args=(full_in, full_out),
                                    daemon=True
                                ).start()

                            if not response.tool_call:
                                self._log_voice_metrics()

                    if response.tool_call:
                        fn_responses = []
                        for fc in response.tool_call.function_calls:
                            print(f"[P.I.P.E] 📞 {fc.name}")
                            fr = await self._execute_tool(fc)
                            fn_responses.append(fr)
                        await self.session.send_tool_response(
                            function_responses=fn_responses
                        )

        except Exception as e:
            print(f"[P.I.P.E] ❌ Recv: {e}")
            traceback.print_exc()
            raise

    async def _play_audio(self):
        print("[P.I.P.E] 🔊 Play started")
        loop = asyncio.get_event_loop()

        stream = sd.RawOutputStream(
            samplerate=RECEIVE_SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=CHUNK_SIZE,
        )
        stream.start()
        try:
            while True:
                chunk = await self.audio_in_queue.get()
                self.set_speaking(True)
                await asyncio.to_thread(stream.write, chunk)
               
                if self.audio_in_queue.empty():
                    self.set_speaking(False)
                   
        except Exception as e:
            print(f"[P.I.P.E] ❌ Play: {e}")
            raise
        finally:
            self.set_speaking(False)
            stream.stop()
            stream.close()

    async def run(self):
        client = genai.Client(
            api_key=get_gemini_api_key(),
            http_options={"api_version": "v1beta"}
        )

        while True:
            try:
                print("[P.I.P.E] 🔌 Connecting...")
                self.ui.set_state("THINKING")
                config = self._build_config()

                async with (
                    client.aio.live.connect(model=LIVE_MODEL, config=config) as session,
                    asyncio.TaskGroup() as tg,
                ):
                    self.session        = session
                    self._loop          = asyncio.get_event_loop()
                    self.audio_in_queue = asyncio.Queue()
                    self.out_queue      = asyncio.Queue(maxsize=10)

                    print("[P.I.P.E] ✅ Connected.")
                    self.ui.set_state("LISTENING")
                    self.ui.write_log("SYS: P.I.P.E online.")

                    tg.create_task(self._send_realtime())
                    tg.create_task(self._listen_audio())
                    tg.create_task(self._receive_audio())
                    tg.create_task(self._play_audio())
                   
            except Exception as e:
                print(f"[P.I.P.E] ⚠️ {e}")
                traceback.print_exc()

            self.set_speaking(False)
            self.ui.set_state("THINKING")
            print("[P.I.P.E] 🔄 Reconnecting in 3s...")
            await asyncio.sleep(3)

def main():
    ui = PipeUI("face.png")

    def runner():
        ui.wait_for_api_key()
        pipe = PipeLive(ui)
        try:
            asyncio.run(pipe.run())
        except KeyboardInterrupt:
            print("\n🔴 Shutting down...")

    threading.Thread(target=runner, daemon=True).start()
    ui.root.mainloop()


if __name__ == "__main__":
    main()