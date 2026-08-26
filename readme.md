# P.I.P.E — Personal Intelligent Processing Entity

## Core Features

| Feature | Description |
| :--- | :--- |
| 🎙️ **Real-time Voice** | Ultra-low latency conversation with multilingual support. |
| 🖥️ **System Control** | Launch applications, manage files, and execute terminal commands. |
| 🧩 **Autonomous Tasks** | High-level planning to break down and solve complex, multi-step goals. |
| 👁️ **Visual Awareness** | Real-time screen processing and webcam computer vision. |
| 🧠 **Persistent Memory** | Deep capability to remember your projects, preferences, and personal context. |
| ⌨️ **Hybrid Input** | Seamlessly switch between keyboard typing and voice interaction. |

---

## What's New in P.I.P.E

* 📂 **Advanced File Handling:** Direct support for file uploads and document processing. Drop PDFs, source code, or images into the assistant to have them analyzed, summarized, or edited instantly.
* 🎨 **Adaptive & Flexible UI:** A complete overhaul of the interface. The new UI is fully resizable and responsive, featuring transparency controls and customizable layouts to fit your workspace perfectly.
* 🐧🍎 **Refined Cross-Platform Stability:** Critical fixes and compatibility enhancements for macOS and Linux, making core system actions highly consistent across all major operating systems.
* ⚡ **Optimized Core Engine:** Significant performance boost in tool-calling logic and response generation, resulting in a **40% faster** interaction speed.
* 🔀 **OpenRouter Integration:** Selected action modules (web search, memory, desktop control, and more) route their LLM calls through OpenRouter's free-tier models. This massively increases effective request limits without extra costs, while Gemini Live continues handling real-time voice and tool-calling.

---

## Architecture

P.I.P.E is built on a modular architecture:

```
P.I.P.E/
├── main.py                 # Entry point — initializes UI + voice pipeline
├── ui.py                   # PyQt6 interface (HUD, panels, controls)
├── core/                   # Core systems
│   ├── capability_registry.py  # What P.I.P.E can do (routing, permissions)
│   ├── event_system.py         # Pub/sub event bus for core events
│   ├── tool_registry.py        # Function declarations for Gemini Live
│   ├── config_loader.py        # .env / config management
│   ├── prompt.txt              # System prompt for the LLM
│   └── action_executor.py      # Tool execution & verification
├── agent/                  # Agentic layer
│   ├── planner_new.py          # Goal → Plan → Steps decomposition
│   ├── pipe_context.py         # Conversation context engine
│   ├── intent_router.py        # User input → capability routing
│   ├── action_resolver.py      # Capability → tool mapping
│   ├── permission_manager.py   # Confirmation gating
│   └── security_gate.py        # Risk evaluation & blocking
├── actions/                # Concrete tool implementations
├── voice/                  # Voice pipeline (FASE 4-5)
│   ├── capture.py              # FFmpeg mic capture + VAD
│   ├── stt.py                  # Faster-Whisper STT
│   ├── tts.py                  # Edge TTS
│   ├── pipeline.py             # Integrated voice pipeline
│   └── ui_bridge.py            # Voice ↔ UI + Agent integration
├── memory/                 # Persistent memory (SQLite + embeddings)
├── config/                 # JSON configs (capabilities, tools)
├── tests_*.py              # Regression tests (B1-B7, C, FASE 4-5)
└── requirements.txt
```

---

## Quick Start

```bash
# 1. Clone and enter
cd P.I.P.E

# 2. Create venv and install
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 3. Configure API keys (copy .env.example to .env and fill in)
#    Required: GEMINI_API_KEY
#    Optional: OPENROUTER_API_KEY, etc.

# 4. Run
./Iniciar P.I.P.E.cmd
# or
.venv\Scripts\python.exe main.py
```

---

## Voice Pipeline (FASE 4-5)

| Component | Technology | Latency Target |
|-----------|------------|----------------|
| Capture   | FFmpeg dshow + WebRTC VAD | < 50ms |
| STT       | Faster-Whisper (tiny/base/small/medium) | 200-800ms |
| TTS       | Edge TTS (neural) | 150-400ms |
| **Total** | **End-to-end** | **~500-1500ms** |

The voice pipeline integrates with the EventBus for real-time metrics and UI state sync.

---

## Testing

Run full regression suite:

```bash
python -m pytest tests_B1_capability_registry.py -v
python -m pytest tests_B2_intent_router.py -v
python -m pytest tests_B3_action_resolver.py -v
python -m pytest tests_B5_permission_manager.py -v
python -m pytest tests_B6_security_gate.py -v
python -m pytest tests_B7_event_system.py -v
python -m pytest tests_C_planner.py -v
python -m pytest tests_fase4_voice_pipeline.py -v
python -m pytest tests_fase5_voice_ui.py -v
```

All 100+ tests should pass.

---

## Configuration

Key files in `config/`:
- `capabilities.json` — What P.I.P.E can do, risk levels, verification
- `tools.json` — Function declarations for Gemini Live

Key env vars in `.env`:
- `GEMINI_API_KEY` — Required for Gemini Live
- `OPENROUTER_API_KEY` — Optional, for action modules
- `OPENROUTER_MODEL` — Default: `meta-llama/llama-3.1-8b-instruct:free`

---

## License

Proprietary — FatihMakes Industries