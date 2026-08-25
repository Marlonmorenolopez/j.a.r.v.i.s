import ast
import json
import re
import sys
import threading
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Callable

from agent.planner_new      import create_plan, replan
from agent.error_handler import analyze_error, generate_fix, ErrorDecision
from agent.action_dispatcher import dispatch_tool
from core.config_loader import get_gemini_api_key, get_base_dir


BASE_DIR = get_base_dir()


def _extract_imports_from_code(source: str) -> list[str]:
    """Extrae nombres de módulos importados directamente del AST del código."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []  # código inválido → se rechazará más tarde
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module.split(".")[0])
    return imports


ALLOWED_TOP_LEVEL_MODULES = frozenset({
    "math", "random", "statistics", "datetime", "time",
    "re", "string", "textwrap", "collections",
    "itertools", "functools", "copy", "json", "csv",
    "pathlib", "typing", "difflib", "heapq", "bisect",
    "abc", "contextlib", "dataclasses", "enum", "hashlib",
    "urllib.parse", "html", "xml.etree.ElementTree",
    "pprint", "logging", "_thread", "threading",
})


def _validate_imports(imports: list[str]) -> bool:
    return all(mod.split(".")[0] in ALLOWED_TOP_LEVEL_MODULES for mod in imports)


def _inject_context(params: dict, tool: str, step_results: dict, goal: str = "") -> dict:
    """Replace placeholders like {{step_1}} with actual results."""
    if not params:
        return params
    result = {}
    for k, v in params.items():
        if isinstance(v, str):
            # Replace {{step_N}} with step_results[N]
            for step_num, step_result in step_results.items():
                placeholder = f"{{{{step_{step_num}}}}}"
                if placeholder in v:
                    v = v.replace(placeholder, str(step_result))
        result[k] = v
    return result


def _run_generated_code(description: str, speak: Callable | None = None) -> str:
    """Generate, validate, save, and execute Python code."""
    try:
        import google.generativeai as genai
    except ImportError:
        return "google-generativeai not installed. Cannot generate code."

    genai.configure(api_key=get_gemini_api_key())
    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = f"""Write a Python script to accomplish this task:
{description}

Rules:
- Use ONLY standard library modules from this list: {', '.join(sorted(ALLOWED_TOP_LEVEL_MODULES))}
- No external dependencies (no requests, no numpy, no pandas, etc.)
- No file I/O outside current directory
- No network access
- Print the final result to stdout
- No markdown, no explanations, just the code
"""
    response = model.generate_content(prompt)
    code = response.text.strip()

    # Remove markdown fences if present
    if code.startswith("```"):
        code = "\n".join(code.split("\n")[1:-1])

    imports = _extract_imports_from_code(code)
    if not _validate_imports(imports):
        return f"Generated code uses disallowed imports: {imports}"

    # Save to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        temp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=BASE_DIR,
        )
        output = result.stdout.strip()
        error = result.stderr.strip()
        if error:
            return f"Code executed with errors:\n{error}\n\nOutput:\n{output}"
        return output or "Code executed successfully (no output)."
    except subprocess.TimeoutExpired:
        return "Code execution timed out (30s)."
    except Exception as e:
        return f"Code execution failed: {e}"
    finally:
        try:
            os.unlink(temp_path)
        except:
            pass


def _translate_content(content: str, target_lang: str = "en") -> str:
    """Translate content using Gemini."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=get_gemini_api_key())
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = f"Translate the following text to {target_lang}. Return ONLY the translation, no explanations:\n\n{content}"
        response = model.generate_content(prompt)
        translated = response.text.strip()
        print(f"[Executor] ✅ Translation done ({target_lang})")
        return translated
    except Exception as e:
        print(f"[Executor] ⚠️ Translation failed: {e}")
        return content


def _call_tool(tool: str, parameters: dict, speak: Callable | None) -> str:
    """Dispatch a tool call using the action dispatcher."""
    try:
        return dispatch_tool(tool, parameters, player=None, speak=speak)
    except ValueError as e:
        # Tool not found in dispatch table
        print(f"[Executor] ⚠️ Tool '{tool}' not registered in dispatcher: {e}")
        return f"Tool '{tool}' not implemented."
    except Exception as e:
        print(f"[Executor] ❌ Tool '{tool}' failed: {e}")
        raise


class AgentExecutor:

    MAX_REPLAN_ATTEMPTS = 2

    def execute(
        self,
        goal:        str,
        speak:       Callable | None        = None,
        cancel_flag: threading.Event | None = None,
    ) -> str:
        print(f"\n[Executor] 🎯 Goal: {goal}")

        replan_attempts = 0
        completed_steps = []
        step_results    = {} 
        plan            = create_plan(goal)

        while True:
            steps = plan.steps

            if not steps:
                msg = "I couldn't create a valid plan for this task, sir."
                if speak: speak(msg)
                return msg

            success      = True
            failed_step  = None
            failed_error = ""

            for i, step in enumerate(steps):
                if cancel_flag and cancel_flag.is_set():
                    if speak: speak("Task cancelled, sir.")
                    return "Task cancelled."

                step_num = i + 1
                tool     = step.tool_name
                desc     = step.description or step.capability_id
                params   = step.parameters

                params = _inject_context(params, tool, step_results, goal=goal)

                print(f"\n[Executor] ▶️ Step {step_num}: [{tool}] {desc}")

                attempt = 1
                step_ok = False

                while attempt <= 3:
                    if cancel_flag and cancel_flag.is_set():
                        break
                    try:
                        result = _call_tool(tool, params, speak)
                        step_results[step_num] = result 
                        completed_steps.append(step)
                        print(f"[Executor] ✅ Step {step_num} done: {str(result)[:100]}")
                        step_ok = True
                        break

                    except Exception as e:
                        error_msg = str(e)
                        print(f"[Executor] ❌ Step {step_num} attempt {attempt} failed: {error_msg}")

                        recovery = analyze_error(step, error_msg, attempt=attempt)
                        decision = recovery["decision"]
                        user_msg = recovery.get("user_message", "")

                        if speak and user_msg:
                            speak(user_msg)

                        if decision == ErrorDecision.RETRY:
                            attempt += 1
                            import time; time.sleep(2)
                            continue

                        elif decision == ErrorDecision.SKIP:
                            print(f"[Executor] ⏭️ Skipping step {step_num}")
                            completed_steps.append(step)
                            step_ok = True
                            break

                        elif decision == ErrorDecision.ABORT:
                            msg = f"Task aborted, sir. {recovery.get('reason', '')}"
                            if speak: speak(msg)
                            return msg

                        else:
                            fix_suggestion = recovery.get("fix_suggestion", "")
                            if fix_suggestion and tool != "generated_code":
                                try:
                                    fixed_step_dict = generate_fix(step, error_msg, fix_suggestion)
                                    if speak: speak("Trying an alternative approach, sir.")
                                    res = _call_tool(
                                        fixed_step_dict["tool"],
                                        fixed_step_dict["parameters"],
                                        speak
                                    )
                                    step_results[step_num] = res
                                    completed_steps.append(step)
                                    step_ok = True
                                    break
                                except Exception as fix_err:
                                    print(f"[Executor] ⚠️ Fix failed: {fix_err}")

                            failed_step  = step
                            failed_error = error_msg
                            success      = False
                            break

                if not step_ok and not failed_step:
                    failed_step  = step
                    failed_error = "Max retries exceeded"
                    success      = False

                if not success:
                    break

            if success:
                return self._summarize(goal, completed_steps, speak)

            if replan_attempts >= self.MAX_REPLAN_ATTEMPTS:
                msg = f"Task failed after {replan_attempts} replan attempts, sir."
                if speak: speak(msg)
                return msg

            print(f"[Executor] 🔄 Replanning (attempt {replan_attempts + 1})...")
            replan_attempts += 1

            failed_desc = getattr(failed_step, 'description', '') or getattr(failed_step, 'capability_id', '')
            plan = replan(
                goal,
                {"tool": getattr(failed_step, 'tool_name', ''), "parameters": getattr(failed_step, 'parameters', {})},
                failed_error,
                step_results,
            )

    def _summarize(self, goal: str, completed_steps: list, speak: Callable | None) -> str:
        """Generate a summary of completed task."""
        summary = f"Task completed: {goal}\n"
        for i, step in enumerate(completed_steps, 1):
            desc = getattr(step, 'description', '') or getattr(step, 'capability_id', '')
            summary += f"  {i}. {desc}\n"
        if speak:
            speak(summary)
        return summary


if __name__ == "__main__":
    # Quick test
    executor = AgentExecutor()
    result = executor.execute("qué hora es", speak=None)
    print(f"\nFinal result: {result}")