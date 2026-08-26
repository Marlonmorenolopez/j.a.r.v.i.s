"""
file_processor.py — P.I.P.E Universal File Processor (Migrado a google.genai SDK)
"""

import os
import re
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime

from google import genai
from google.genai import types


def _get_api_key() -> str:
    config_path = Path(__file__).resolve().parent.parent / "config" / "api_keys.json"
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)["gemini_api_key"]


def _get_client():
    return genai.Client(api_key=_get_api_key())


def _detect_type(path: Path) -> str:
    ext = path.suffix.lower().lstrip(".")
    image_exts = {"jpg", "jpeg", "png", "gif", "webp", "bmp", "tiff", "svg", "ico"}
    video_exts = {"mp4", "avi", "mov", "mkv", "wmv", "flv", "webm", "m4v", "3gp"}
    audio_exts = {"mp3", "wav", "ogg", "m4a", "aac", "flac", "wma", "opus"}
    code_exts  = {"py", "js", "ts", "jsx", "tsx", "html", "css", "java", "c",
                  "cpp", "cs", "go", "rs", "rb", "php", "swift", "kt", "sh",
                  "bash", "ps1", "lua", "r", "m", "sql", "yaml", "toml"}
    archive_exts = {"zip", "rar", "tar", "gz", "7z", "bz2", "xz"}

    if ext in image_exts:    return "image"
    if ext in video_exts:    return "video"
    if ext in audio_exts:    return "audio"
    if ext in code_exts:     return "code"
    if ext in archive_exts:  return "archive"
    if ext == "pdf":         return "pdf"
    if ext in ("docx", "doc"): return "docx"
    if ext in ("txt", "md", "rst", "log"): return "text"
    if ext in ("csv", "tsv"): return "csv"
    if ext in ("xlsx", "xls", "ods"): return "excel"
    if ext == "json":        return "json"
    if ext == "xml":         return "xml"
    if ext in ("pptx", "ppt"): return "pptx"
    return "unknown"


def _file_size_str(path: Path) -> str:
    size = path.stat().st_size
    if size < 1024:        return f"{size} B"
    if size < 1024**2:     return f"{size/1024:.1f} KB"
    if size < 1024**3:     return f"{size/1024**2:.1f} MB"
    return f"{size/1024**3:.1f} GB"


def _output_path(src: Path, suffix: str, new_ext: str = None) -> Path:
    ext  = new_ext or src.suffix
    name = f"{src.stem}_{suffix}{ext}"
    return src.parent / name


def _process_image(path: Path, action: str, params: dict, speak=None) -> str:
    action = action or "describe"

    if action in ("describe", "ocr", "analyze", "read", "extract_text"):
        try:
            client = _get_client()
            uploaded = client.files.upload(file=path)
            
            prompt = {
                "describe": "Describe esta imagen en detalle.",
                "ocr":      "Extrae todo el texto visible en esta imagen de forma estructurada.",
                "analyze":  "Analiza esta imagen minuciosamente: objetos, texto, contexto.",
                "read":     "Lee todo el texto en esta imagen manteniendo el formato original.",
                "extract_text": "Transcripción completa del texto de esta imagen.",
            }.get(action, "Describe esta imagen.")

            if params.get("instruction"):
                prompt = params["instruction"]

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[uploaded, prompt]
            )
            result = response.text.strip()

            if len(result) > 600 and params.get("save", True):
                out = _output_path(path, "result", ".txt")
                out.write_text(result, encoding="utf-8")
                return f"{result[:400]}...\n\nResultado completo guardado en: {out.name}"
            return result
        except Exception as e:
            return f"Error al analizar la imagen con IA: {e}"

    # Manipulaciones locales con PIL
    try:
        from PIL import Image
    except ImportError:
        return "Pillow no está instalado. Ejecuta: pip install Pillow"

    if action == "resize":
        width  = int(params.get("width", 0))
        height = int(params.get("height", 0))
        scale  = float(params.get("scale", 0))
        try:
            img = Image.open(path)
            w, h = img.size
            if scale:
                new_size = (int(w * scale), int(h * scale))
            elif width and height:
                new_size = (width, height)
            elif width:
                new_size = (width, int(h * width / w))
            elif height:
                new_size = (int(w * height / h), height)
            else:
                return "Especifica ancho, alto o escala."
            out = _output_path(path, f"resized_{new_size[0]}x{new_size[1]}")
            img.resize(new_size, Image.LANCZOS).save(out)
            return f"Imagen redimensionada de {w}x{h} a {new_size[0]}x{new_size[1]}. Guardado: {out.name}"
        except Exception as e:
            return f"Fallo al redimensionar: {e}"

    if action == "convert":
        fmt = params.get("format", "png").lower().strip(".")
        try:
            img = Image.open(path).convert("RGB") if fmt in ("jpg", "jpeg") else Image.open(path)
            out = _output_path(path, "converted", f".{fmt}")
            img.save(out)
            return f"Convertido a {fmt.upper()}. Guardado: {out.name}"
        except Exception as e:
            return f"Fallo al convertir: {e}"

    return f"Acción de imagen no reconocida: {action}"


def _process_pdf(path: Path, action: str, params: dict, speak=None) -> str:
    action = action or "summarize"
    client = _get_client()

    # Intentar extracción nativa de texto primero
    text = ""
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"
    except Exception:
        pass

    # Si no tiene texto plano (PDF escaneado/imágenes), subir el archivo directamente a Gemini OCR
    if not text.strip():
        try:
            uploaded = client.files.upload(file=path)
            prompt = "Analiza y lee todo el contenido de este archivo PDF, incluyendo imágenes o texto escaneado."
            if action == "summarize":
                prompt = "Resume detalladamente el contenido de este documento PDF."
            elif action == "extract_text":
                prompt = "Extrae y transcribe todo el texto visible en este PDF."

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[uploaded, prompt]
            )
            return response.text.strip()
        except Exception as e:
            return f"Fallo en el procesamiento visión del PDF: {e}"

    if action == "extract_text":
        out = _output_path(path, "text", ".txt")
        out.write_text(text, encoding="utf-8")
        return f"Texto extraído ({len(text)} caracteres). Guardado: {out.name}"

    prompt_map = {
        "summarize": f"Resume este documento PDF de forma completa y clara:\n\n{text[:50000]}",
        "analyze":   f"Analiza minuciosamente este documento:\n\n{text[:50000]}",
        "reformat":  f"Reestructura este texto con formato limpio:\n\n{text[:50000]}",
    }
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_map.get(action, f"Analiza el siguiente texto:\n\n{text[:50000]}")
        )
        result = response.text.strip()
        if len(result) > 600 and params.get("save", True):
            out = _output_path(path, action, ".txt")
            out.write_text(result, encoding="utf-8")
            return f"{result[:400]}...\n\nResultado completo guardado: {out.name}"
        return result
    except Exception as e:
        return f"Fallo en el análisis IA del PDF: {e}"


def _process_text_doc(path: Path, file_type: str, action: str, params: dict, speak=None) -> str:
    action = action or "summarize"

    def _read_content() -> str:
        if file_type == "docx":
            try:
                import docx
                doc = docx.Document(path)
                return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            except ImportError:
                return "Librería python-docx no instalada. Ejecuta: pip install python-docx"
            except Exception as e:
                return f"Error al leer DOCX: {e}"
        return path.read_text(encoding="utf-8", errors="ignore")

    content = _read_content()
    if not content.strip():
        return "El archivo está vacío o no se pudo extraer texto."

    if action == "word_count":
        words = len(content.split())
        chars = len(content)
        lines = content.count("\n")
        return f"Conteo: {words} palabras, {chars} caracteres, {lines} líneas."

    if action == "extract_text":
        if file_type != "txt":
            out = _output_path(path, "extracted", ".txt")
            out.write_text(content, encoding="utf-8")
            return f"Texto extraído. Guardado: {out.name}"
        return content[:3000]

    client = _get_client()
    prompt = f"Procesa el siguiente documento ({action}):\n\n{content[:50000]}"

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        return f"Error procesando documento: {e}"


def _process_data(path: Path, file_type: str, action: str, params: dict, speak=None) -> str:
    try:
        import pandas as pd
    except ImportError:
        return "pandas no instalado. Ejecuta: pip install pandas openpyxl"

    action = action or "analyze"
    try:
        df = pd.read_csv(path) if file_type == "csv" else pd.read_excel(path)
    except Exception as e:
        return f"Error leyendo archivo de datos: {e}"

    if action == "info":
        return f"Filas: {len(df)}, Columnas: {len(df.columns)}\nColumnas: {', '.join(df.columns)}"

    client = _get_client()
    preview = df.head(50).to_string()
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"Analiza este conjunto de datos:\nColumnas: {list(df.columns)}\nVista previa:\n{preview}"
        )
        return response.text.strip()
    except Exception as e:
        return f"Error analizando datos: {e}"


def file_processor(parameters: dict, player=None, speak=None) -> str:
    file_path_str = parameters.get("file_path", "").strip()
    if not file_path_str:
        return "No se proporcionó la ruta del archivo."

    path = Path(file_path_str)
    if not path.exists():
        return f"Archivo no encontrado: {file_path_str}"

    file_type = _detect_type(path)
    action    = (parameters.get("action") or "").lower().strip()
    params    = {**parameters, "instruction": parameters.get("instruction", "")}

    log_msg = f"[FileProcessor] {file_type.upper()} | {path.name} | action={action or 'auto'}"
    print(log_msg)
    if player:
        player.write_log(log_msg)

    dispatch = {
        "image": _process_image,
        "pdf":   _process_pdf,
        "docx":  lambda p, a, pm, s: _process_text_doc(p, "docx", a, pm, s),
        "text":  lambda p, a, pm, s: _process_text_doc(p, "text", a, pm, s),
        "csv":   lambda p, a, pm, s: _process_data(p, "csv", a, pm, s),
        "excel": lambda p, a, pm, s: _process_data(p, "excel", a, pm, s),
    }

    handler = dispatch.get(file_type)
    if not handler:
        return f"Tipo de archivo no soportado directamente: {file_type}"

    try:
        result = handler(path, action, params, speak)
        return result or "Completado."
    except Exception as e:
        return f"Error general procesando archivo: {e}"