# actions/open_app.py
# MARK XXXIX-OR — Native Windows App Launcher
# Elimina completamente pyautogui. Usa os.startfile, subprocess, webbrowser.

import os
import re
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Alias de aplicaciones comunes -> ejecutables, comandos o protocolos Windows
# ---------------------------------------------------------------------------
_WINDOWS_APP_ALIASES = {
    # --- Protocolos de apps (Microsoft Store / UWP) ---
    "whatsapp":            "whatsapp:",
    "spotify":             "spotify:",
    "discord":             "discord:",
    "telegram":            "telegram:",
    "zoom":                "zoom:",
    "slack":               "slack:",
    "steam":               "steam:",
    "notion":              "notion:",

    # --- Navegadores ---
    "chrome":              "chrome",
    "google chrome":       "chrome",
    "navegador":           "chrome",
    "navegador web":       "chrome",
    "navegador internet":  "chrome",
    "firefox":             "firefox",
    "mozilla firefox":     "firefox",
    "mozilla":             "firefox",
    "edge":                "msedge",
    "microsoft edge":      "msedge",
    "brave":               "brave",
    "brave browser":       "brave",
    "internet explorer":   "iexplore",
    "ie":                  "iexplore",

    # --- Office ---
    "word":                "winword",
    "microsoft word":      "winword",
    "excel":               "excel",
    "microsoft excel":     "excel",
    "powerpoint":          "powerpnt",
    "power point":         "powerpnt",
    "microsoft powerpoint": "powerpnt",
    "outlook":             "outlook",
    "microsoft outlook":   "outlook",
    "outlook de microsoft": "outlook",

    # --- Utilidades del sistema ---
    "notepad":             "notepad.exe",
    "bloc de notas":       "notepad.exe",
    "calculadora":         "calc.exe",
    "calculator":          "calc.exe",
    "terminal":            "cmd.exe",
    "cmd":                 "cmd.exe",
    "explorer":            "explorer.exe",
    "file explorer":       "explorer.exe",
    "paint":               "mspaint.exe",
    "paint 3d":            "mspaint.exe",
    "task manager":        "taskmgr.exe",
    "administrador de tareas": "taskmgr.exe",
    "settings":            "ms-settings:",
    "configuración":       "ms-settings:",
    "powershell":          "powershell.exe",
    "powershell 7":        "pwsh",

    # --- Reproductores ---
    "vlc":                 "vlc",
    "vlc media player":    "vlc",
    "video lantern":       "vlc",
    "video lan":           "vlc",
    "media player":        "wmplayer",
    "windows media player": "wmplayer",

    # --- Apps de producción / diseño ---
    "vscode":              "code",
    "visual studio code":  "code",
    "visual studio":       "devenv",
    "blender":             "blender",
    "blender foundation":  "blender",
    "figma":               "Figma",
    "figma desktop":       "Figma",
    "capcut":              "CapCut",
    "cap cut":             "CapCut",

    # --- Apps de comunicación / trabajo ---
    "skype":               "skype",
    "microsoft skype":     "skype",
    "teams":               "teams",
    "microsoft teams":     "teams",
    "slack":               "slack",
    "signal":              "signal",
    "wechat":              "wechat",
    "kakaotalk":           "KakaoTalk",
    "telegram desktop":    "telegram",
    "postman":             "Postman",
    "postman desktop":     "Postman",

    # --- Otras apps conocidas ---
    "obsidian":            "Obsidian",
    "obsidian app":        "Obsidian",
    "discord":             "Discord",
    "steam":               "steam",
    "steam client":        "steam",
    "spotify desktop":     "spotify",

    # --- Email ---
    "mail":                "mailto:",
    "email":               "mailto:",
    "correo":              "mailto:",
    "correo electrónico":  "mailto:",
}


def _is_url(raw: str) -> bool:
    """Detecta si el texto parece una URL."""
    url_re = re.compile(
        r"^(https?://|www\.|file://|mailto:|ms-edge://|tel:|ftp://)",
        re.IGNORECASE,
    )
    return bool(url_re.match(raw.strip()))


def _normalize_app_name(raw: str) -> str:
    """Devuelve el nombre canónico o el ejecutable correspondiente."""
    key = raw.lower().strip()
    if key in _WINDOWS_APP_ALIASES:
        return _WINDOWS_APP_ALIASES[key]
    for alias, value in _WINDOWS_APP_ALIASES.items():
        if alias in key or key in alias:
            return value
    return raw


def _resolve_executable(app_name: str) -> Optional[Path]:
    """
    Intenta resolver un nombre de aplicación a una ruta de ejecutable existente.
    Busca en PATH, en rutas comunes de Windows con búsqueda recursiva limitada,
    y en carpetas específicas de Office y otras suites.
    """
    name = app_name.strip()
    if not name:
        return None

    # 1. Nombre ya es ruta absoluta o relativa existente
    p = Path(name)
    if p.exists():
        return p

    # 2. Busca en PATH
    exe = shutil.which(name)
    if exe:
        return Path(exe)

    # También busca con extensión .exe explícita en PATH
    exe_with_ext = shutil.which(name + ".exe")
    if exe_with_ext:
        return Path(exe_with_ext)

    # 3. Rutas comunes de Windows + carpetas específicas
    common_dirs = [
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")),
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")),
        Path(os.environ.get("LOCALAPPDATA", r"C:\Users\Public\AppData\Local")),
        Path(os.environ.get("APPDATA", r"C:\Users\Public\AppData\Roaming")),
        Path(r"C:\Windows\System32"),
    ]

    # Carpetas de Office comunes
    office_dirs = []
    for pf in [
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")),
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")),
    ]:
        for office_sub in ["Microsoft Office", "Microsoft Office 16.0", "Office"]:
            office_dirs.append(pf / office_sub)
        office_dirs.append(pf / "Microsoft Office" / "root" / "Office16")
        office_dirs.append(pf / "Microsoft Office 16.0" / "root" / "Office16")

    all_search_dirs = common_dirs + office_dirs

    for base in all_search_dirs:
        if not base.exists():
            continue

        # 3a. Candidatos directos
        candidates = [
            base / name,
            base / f"{name}.exe",
            base / name / "Application.exe",
            base / name / f"{name}.exe",
        ]
        for c in candidates:
            if c.exists():
                return c

        # 3b. Búsqueda recursiva limitada (profundidad 3) por nombre de carpeta o archivo
        try:
            for root, dirs, files in os.walk(str(base)):
                depth = root.replace(str(base), "").count(os.sep)
                if depth > 3:
                    dirs[:] = []
                    continue
                for fname in files:
                    if fname.lower().endswith(".exe"):
                        fbase = fname[:-4]
                        if (
                            fbase.lower() == name.lower()
                            or name.lower() in fbase.lower()
                            or fbase.lower() in name.lower()
                        ):
                            return Path(root) / fname
                for dname in dirs:
                    if dname.lower() == name.lower():
                        for candidate in [
                            Path(root) / dname / f"{name}.exe",
                            Path(root) / dname / f"{dname}.exe",
                            Path(root) / dname / "Application.exe",
                            Path(root) / dname / "launcher.exe",
                        ]:
                            if candidate.exists():
                                return candidate
        except (PermissionError, OSError):
            continue

    # 4. Si empieza por "ms-" (protocolo/URI de MS), no hay ejecutable directo
    if name.lower().startswith("ms-"):
        return None

    return None


def _open_with_os_startfile(path: Path) -> bool:
    """Abre un archivo o carpeta con la asociación nativa de Windows."""
    try:
        os.startfile(str(path))
        return True
    except Exception:
        return False


def _open_with_explorer(path: Path) -> bool:
    """Abre una carpeta o archivo usando explorer explícitamente."""
    try:
        subprocess.run(
            ["explorer", str(path)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return True
    except Exception:
        return False


def _launch_app(exe_path: Path) -> bool:
    """Lanza un ejecutable usando subprocess.Popen (Windows nativo)."""
    try:
        subprocess.Popen(
            [str(exe_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


def _open_url(url: str) -> bool:
    """Abre una URL con el navegador por defecto."""
    try:
        # Normaliza si falta esquema
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
            url = "https://" + url
        return webbrowser.open(url, new=2, autoraise=True)
    except Exception:
        return False


def _is_system_folder(raw: str) -> Optional[Path]:
    """
    Detecta si el nombre se refiere a una carpeta del sistema como
    Escritorio, Descargas, Documentos, Imágenes, Música, Videos, etc.
    Devuelve la ruta resuelta o None.
    """
    key = raw.strip().lower()

    env_map = {
        "escritorio": "Desktop",
        "desktop": "Desktop",
        "downloads": "Downloads",
        "descargas": "Downloads",
        "documentos": "Documents",
        "documents": "Documents",
        "my documents": "Documents",
        "mis documentos": "Documents",
        "imagenes": "Pictures",
        " imágenes": "Pictures",
        "images": "Pictures",
        "picture": "Pictures",
        "pictures": "Pictures",
        "musica": "Music",
        " música": "Music",
        "music": "Music",
        "videos": "Videos",
        "video": "Videos",
        "home": "UserProfile",
    }

    if key in env_map:
        tag = env_map[key]
        if tag == "UserProfile":
            return Path.home()

        shell_known = {
            "Desktop": "Desktop",
            "Downloads": "Downloads",
            "Documents": "Documents",
            "Pictures": "Pictures",
            "Music": "Music",
            "Videos": "Videos",
        }
        fid = shell_known.get(tag)
        if not fid:
            return None
        try:
            import winreg
            key_reg = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders",
            )
            val, _ = winreg.QueryValueEx(key_reg, fid)
            winreg.CloseKey(key_reg)
            p = Path(val)
            if p.exists():
                return p
        except Exception:
            pass

        # Fallback mediante variables de entorno de usuario
        env_var_map = {
            "Desktop": "USERPROFILE",
            "Downloads": "USERPROFILE",
            "Documents": "USERPROFILE",
            "Pictures": "USERPROFILE",
            "Music": "USERPROFILE",
            "Videos": "USERPROFILE",
        }
        base_env = env_var_map.get(tag)
        if base_env:
            base = Path(os.environ.get(base_env, ""))
            sub = {
                "Desktop": "Desktop",
                "Downloads": "Downloads",
                "Documents": "Documents",
                "Pictures": "Pictures",
                "Music": "Music",
                "Videos": "Videos",
            }.get(tag)
            if sub:
                cand = base / sub
                if cand.exists():
                    return cand

    return None


def _search_file_in_common_locations(filename: str) -> Optional[Path]:
    """
    Cuando el usuario menciona un archivo por nombre (p.ej. 'informe.docx',
    'cv.txt') pero no da la ruta completa, buscar en ubicaciones comunes.
    """
    name = filename.strip()
    p = Path(name)
    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p.resolve()

    # Si el nombre incluye extensión, intenta como nombre exacto
    candidates = []

    # Desktop
    desktop = Path.home() / "Desktop"
    if desktop.exists():
        for f in desktop.rglob(p.name):
            candidates.append(f)
        for f in desktop.glob(p.name):
            candidates.append(f)

    # Downloads
    downloads = Path.home() / "Downloads"
    if downloads.exists():
        for f in downloads.rglob(p.name):
            candidates.append(f)
        for f in downloads.glob(p.name):
            candidates.append(f)

    # Documents
    documents = Path.home() / "Documents"
    if documents.exists():
        for f in documents.rglob(p.name):
            candidates.append(f)
        for f in documents.glob(p.name):
            candidates.append(f)

    # Buscar por nombre base + extensiones comunes
    base = p.stem if p.suffix else p.name
    if base != p.name:
        for ext in [".docx", ".doc", ".txt", ".pdf", ".xlsx", ".xls", ".pptx", ".ppt", ".csv", ".json", ".py", ".md", ".jpg", ".png", ".jpeg"]:
            cand_name = base + ext
            if desktop.exists():
                for f in desktop.rglob(cand_name):
                    candidates.append(f)
            if downloads.exists():
                for f in downloads.rglob(cand_name):
                    candidates.append(f)
            if documents.exists():
                for f in documents.rglob(cand_name):
                    candidates.append(f)

    # Quitar duplicados, prioridad por recency
    seen = {}
    for c in candidates:
        if c.exists():
            try:
                mtime = c.stat().st_mtime
            except OSError:
                mtime = 0
            seen[c] = mtime
    if not seen:
        return None
    best = max(seen, key=seen.get)
    return best


def _interpret_and_open(app_name: str) -> str:
    """
    Decide qué hacer con el argumento y lo abre.
    Cubre: URL, archivo/carpeta por ruta, carpeta del sistema, archivo por nombre,
    aplicación conocida, aplicación por ejecutable.
    """
    raw = app_name.strip()
    if not raw:
        return "Please specify which application or file to open, sir."

    norm = _normalize_app_name(raw)

    # 1. URL
    if _is_url(raw) or _is_url(norm):
        url_to_open = norm if _is_url(norm) else raw
        if _open_url(url_to_open):
            return f"Opened {url_to_open} in your browser, sir."
        return f"Could not open {url_to_open}, sir."

    # 2. Carpeta del sistema (Escritorio, Descargas, etc.)
    system_folder = _is_system_folder(raw)
    if system_folder:
        if _open_with_os_startfile(system_folder):
            return f"Opened {system_folder}, sir."
        if _open_with_explorer(system_folder):
            return f"Opened {system_folder}, sir."
        return f"Could not open {system_folder}, sir."

    # 3. Ruta absoluta o relativa existente (archivo o carpeta)
    p = Path(raw)
    if p.is_absolute() or p.exists():
        if p.is_dir():
            if _open_with_os_startfile(p):
                return f"Opened folder {p}, sir."
            if _open_with_explorer(p):
                return f"Opened folder {p}, sir."
            return f"Could not open folder {p}, sir."
        if p.is_file():
            if _open_with_os_startfile(p):
                return f"Opened file {p}, sir."
            if _open_with_explorer(p):
                return f"Opened file {p}, sir."
            return f"Could not open file {p}, sir."

    # 4. Nombre de archivo sin ruta: buscar en ubicaciones comunes
    found = _search_file_in_common_locations(raw)
    if found:
        if _open_with_os_startfile(found):
            return f"Opened {found}, sir."
        if _open_with_explorer(found):
            return f"Opened {found}, sir."
        return f"Found {found}, sir, but could not open it."

    # 5. App conocida (alias -> ejecutable o comando)
    exe = _resolve_executable(norm)
    if exe:
        if _launch_app(exe):
            return f"Opened {raw} successfully, sir."
        return f"Tried to open {raw}, sir, but couldn't launch it."

    # 6. Manejo explícito de protocolos Windows (whatsapp:, spotify:, etc.)
    if norm.endswith(":") or raw.lower().endswith(":"):
        try:
            os.startfile(norm)
            return f"Opened {raw} successfully, sir."
        except Exception:
            pass
        try:
            subprocess.run(
                ["cmd", "/c", "start", "", norm],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            return f"Opened {raw} successfully, sir."
        except Exception:
            pass

    # 7. Último recurso: intentar lanzar directamente con el cmd start
    try:
        subprocess.run(
            ["cmd", "/c", "start", "", raw],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return f"Opened {raw} successfully, sir."
    except Exception:
        pass

    return (
        f"I tried to open {raw}, sir, but couldn't identify what to launch. "
        f"It may not be installed, or the name was ambiguous."
    )


def open_app(
    parameters=None,
    response=None,
    player=None,
    session_memory=None,
) -> str:
    """
    Abre aplicaciones, archivos y carpetas en Windows usando métodos nativos
    (os.startfile, subprocess, webbrowser). No usa pyautogui.

    Parámetros:
        app_name: nombre de la aplicación, archivo, carpeta o URL.
    """
    app_name = (parameters or {}).get("app_name", "").strip()

    if not app_name:
        return "Please specify which application to open, sir."

    print(f"[open_app] 🚀 Opening: {app_name}")

    if player:
        player.write_log(f"[open_app] {app_name}")

    return _interpret_and_open(app_name)
