"""Small OS helpers shared by the top-level scripts, so they run on Windows and macOS (Linux best effort).

  venv_python(root)   the project venv's interpreter (.venv\\Scripts\\python.exe or .venv/bin/python), else this one
  free_ram_gb()       available system RAM in GiB (None if unknown)
  power()             (on_ac, battery_pct); pct 255 = no battery / unknown, like Windows' GetSystemPowerStatus
  open_path(p)        open a file or folder with the default app
  POPEN_KW            subprocess flags for a background child (no console window / own process group)
  kill_tree(proc)     stop a child started with POPEN_KW and everything it spawned
"""
import ctypes
import os
import re
import signal
import subprocess
import sys

WIN = os.name == "nt"
MAC = sys.platform == "darwin"


def venv_python(root, fallback=sys.executable):
    p = os.path.join(root, ".venv", "Scripts", "python.exe") if WIN else os.path.join(root, ".venv", "bin", "python")
    return p if os.path.exists(p) else fallback


def free_ram_gb():
    if WIN:
        class MS(ctypes.Structure):
            _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [(n, ctypes.c_ulonglong) for n in ("tp", "ap", "tpf", "apf", "tv", "av", "ae")]
        ms = MS(); ms.l = ctypes.sizeof(ms); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
        return ms.ap / 2**30
    try:
        import psutil
        return psutil.virtual_memory().available / 2**30
    except ImportError:
        return None


def power():
    if WIN:
        class SPS(ctypes.Structure):
            _fields_ = [("ac", ctypes.c_ubyte), ("flag", ctypes.c_ubyte), ("pct", ctypes.c_ubyte), ("r", ctypes.c_ubyte), ("life", ctypes.c_ulong), ("full", ctypes.c_ulong)]
        s = SPS(); ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s))
        return s.ac == 1, s.pct
    if MAC:
        try:
            out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=10).stdout
        except (OSError, subprocess.SubprocessError):
            return True, 255
        m = re.search(r"(\d+)%", out)
        return "AC Power" in out or m is None, int(m[1]) if m else 255     # desktop Macs: no battery line
    try:
        import psutil
        b = psutil.sensors_battery()
        return (True, 255) if b is None else (bool(b.power_plugged), int(b.percent))
    except (ImportError, AttributeError):
        return True, 255


def open_path(p):
    if WIN:
        os.startfile(p)
    else:
        subprocess.Popen(["open" if MAC else "xdg-open", p])


POPEN_KW = {"creationflags": 0x08000000} if WIN else {"start_new_session": True}   # CREATE_NO_WINDOW / own process group


def kill_tree(proc):
    if WIN:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    else:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
