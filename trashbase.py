import sys
import ctypes
import time
import random
import threading
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

import pyperclip
from pynput import keyboard

# --- Définitions de l'API Windows (ctypes) ---

SendInput = ctypes.windll.user32.SendInput
PUL = ctypes.POINTER(ctypes.c_ulong)

class KeyBdInput(ctypes.Structure):
    """Structure représentant une entrée clavier pour l'API Windows."""
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", PUL),
    ]

class HardwareInput(ctypes.Structure):
    """Structure représentant une entrée matérielle générique."""
    _fields_ = [
        ("uMsg", ctypes.c_ulong),
        ("wParamL", ctypes.c_short),
        ("wParamH", ctypes.c_ushort),
    ]

class MouseInput(ctypes.Structure):
    """Structure représentant une entrée souris."""
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", PUL),
    ]

class Input_I(ctypes.Union):
    """Union des différents types d'entrées possibles."""
    _fields_ = [("ki", KeyBdInput), ("mi", MouseInput), ("hi", HardwareInput)]

class Input(ctypes.Structure):
    """Structure principale envoyée à la fonction SendInput de Windows."""
    _fields_ = [("type", ctypes.c_ulong), ("ii", Input_I)]

# --- Variables Globales et Configuration ---

DEBOUNCE_SEC: float = 0.15
TAUNTS_FILE: str = "taunts.txt"
TEXT_MAP: Dict[str, str] = {
    "+": "男的来了女的来了... [Texte tronqué]",
    "*": "유럽 ​​대륙을 떠도는 유령...",
    "/": "프롤레타리아트가 계급으로...",
}

_hotkey_listener: Optional[keyboard.Listener] = None
_hotkey_thread: Optional[threading.Thread] = None
_hotkey_stop: threading.Event = threading.Event()
last_press_time: float = 0.0
lock: threading.Lock = threading.Lock()

# --- Fonctions de Manipulation du Clavier ---

def press_enter() -> None:
    """Simule l'appui et le relâchement de la touche Entrée."""
    ctypes.windll.user32.keybd_event(0x0D, 0, 0, 0)  # Enter enfoncé
    ctypes.windll.user32.keybd_event(0x0D, 0, 2, 0)  # Enter relâché

def send_unicode_char(char: str) -> None:
    """Envoie un caractère Unicode unique via l'API SendInput.

    Args:
        char: Le caractère à envoyer.
    """
    # KEYEVENTF_UNICODE = 0x0004
    uni_input = Input(type=1, ii=Input_I())
    uni_input.ii.ki = KeyBdInput(
        0, ord(char), 0x0004, 0, ctypes.pointer(ctypes.c_ulong(0))
    )
    SendInput(1, ctypes.pointer(uni_input), ctypes.sizeof(uni_input))

    # KEYEVENTF_KEYUP = 0x0002 | KEYEVENTF_UNICODE = 0x0004 -> 0x0006
    uni_input.ii.ki.dwFlags = 0x0006
    SendInput(1, ctypes.pointer(uni_input), ctypes.sizeof(uni_input))

def send_text_to_game(text: str) -> None:
    """Ouvre le chat du jeu, tape le texte et l'envoie.

    Args:
        text: La chaîne de caractères à saisir dans le jeu.
    """
    time.sleep(0.3)
    press_enter()  # Ouvre la boîte de dialogue
    time.sleep(0.1)

    for ch in text:
        send_unicode_char(ch)
        time.sleep(0.003)  # Délai minimal pour éviter la saturation du buffer

    time.sleep(0.1)
    press_enter()  # Envoie le message
    print(f"[{time.strftime('%H:%M:%S')}] Envoyé : {text[:30]}...")

# --- Logique de Chargement et de Gestion ---

def load_taunts(path: str = TAUNTS_FILE) -> List[str]:
    """Charge les phrases de provocation depuis un fichier texte.

    Args:
        path: Chemin vers le fichier .txt.

    Returns:
        Une liste de phrases. Renvoie une phrase par défaut si le fichier est vide ou absent.
    """
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if lines:
            return lines
    return ["Fichier taunts.txt manquant ou vide."]

def on_press(key: keyboard.KeyCode) -> Optional[bool]:
    """Gère les événements de pression de touche avec anti-rebond.

    Args:
        key: La touche pressée détectée par le listener.

    Returns:
        False si la touche Esc est pressée pour arrêter le listener, sinon None.
    """
    global last_press_time
    now = time.time()

    with lock:
        if now - last_press_time < DEBOUNCE_SEC:
            return
        last_press_time = now

    try:
        if hasattr(key, "char"):
            msg: Optional[str] = None
            
            if key.char == "-":
                msg = random.choice(taunts)
            elif key.char in TEXT_MAP:
                msg = TEXT_MAP[key.char]

            if msg:
                pyperclip.copy(msg)
                send_text_to_game(msg)
                return

    except AttributeError:
        pass

    if key == keyboard.Key.esc:
        print("Arrêt du programme détecté (Esc).")
        return False

# --- Point d'entrée ---

def main() -> None:
    """Lance le listener de clavier principal."""
    global taunts
    taunts = load_taunts()
    
    print("=== Listener Actif ===")
    print("Touches : '-' pour taunts aléatoires, '+', '*', '/' pour textes prédéfinis.")
    print("Appuyez sur 'Esc' pour quitter.")
    
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

if __name__ == "__main__":
    main()