#!/usr/bin/env python3
"""
run_offline_test.py

Kompletny test całego systemu zbierania danych bez fizycznego drona i bez kamer OptiTrack.
1. Uruchamia symulator OptiTrack (odtwarzanie 3539 rzeczywistych ramek z laboratorium).
2. Uruchamia optitrack_flight_logger.py.
3. Symuluje 10-sekundową misję lotu drona (start -> wznoszenie -> przód -> zawis -> lądowanie).
4. Automatycznie zatrzymuje rejestrator i wywołuje plot_flight_data.py.
5. Generuje wykresy 3D i wykresy czasowe w folderze flight_logs/.

Użycie:
    python3 run_offline_test.py
"""

import sys
import time
import json
import socket
import signal
import subprocess
from pathlib import Path

PYTHON_BIN = sys.executable
SCRIPT_DIR = Path(__file__).parent.resolve()
REPLAY_SCRIPT = SCRIPT_DIR / "dummy_optitrack_replay.py"
LOGGER_SCRIPT = SCRIPT_DIR / "optitrack_flight_logger.py"
PLOT_SCRIPT = SCRIPT_DIR / "plot_flight_data.py"
LOG_DIR = SCRIPT_DIR / "flight_logs"


def main():
    print("=" * 70)
    print("🧪 [TEST SYSTEMU TRACKINGU OFFLINE - BEZ FIZYCZNEGO DRONA]")
    print("=" * 70)

    # 1. Start Replay OptiTrack
    print("\n1️⃣ Uruchamiam symulator OptiTrack (reemisja nagrania z sali @ 100 Hz)...")
    p_replay = subprocess.Popen([
        PYTHON_BIN, str(REPLAY_SCRIPT), "--rate", "100"
    ])

    time.sleep(0.5)

    # 2. Start Rejestratora
    print("2️⃣ Uruchamiam optitrack_flight_logger.py...")
    p_logger = subprocess.Popen([
        PYTHON_BIN, str(LOGGER_SCRIPT), "--output-dir", str(LOG_DIR)
    ])

    time.sleep(1.0)

    # 3. Symulacja misji drona i nadawanie telemetrii
    print("3️⃣ Rozpoczynam symulację lotu i przesyłanie telemetrii gestów...")
    telem_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    mission_timeline = [
        # (czas trwania [s], stan, komenda, gest, conf)
        (1.5, "GROUND", "idle", "none", 0.0),
        (2.0, "TAKING_OFF", "takeoff", "openhand", 0.94),
        (2.0, "FLYING", "hover", "closehand", 0.88),
        (2.5, "FLYING", "forward", "openhand", 0.96),
        (1.5, "FLYING", "hover", "none", 0.0),
        (2.0, "FLYING", "up", "v", 0.91),
        (1.5, "FLYING", "hover", "none", 0.0),
        (2.0, "LANDING", "land", "ok", 0.98),
        (1.0, "LANDED", "idle", "none", 0.0),
    ]

    for duration, state, cmd, gesture, conf in mission_timeline:
        t_end = time.time() + duration
        print(f"   ▶ Stan: {state:<11} | Komenda: {cmd:<9} | Gest: {gesture:<9} ({duration}s)")
        while time.time() < t_end:
            now = time.time()
            payload = {
                "flight_state": state,
                "active_command": cmd,
                "command_timestamp": now,
                "detected_gesture": gesture,
                "gesture_conf": conf,
                "gesture_timestamp": now,
                "window_status": f"{gesture.upper()}: 8/10" if gesture != "none" else "brak",
                "timestamp": now,
            }
            try:
                telem_sock.sendto(json.dumps(payload).encode("utf-8"), ("127.0.0.1", 12112))
            except Exception:
                pass
            time.sleep(0.04)  # ~25 Hz (jak kamera AI-Deck)

    telem_sock.close()

    print("\n4️⃣ Zatrzymuję rejestrator i zamykam pliki...")
    p_logger.send_signal(signal.SIGINT)
    p_logger.wait(timeout=5)

    p_replay.terminate()
    p_replay.wait(timeout=3)

    # 5. Odnalezienie najnowszego pliku CSV
    csv_files = sorted(LOG_DIR.glob("flight_session_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not csv_files:
        print("❌ Nie znaleziono zapisanego pliku CSV!")
        sys.exit(1)

    latest_csv = csv_files[0]
    print(f"\n5️⃣ Generuję wykresy z sesji: {latest_csv.name}...")
    subprocess.run([PYTHON_BIN, str(PLOT_SCRIPT), "--csv", str(latest_csv), "--save"])

    print("\n" + "=" * 70)
    print("🎉 [TEST ZAKOŃCZONY SUKCESEM!]")
    print(f"   • Wygenerowany CSV:     {latest_csv}")
    print(f"   • Wykres trajektorii 3D: {latest_csv.with_suffix('')}_traj3d.png")
    print(f"   • Wykres czasowy osi:   {latest_csv.with_suffix('')}_timeseries.png")
    print(f"   • Wykres gestów:        {latest_csv.with_suffix('')}_gestures.png")
    print("=" * 70)


if __name__ == "__main__":
    main()

