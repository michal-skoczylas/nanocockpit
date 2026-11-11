#!/usr/bin/env python3
"""
dummy_optitrack_replay.py

Symulator / odtwarzacz nagranych danych OptiTrack (np. logger_QDrone.txt) przez UDP.
Umożliwia testowanie optitrack_flight_logger oraz gesture_flight_controller
offline bez konieczności przebywania w laboratorium.
"""

import sys
import time
import socket
import argparse
from pathlib import Path

DEFAULT_LOG = "/Users/michalskoczylas/Downloads/OT/logger_QDrone.txt"


def main():
    parser = argparse.ArgumentParser(description="Odtwarzacz pakietów OptiTrack UDP z pliku tekstowego")
    parser.add_argument("--file", default=DEFAULT_LOG, help=f"Ścieżka do pliku z logami OptiTrack (domyślnie {DEFAULT_LOG})")
    parser.add_argument("--host", default="127.0.0.1", help="Adres docelowy (domyślnie 127.0.0.1 lub 255.255.255.255 dla broadcast)")
    parser.add_argument("--port", type=int, default=12111, help="Port docelowy (domyślnie 12111)")
    parser.add_argument("--rate", type=float, default=200.0, help="Częstotliwość nadawania w Hz (domyślnie 200 Hz)")
    parser.add_argument("--loop", action="store_true", default=True, help="Odtwarzaj w pętli nieskończonej")
    args = parser.parse_args()

    log_path = Path(args.file).resolve()
    if not log_path.exists():
        print(f"❌ Plik logów nie istnieje: {log_path}")
        sys.exit(1)

    print(f"📂 Wczytywanie ramek z: {log_path}...")
    with open(log_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    total_frames = len(lines)
    print(f"✅ Wczytano {total_frames} ramek OptiTrack.")
    print(f"📡 Nadawanie UDP na {args.host}:{args.port} z częstotliwością {args.rate:.0f} Hz...")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    if args.host.endswith(".255"):
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    period = 1.0 / max(1.0, args.rate)
    idx = 0

    try:
        while True:
            t_start = time.time()
            data = lines[idx].encode("utf-8")
            sock.sendto(data, (args.host, args.port))

            idx += 1
            if idx >= total_frames:
                if args.loop:
                    idx = 0
                else:
                    break

            elapsed = time.time() - t_start
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n🛑 Zatrzymano symulator OptiTrack.")
    finally:
        sock.close()


if __name__ == "__main__":
    main()

