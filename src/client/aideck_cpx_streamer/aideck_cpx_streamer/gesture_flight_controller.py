#!/usr/bin/env python3
"""
gesture_flight_controller.py

Krótki i czytelny skrypt sterowania dronem Crazyflie za pomocą 8 gestów dłoni.
Zawiera podgląd z kamery AI-Deck oraz kluczowe zabezpieczenia lotu.

8 gestów i ich kierunki/zachowania:
  - openhand  -> Start (z ziemi) / Ruch w przód (w powietrzu)
  - shaka     -> Ruch w tył (mc.back)
  - left      -> Ruch w lewo (mc.left)
  - right     -> Ruch w prawo (mc.right)
  - v         -> Wznoszenie w górę (mc.up)
  - dislike   -> Opadanie w dół (mc.down)
  - ok        -> Zatrzymanie / zawis (mc.stop)
  - closehand -> Bezpieczne lądowanie (mc.land)
"""

import sys
import time
import queue
import signal
import argparse
import threading
from pathlib import Path

import cv2
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils import uri_helper

from aideck_cpx_streamer.cpx import StreamerClient
from aideck_cpx_streamer.gesture_detector_viewer import (
    DroneGestureDetector,
    find_weights_file,
    GESTURE_COLORS,
)

# Bardzo bezpieczne, niskie parametry lotu
URI = uri_helper.uri_from_env(default="radio://0/80/2M/E7E7E7E7E6")
DEFAULT_HEIGHT = 0.40   # niska wysokość bazowa [m] (40 cm)
STEP_DIST = 0.10        # mały mikro-krok [m] (10 cm)
VELOCITY = 0.08         # bardzo niska, łagodna prędkość ruchu [m/s] (8 cm/s)
CONF_THRESH = 0.65      # próg pewności ResNet
DEBOUNCE_FRAMES = 3     # wymagana liczba kolejnych klatek gestu
WATCHDOG_TIMEOUT = 2.0  # maks. czas [s] bez klatki z kamery przed lądowaniem awaryjnym

GESTURE_ACTIONS = {
    "openhand": "START / PRZÓD (forward)",
    "shaka": "TYŁ (back)",
    "left": "LEWO (left)",
    "right": "PRAWO (right)",
    "v": "GÓRA (up)",
    "dislike": "DÓŁ (down)",
    "ok": "STOP / HOVER",
    "closehand": "LĄDOWANIE (land)",
}

# Kolejki i zdarzenia synchronizujące
cmd_queue = queue.Queue(maxsize=1)
stop_event = threading.Event()
takeoff_event = threading.Event()
last_frame_time = time.time()


def drone_worker(uri: str):
    """Wątek drona: bezpieczny start, obsługa komend i lądowanie awaryjne."""
    global last_frame_time
    print(f"[Crazyflie] Inicjalizacja CRTP i łączenie z {uri}...")
    cflib.crtp.init_drivers()

    try:
        with SyncCrazyflie(uri, cf=Crazyflie(rw_cache="./cache")) as scf:
            print("🟢 [Crazyflie] Połączono! Dron stoi bezpiecznie na ziemi.")
            print("👉 Pokaż OPENHAND lub naciśnij klawisz 't' w oknie, aby wystartować.")

            # 1. Czekaj na ziemi na jawne potwierdzenie startu
            while not stop_event.is_set() and not takeoff_event.is_set():
                time.sleep(0.05)

            if stop_event.is_set():
                return

            # 2. Start w powietrze (MotionCommander)
            print(f"[Crazyflie] Startuję powoli na bezpieczną wysokość {DEFAULT_HEIGHT}m...")
            with MotionCommander(scf, default_height=DEFAULT_HEIGHT) as mc:
                while not stop_event.is_set():
                    # Watchdog: brak obrazu z Wi-Fi -> lądowanie awaryjne
                    if time.time() - last_frame_time > WATCHDOG_TIMEOUT:
                        print("⚠️ [Watchdog Failsafe] Utrata klatek z kamery > 2s! Ląduję...")
                        mc.land()
                        stop_event.set()
                        break

                    try:
                        gesture = cmd_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    print(f"🎮 [Ruch drona]: {GESTURE_ACTIONS.get(gesture, gesture)}")

                    if gesture == "closehand":
                        print("🛑 [Lądowanie] Zamknięta dłoń - ląduję natychmiast.")
                        mc.land()
                        stop_event.set()
                        break
                    elif gesture == "openhand":
                        mc.forward(STEP_DIST, velocity=VELOCITY)
                    elif gesture == "shaka":
                        mc.back(STEP_DIST, velocity=VELOCITY)
                    elif gesture == "left":
                        mc.left(STEP_DIST, velocity=VELOCITY)
                    elif gesture == "right":
                        mc.right(STEP_DIST, velocity=VELOCITY)
                    elif gesture == "v":
                        mc.up(STEP_DIST, velocity=VELOCITY)
                    elif gesture == "dislike":
                        mc.down(STEP_DIST, velocity=VELOCITY)
                    elif gesture == "ok":
                        mc.stop()
    except Exception as e:
        print(f"❌ [Crazyflie Błąd]: {e}")
    finally:
        stop_event.set()
        print("[Crazyflie] Wątek drona zakończony bezpiecznie.")


def main():
    global last_frame_time
    parser = argparse.ArgumentParser(description="Sterowanie Crazyflie 8 gestami dłoni z podglądem wideo")
    parser.add_argument("-host", default="aideck.local", help="Host kamery AI-Deck")
    parser.add_argument("-port", type=int, default=5000, help="Port AI-Deck")
    parser.add_argument("--uri", default=URI, help="URI radia Crazyflie")
    parser.add_argument("--dry-run", action="store_true", help="Test wizji bez włączania drona")
    args = parser.parse_args()

    # Bezpieczna obsługa Ctrl+C
    def _sigint_handler(sig, frame):
        print("\n🛑 [SIGINT / Ctrl+C] Przerwanie skryptu - natychmiastowe lądowanie...")
        stop_event.set()

    signal.signal(signal.SIGINT, _sigint_handler)

    # Załadowanie modeli YOLO + ResNet
    thesis_dir = Path(__file__).resolve().parents[4]
    dane_weights = thesis_dir / "dane" / "weights"
    detector = DroneGestureDetector(
        yolo_path=find_weights_file([dane_weights / "yolo_hand_detector.pt"]),
        classifier_bin_path=find_weights_file([dane_weights / "resnet18_binary_best.torchscript.pt"]),
        classifier_all_path=find_weights_file([dane_weights / "resnet_all_gestures" / "resnet_all8_best.torchscript.pt"]),
        mode="all",
    )

    # Uruchomienie wątku drona (bez daemon=True, z kontrolowanym wyjściem)
    drone_thread = None
    if not args.dry_run:
        drone_thread = threading.Thread(target=drone_worker, args=(args.uri,), daemon=False)
        drone_thread.start()
    else:
        print("🟡 [TRYB DRY-RUN] Symulacja bez łączenia z dronem.")
        takeoff_event.set()

    client = StreamerClient(host=args.host, port=args.port, udp_send=True)
    streak_gesture, streak_count = None, 0
    current_action = "Czeka na start (OPENHAND / 't')"

    print("\n" + "=" * 65)
    print("🚁 [KONTROLER CRAZYFLIE - TRYB BEZPIECZNY]")
    print(f"   Prędkość ruchu: {VELOCITY*100:.0f} cm/s | Krok: {STEP_DIST*100:.0f} cm")
    print("   [t] - Start z ziemi | [q] / [ESC] - Bezpieczne lądowanie i wyjście")
    print("=" * 65 + "\n")

    try:
        for frame, tof_frame, meta in client.receive():
            if stop_event.is_set():
                break
            client.send_reply(meta, None)
            last_frame_time = time.time()

            vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if len(frame.shape) == 2 else frame.copy()
            h, w = vis.shape[:2]

            # Detekcja AI
            info = detector.process_frame(vis)
            gesture = info["gesture"]
            conf = info["gesture_conf"]

            # Debouncing: gest musi trwać N kolejnych klatek
            if info["detected"] and conf >= CONF_THRESH:
                streak_count = (streak_count + 1) if gesture == streak_gesture else 1
                streak_gesture = gesture

                if streak_count >= DEBOUNCE_FRAMES:
                    streak_count = 0

                    # Jeśli dron jeszcze nie wystartował, 'openhand' uruchamia start
                    if not takeoff_event.is_set():
                        if gesture == "openhand":
                            print("🛫 [Start] Wykryto OPENHAND - startuję drona!")
                            takeoff_event.set()
                            current_action = "Startowanie..."
                    else:
                        current_action = GESTURE_ACTIONS.get(gesture, gesture)
                        if args.dry_run:
                            print(f"🟡 [DRY-RUN]: {gesture.upper()} -> {current_action}")
                        else:
                            if cmd_queue.empty():
                                cmd_queue.put(gesture)
            else:
                streak_count = 0
                streak_gesture = None

            # --- PODGLĄD WIDEO ---
            # 1. Bounding box dłoni
            if info["detected"] and info["box"] is not None:
                x1, y1, x2, y2 = info["box"]
                color = GESTURE_COLORS.get(gesture, (0, 255, 0))
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                label = f"{gesture.upper()} {conf*100:.0f}%"
                cv2.putText(vis, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)

            # 2. Miniatura wycinka dłoni (PiP)
            if info.get("crop") is not None:
                crop = cv2.resize(info["crop"], (80, 80))
                vis[h - 85:h - 5, w - 85:w - 5] = crop
                cv2.rectangle(vis, (w - 86, h - 86), (w - 4, h - 4), (0, 255, 255), 1)

            # 3. Pasek statusu
            cv2.rectangle(vis, (0, 0), (w, 28), (20, 20, 20), -1)
            cv2.putText(vis, f"AKCJA: {current_action}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("NanoCockpit AI-Deck: Podglad Kamery Drona", vis)

            # 4. Czujnik TOF jeśli dostępny
            if tof_frame is not None:
                tof_norm = cv2.normalize(tof_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                tof_vis = cv2.applyColorMap(tof_norm, cv2.COLORMAP_TURBO)
                cv2.imshow("Czujnik TOF", cv2.resize(tof_vis, (160, 160), interpolation=cv2.INTER_NEAREST))

            # Obsługa klawiatury
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("🛑 Zatrzymanie przez użytkownika (q/ESC)")
                stop_event.set()
                break
            elif key == ord("t") and not takeoff_event.is_set():
                print("🛫 [Klawiatura] Start drona klawiszem 't'")
                takeoff_event.set()
                current_action = "Startowanie..."

    finally:
        stop_event.set()
        client.shutdown()
        cv2.destroyAllWindows()
        if drone_thread and drone_thread.is_alive():
            print("⏳ Czekam na wylądowanie drona i zamknięcie połączenia...")
            drone_thread.join(timeout=4.0)
        print("✅ Program zakończony.")


if __name__ == "__main__":
    main()
