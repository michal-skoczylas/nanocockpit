#!/usr/bin/env python3
"""
gesture_flight_controller.py

Krótki i czytelny skrypt sterowania dronem Crazyflie za pomocą 8 gestów dłoni
z podglądem obrazu z kamery AI-Deck w czasie rzeczywistym.

8 gestów i ich kierunki/zachowania:
  - openhand  -> Ruch w przód (mc.forward)
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

# Parametry lotu
URI = uri_helper.uri_from_env(default="radio://0/80/2M/E7E7E7E7E6")
DEFAULT_HEIGHT = 0.45   # wysokość bazowa [m]
STEP_DIST = 0.15        # mały krok przesunięcia [m] (15 cm)
VELOCITY = 0.15         # bezpieczna, niska prędkość [m/s]
CONF_THRESH = 0.65      # minimalna pewność gestu
DEBOUNCE_FRAMES = 3     # gest musi trwać N kolejnych klatek

GESTURE_ACTIONS = {
    "openhand": "PRZÓD (forward)",
    "shaka": "TYŁ (back)",
    "left": "LEWO (left)",
    "right": "PRAWO (right)",
    "v": "GÓRA (up)",
    "dislike": "DÓŁ (down)",
    "ok": "STOP / HOVER",
    "closehand": "LĄDOWANIE (land)",
}

cmd_queue = queue.Queue(maxsize=1)
running = True


def drone_worker(uri: str):
    """Wątek MotionCommander - odbiera gesty i steruje dronem."""
    global running
    print(f"[Crazyflie] Inicjalizacja CRTP i łączenie z {uri}...")
    cflib.crtp.init_drivers()

    try:
        with SyncCrazyflie(uri, cf=Crazyflie(rw_cache="./cache")) as scf:
            time.sleep(1.0)
            print(f"[Crazyflie] Start na wysokość {DEFAULT_HEIGHT}m...")
            with MotionCommander(scf, default_height=DEFAULT_HEIGHT) as mc:
                while running:
                    try:
                        gesture = cmd_queue.get(timeout=0.2)
                    except queue.Empty:
                        continue

                    print(f"🎮 [Ruch] {GESTURE_ACTIONS.get(gesture, gesture)}")

                    if gesture == "closehand":
                        print("🛑 [Lądowanie] Zamknięta dłoń - ląduję!")
                        mc.land()
                        running = False
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
        running = False
        print("[Crazyflie] Zakończono wątek drona.")


def main():
    global running
    parser = argparse.ArgumentParser(description="Sterowanie Crazyflie 8 gestami dłoni z podglądem wideo")
    parser.add_argument("-host", default="aideck.local", help="Host kamery AI-Deck")
    parser.add_argument("-port", type=int, default=5000, help="Port AI-Deck")
    parser.add_argument("--uri", default=URI, help="URI radia Crazyflie")
    parser.add_argument("--dry-run", action="store_true", help="Test wizji bez włączania drona")
    args = parser.parse_args()

    # Załadowanie modeli YOLO + ResNet
    thesis_dir = Path(__file__).resolve().parents[4]
    dane_weights = thesis_dir / "dane" / "weights"
    detector = DroneGestureDetector(
        yolo_path=find_weights_file([dane_weights / "yolo_hand_detector.pt"]),
        classifier_bin_path=find_weights_file([dane_weights / "resnet18_binary_best.torchscript.pt"]),
        classifier_all_path=find_weights_file([dane_weights / "resnet_all_gestures" / "resnet_all8_best.torchscript.pt"]),
        mode="all",
    )

    # Uruchomienie sterowania w osobnym wątku
    if not args.dry_run:
        threading.Thread(target=drone_worker, args=(args.uri,), daemon=True).start()
    else:
        print("🟡 [TRYB DRY-RUN] Symulacja bez łączenia z dronem.")

    client = StreamerClient(host=args.host, port=args.port, udp_send=True)
    streak_gesture, streak_count = None, 0
    current_action = "Zawis / Czeka na gest"

    print("\n" + "=" * 65)
    print("🚁 [KONTROLER CRAZYFLIE Z PODGLĄDEM WIDEO AKTYWNY]")
    print("   openhand: przód | shaka: tył     | left: lewo | right: prawo")
    print("   v: góra         | dislike: dół   | ok: stop   | closehand: lądowanie")
    print("   [q] / [ESC] - wyjście i lądowanie")
    print("=" * 65 + "\n")

    try:
        for frame, tof_frame, meta in client.receive():
            if not running:
                break
            client.send_reply(meta, None)

            # Obraz z kamery drona
            vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if len(frame.shape) == 2 else frame.copy()
            h, w = vis.shape[:2]

            # Detekcja dłoni i klasyfikacja gestu
            info = detector.process_frame(vis)
            gesture = info["gesture"]
            conf = info["gesture_conf"]

            # Filtr debouncing (stabilność gestu przez N klatek)
            if info["detected"] and conf >= CONF_THRESH:
                streak_count = (streak_count + 1) if gesture == streak_gesture else 1
                streak_gesture = gesture

                if streak_count >= DEBOUNCE_FRAMES:
                    streak_count = 0
                    current_action = GESTURE_ACTIONS.get(gesture, gesture)
                    if args.dry_run:
                        print(f"🟡 [DRY-RUN]: {gesture.upper()} -> {current_action}")
                    else:
                        if cmd_queue.empty():
                            cmd_queue.put(gesture)
            else:
                streak_count = 0
                streak_gesture = None

            # --- PODGLĄD WIDEO Z DRONA ---
            # 1. Bounding box wokół dłoni
            if info["detected"] and info["box"] is not None:
                x1, y1, x2, y2 = info["box"]
                color = GESTURE_COLORS.get(gesture, (0, 255, 0))
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                label = f"{gesture.upper()} {conf*100:.0f}%"
                cv2.putText(vis, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)

            # 2. Miniaturka wycinka dłoni (Picture-in-Picture) w prawym dolnym rogu
            if info.get("crop") is not None:
                crop = cv2.resize(info["crop"], (80, 80))
                vis[h - 85:h - 5, w - 85:w - 5] = crop
                cv2.rectangle(vis, (w - 86, h - 86), (w - 4, h - 4), (0, 255, 255), 1)

            # 3. Pasek statusu drona u góry
            status_text = f"AKCJA: {current_action}"
            cv2.rectangle(vis, (0, 0), (w, 28), (20, 20, 20), -1)
            cv2.putText(vis, status_text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)

            # Wyświetlenie okna kamery
            cv2.imshow("NanoCockpit AI-Deck: Podglad Kamery Drona", vis)

            # 4. Opcjonalny podgląd czujnika TOF (jeśli dron przesyła)
            if tof_frame is not None:
                tof_norm = cv2.normalize(tof_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                tof_vis = cv2.applyColorMap(tof_norm, cv2.COLORMAP_TURBO)
                cv2.imshow("Czujnik TOF", cv2.resize(tof_vis, (160, 160), interpolation=cv2.INTER_NEAREST))

            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                print("🛑 Zatrzymanie przez użytkownika (q/ESC)")
                break
    finally:
        running = False
        client.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
