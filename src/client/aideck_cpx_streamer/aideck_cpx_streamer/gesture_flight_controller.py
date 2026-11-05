#!/usr/bin/env python3
"""
gesture_flight_controller.py

Krótki i czytelny skrypt sterowania dronem Crazyflie za pomocą 8 gestów dłoni.
Opiera się na oficjalnym cflib MotionCommander i dwustopniowej detekcji (YOLO + ResNet).

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
from aideck_cpx_streamer.gesture_detector_viewer import DroneGestureDetector, find_weights_file

# Domyślne parametry eksperymentu (niewielkie, bezpieczne ruchy)
URI = uri_helper.uri_from_env(default="radio://0/99/2M/E7E7E7E7E6")
DEFAULT_HEIGHT = 0.45   # wysokość bazowa [m]
STEP_DIST = 0.15        # mały krok przesunięcia [m] (15 cm)
VELOCITY = 0.15         # bezpieczna, niska prędkość [m/s]
CONF_THRESH = 0.65      # minimalna pewność gestu
DEBOUNCE_FRAMES = 3     # gest musi trwać N kolejnych klatek

# Kolejka poleceń i flaga zatrzymania
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

                    print(f"🎮 [Ruch] Wykonuję akcję dla gestu: {gesture.upper()}")

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
    parser = argparse.ArgumentParser(description="Sterowanie Crazyflie 8 gestami dłoni")
    parser.add_argument("-host", default="aideck.local", help="Host kamery AI-Deck")
    parser.add_argument("-port", type=int, default=5000, help="Port AI-Deck")
    parser.add_argument("--uri", default=URI, help="URI radia Crazyflie")
    parser.add_argument("--dry-run", action="store_true", help="Test wizji bez włączania drona")
    args = parser.parse_args()

    # Automatyczne odnalezienie modeli
    thesis_dir = Path(__file__).resolve().parents[4]
    dane_weights = thesis_dir / "dane" / "weights"
    detector = DroneGestureDetector(
        yolo_path=find_weights_file([dane_weights / "yolo_hand_detector.pt"]),
        classifier_bin_path=find_weights_file([dane_weights / "resnet18_binary_best.torchscript.pt"]),
        classifier_all_path=find_weights_file([dane_weights / "resnet_all_gestures" / "resnet_all8_best.torchscript.pt"]),
        mode="all",
    )

    # Uruchomienie drona w osobnym wątku (jeśli nie dry-run)
    if not args.dry_run:
        threading.Thread(target=drone_worker, args=(args.uri,), daemon=True).start()
    else:
        print("🟡 [TRYB DRY-RUN] Symulacja bez łączenia z dronem.")

    client = StreamerClient(host=args.host, port=args.port, udp_send=True)
    streak_gesture, streak_count = None, 0

    print("\n" + "=" * 60)
    print("🚁 [KONTROLER 8 GESTÓW CRAZYFLIE - AKTYWNY]")
    print("   openhand: przód | shaka: tył     | left: lewo | right: prawo")
    print("   v: góra         | dislike: dół   | ok: stop   | closehand: lądowanie")
    print("   [q] / [ESC] - wyjście i lądowanie")
    print("=" * 60 + "\n")

    try:
        for frame, _, meta in client.receive():
            if not running:
                break
            client.send_reply(meta, None)

            # Przetwarzanie obrazu
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if len(frame.shape) == 2 else frame
            info = detector.process_frame(frame_bgr)
            gesture = info["gesture"]
            conf = info["gesture_conf"]

            # Debouncing: gest musi trwać kilka klatek
            if info["detected"] and conf >= CONF_THRESH:
                streak_count = (streak_count + 1) if gesture == streak_gesture else 1
                streak_gesture = gesture

                if streak_count >= DEBOUNCE_FRAMES:
                    streak_count = 0  # reset po przekazaniu komendy
                    if args.dry_run:
                        print(f"🟡 [DRY-RUN GEST]: {gesture.upper()} ({conf*100:.0f}%)")
                    else:
                        if cmd_queue.empty():
                            cmd_queue.put(gesture)
            else:
                streak_count = 0
                streak_gesture = None

            # Prosty, czytelny podgląd
            label = f"{gesture.upper()} ({conf*100:.0f}%)" if info["detected"] else "Brak dloni"
            cv2.putText(frame_bgr, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Crazyflie Gesture Control", frame_bgr)

            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                print("🛑 Zatrzymanie przez użytkownika (q/ESC)")
                break
    finally:
        running = False
        client.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
