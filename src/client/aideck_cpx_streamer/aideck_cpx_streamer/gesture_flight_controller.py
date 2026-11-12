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
import json
import socket
import queue
import signal
import argparse
import threading
from pathlib import Path
from collections import deque, Counter

import cv2
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils import uri_helper

from aideck_cpx_streamer.cpx import StreamerClient
from aideck_cpx_streamer.gesture_detector_viewer import (
    DroneGestureDetector,
    GESTURE_COLORS,
)

# ==============================================================================
# 🎯 ŚCIEŻKI DO MODELI AI (wpisz ścieżki bezpośrednio tutaj w kodzie)
# ==============================================================================
YOLO_MODEL_PATH = "/Users/michalskoczylas/Documents/studia/thesis/nanocockpit/src/client/aideck_cpx_streamer/models/yolo_hand_detector_2026-11-04.pt"
RESNET_MODEL_PATH = "/Users/michalskoczylas/Documents/studia/thesis/nanocockpit/src/client/aideck_cpx_streamer/models/resnet_best_2026-11-04.torchscript.pt"
RESNET_BINARY_PATH = None  # Opcjonalnie: ścieżka dla modelu 2-klasowego (jeśli None, używa RESNET_MODEL_PATH)


# Bardzo bezpieczne, niskie parametry lotu
URI = uri_helper.uri_from_env(default="radio://0/99/2M/E7E7E7E7E7")
DEFAULT_HEIGHT = 0.50   # niska wysokość bazowa [m] (50 cm)
STEP_DIST = 0.10        # mały mikro-krok [m] (10 cm)
VELOCITY = 0.08         # bardzo niska, łagodna prędkość ruchu [m/s] (8 cm/s)
CONF_THRESH = 0.65      # próg pewności ResNet
WATCHDOG_TIMEOUT = 2.0  # maks. czas [s] bez klatki z kamery przed lądowaniem awaryjnym

# ==============================================================================
# 🪟 OKNO PRZESUWNE DETEKCJI GESTÓW (Sliding Window)
# ==============================================================================
# Okno analizuje N ostatnich klatek. Aby gest został aktywowany, musi wystąpić
# w co najmniej M klatkach w obrębie tego okna (M z N klatek).
WINDOW_SIZE = 10           # N: rozmiar okna przesuwnego (liczba ostatnich klatek)
MIN_GESTURE_FRAMES = 7     # M: minimalna liczba klatek z danym gestem do wyzwolenia akcji
ACTIVATION_COOLDOWN = 0.6  # [s] minimalny odstęp czasu przed kolejną aktywacją tego samego gestu


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


class TelemetryPublisher:
    """Wysyła bieżący stan kontrolera i drona przez UDP JSON do skryptu rejestrującego (np. optitrack_flight_logger)."""

    def __init__(self, host: str = "127.0.0.1", port: int = 12112, enabled: bool = True):
        self.host = host
        self.port = port
        self.enabled = enabled
        self._sock = None
        if self.enabled:
            try:
                self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                print(f"📡 [Telemetria] Aktywne nadawanie UDP na {self.host}:{self.port}")
            except Exception as e:
                print(f"⚠️ [Telemetria] Błąd tworzenia gniazda UDP: {e}")
                self.enabled = False

        self._lock = threading.Lock()
        self.state = {
            "flight_state": "GROUND",
            "active_command": "idle",
            "command_timestamp": time.time(),
            "detected_gesture": None,
            "gesture_conf": 0.0,
            "gesture_timestamp": time.time(),
            "window_status": "brak",
        }

    def update_flight(self, flight_state: str = None, command: str = None):
        with self._lock:
            now = time.time()
            if flight_state is not None:
                self.state["flight_state"] = flight_state
            if command is not None:
                self.state["active_command"] = command
                self.state["command_timestamp"] = now
        self.send()

    def update_vision(self, gesture: str = None, conf: float = 0.0, window_status: str = None):
        with self._lock:
            now = time.time()
            self.state["detected_gesture"] = gesture
            self.state["gesture_conf"] = round(float(conf), 4)
            self.state["gesture_timestamp"] = now
            if window_status is not None:
                self.state["window_status"] = window_status
        self.send()

    def send(self):
        if not self.enabled or self._sock is None:
            return
        with self._lock:
            payload = dict(self.state)
            payload["timestamp"] = time.time()
        try:
            msg = json.dumps(payload).encode("utf-8")
            self._sock.sendto(msg, (self.host, self.port))
        except Exception:
            pass

    def close(self):
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None


def drone_worker(uri: str, telemetry: TelemetryPublisher = None):
    """Wątek drona: bezpieczny start, obsługa komend i lądowanie awaryjne."""
    global last_frame_time
    if telemetry:
        telemetry.update_flight(flight_state="GROUND", command="idle")
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
            if telemetry:
                telemetry.update_flight(flight_state="TAKING_OFF", command="takeoff")
            print(f"[Crazyflie] Startuję powoli na bezpieczną wysokość {DEFAULT_HEIGHT}m...")
            with MotionCommander(scf, default_height=DEFAULT_HEIGHT) as mc:
                if telemetry:
                    telemetry.update_flight(flight_state="FLYING", command="hover")
                while not stop_event.is_set():
                    # Watchdog: brak obrazu z Wi-Fi -> lądowanie awaryjne
                    if time.time() - last_frame_time > WATCHDOG_TIMEOUT:
                        print("⚠️ [Watchdog Failsafe] Utrata klatek z kamery > 2s! Ląduję...")
                        if telemetry:
                            telemetry.update_flight(flight_state="LANDING", command="failsafe_land")
                        mc.land()
                        if telemetry:
                            telemetry.update_flight(flight_state="LANDED", command="idle")
                        stop_event.set()
                        break

                    try:
                        gesture = cmd_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    print(f"🎮 [Ruch drona]: {GESTURE_ACTIONS.get(gesture, gesture)}")

                    if gesture == "closehand":
                        print("🛑 [Lądowanie] Zamknięta dłoń - ląduję natychmiast.")
                        if telemetry:
                            telemetry.update_flight(flight_state="LANDING", command="land")
                        mc.land()
                        if telemetry:
                            telemetry.update_flight(flight_state="LANDED", command="idle")
                        stop_event.set()
                        break
                    elif gesture == "openhand":
                        if telemetry:
                            telemetry.update_flight(flight_state="FLYING", command="forward")
                        mc.forward(STEP_DIST, velocity=VELOCITY)
                        if telemetry:
                            telemetry.update_flight(flight_state="FLYING", command="hover")
                    elif gesture == "shaka":
                        if telemetry:
                            telemetry.update_flight(flight_state="FLYING", command="back")
                        mc.back(STEP_DIST, velocity=VELOCITY)
                        if telemetry:
                            telemetry.update_flight(flight_state="FLYING", command="hover")
                    elif gesture == "left":
                        if telemetry:
                            telemetry.update_flight(flight_state="FLYING", command="left")
                        mc.left(STEP_DIST, velocity=VELOCITY)
                        if telemetry:
                            telemetry.update_flight(flight_state="FLYING", command="hover")
                    elif gesture == "right":
                        if telemetry:
                            telemetry.update_flight(flight_state="FLYING", command="right")
                        mc.right(STEP_DIST, velocity=VELOCITY)
                        if telemetry:
                            telemetry.update_flight(flight_state="FLYING", command="hover")
                    elif gesture == "v":
                        if telemetry:
                            telemetry.update_flight(flight_state="FLYING", command="up")
                        mc.up(STEP_DIST, velocity=VELOCITY)
                        if telemetry:
                            telemetry.update_flight(flight_state="FLYING", command="hover")
                    elif gesture == "dislike":
                        if telemetry:
                            telemetry.update_flight(flight_state="FLYING", command="down")
                        mc.down(STEP_DIST, velocity=VELOCITY)
                        if telemetry:
                            telemetry.update_flight(flight_state="FLYING", command="hover")
                    elif gesture == "ok":
                        if telemetry:
                            telemetry.update_flight(flight_state="FLYING", command="stop")
                        mc.stop()
                        if telemetry:
                            telemetry.update_flight(flight_state="FLYING", command="hover")
    except Exception as e:
        print(f"❌ [Crazyflie Błąd]: {e}")
    finally:
        if telemetry:
            telemetry.update_flight(flight_state="LANDED", command="idle")
        stop_event.set()
        print("[Crazyflie] Wątek drona zakończony bezpiecznie.")


def main():
    global last_frame_time
    parser = argparse.ArgumentParser(description="Sterowanie Crazyflie 8 gestami dłoni z podglądem wideo")
    parser.add_argument("-host", default="aideck.local", help="Host kamery AI-Deck")
    parser.add_argument("-port", type=int, default=5000, help="Port AI-Deck")
    parser.add_argument("--uri", default=URI, help="URI radia Crazyflie")
    parser.add_argument("--dry-run", action="store_true", help="Test wizji bez włączania drona")
    parser.add_argument("--mode", default="all", choices=["all", "binary"], help="Tryb gestów: 'all' (8 gestów) lub 'binary' (open/close)")
    parser.add_argument("--yolo-weights", default=None, help="Ścieżka do wag modelu YOLO (opcjonalnie)")
    parser.add_argument("--resnet-weights", default=None, help="Ścieżka do wag modelu ResNet (opcjonalnie)")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"], help="Urządzenie obliczeniowe")
    parser.add_argument("--conf", type=float, default=0.35, help="Próg detekcji dłoni YOLO")
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE, help="Rozmiar okna przesuwnego (N klatek)")
    parser.add_argument("--min-frames", type=int, default=MIN_GESTURE_FRAMES, help="Minimalna liczba klatek z gestem w oknie (M)")
    parser.add_argument("--cooldown", type=float, default=ACTIVATION_COOLDOWN, help="Minimalny odstęp [s] między kolejnymi aktywacjami tego samego gestu")
    parser.add_argument("--webcam", action="store_true", help="Użyj kamerki laptopa zamiast Wi-Fi AI-Deck (testy w domu)")
    parser.add_argument("--telemetry-host", default="127.0.0.1", help="Host docelowy dla telemetrii (skrypt tracking/logger)")
    parser.add_argument("--telemetry-port", type=int, default=12112, help="Port UDP dla telemetrii (skrypt tracking/logger)")
    parser.add_argument("--no-telemetry", action="store_true", help="Wyłącz nadawanie telemetrii UDP")
    args = parser.parse_args()


    # Bezpieczna obsługa Ctrl+C
    def _sigint_handler(sig, frame):
        print("\n🛑 [SIGINT / Ctrl+C] Przerwanie skryptu - natychmiastowe lądowanie...")
        stop_event.set()

    signal.signal(signal.SIGINT, _sigint_handler)

    # Ścieżki do modeli zdefiniowane w kodzie (lub opcjonalnie nadpisane z CLI)
    yolo_raw = args.yolo_weights or YOLO_MODEL_PATH
    resnet_raw = args.resnet_weights or RESNET_MODEL_PATH
    bin_raw = RESNET_BINARY_PATH or resnet_raw

    if not yolo_raw:
        print("❌ [Błąd]: Nie podano ścieżki do modelu YOLO. Wpisz ścieżkę w zmiennej YOLO_MODEL_PATH w kodzie.")
        sys.exit(1)
    if not resnet_raw:
        print("❌ [Błąd]: Nie podano ścieżki do modelu ResNet. Wpisz ścieżkę w zmiennej RESNET_MODEL_PATH w kodzie.")
        sys.exit(1)

    yolo_file = Path(yolo_raw).expanduser().resolve()
    resnet_all_file = Path(resnet_raw).expanduser().resolve()
    resnet_bin_file = Path(bin_raw).expanduser().resolve()

    if not yolo_file.exists():
        print(f"❌ [Błąd]: Plik modelu YOLO nie istnieje pod ścieżką:\n   {yolo_file}\nSprawdź zmienną YOLO_MODEL_PATH w kodzie.")
        sys.exit(1)
    if not resnet_all_file.exists():
        print(f"❌ [Błąd]: Plik modelu ResNet nie istnieje pod ścieżką:\n   {resnet_all_file}\nSprawdź zmienną RESNET_MODEL_PATH w kodzie.")
        sys.exit(1)
    if not resnet_bin_file.exists():
        print(f"❌ [Błąd]: Plik modelu binarnego ResNet nie istnieje pod ścieżką:\n   {resnet_bin_file}")
        sys.exit(1)

    print("📦 [Załadowane Modele AI]:")
    print(f"   Detektor YOLO:   {yolo_file}")
    print(f"   Klasyfikator 8G: {resnet_all_file}")
    if RESNET_BINARY_PATH:
        print(f"   Klasyfikator 2G: {resnet_bin_file}")

    detector = DroneGestureDetector(
        yolo_path=yolo_file,
        classifier_bin_path=resnet_bin_file,
        classifier_all_path=resnet_all_file,
        mode=args.mode,
        conf_thresh=args.conf,
        device_str=args.device,
        smooth_frames=1,  # Przekazujemy surowe predykcje klatek do naszego okna przesuwnego
    )

    # Inicjalizacja modułu telemetrii UDP
    telemetry = TelemetryPublisher(
        host=args.telemetry_host,
        port=args.telemetry_port,
        enabled=not args.no_telemetry,
    )

    # Uruchomienie wątku drona (bez daemon=True, z kontrolowanym wyjściem)
    drone_thread = None
    if not args.dry_run:
        drone_thread = threading.Thread(target=drone_worker, args=(args.uri, telemetry), daemon=False)
        drone_thread.start()
    else:
        print("🟡 [TRYB DRY-RUN] Symulacja bez łączenia z dronem.")
        telemetry.update_flight(flight_state="GROUND", command="idle")
        takeoff_event.set()

    cap = None
    client = None
    if args.webcam:
        print("📷 [Kamera] Uruchamianie lokalnej kamerki...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ [Błąd]: Nie można otworzyć lokalnej kamerki.")
            sys.exit(1)
    else:
        client = StreamerClient(host=args.host, port=args.port, udp_send=True)

    def frame_stream():
        if args.webcam:
            while not stop_event.is_set():
                ret, f = cap.read()
                if not ret:
                    time.sleep(0.03)
                    continue
                yield f, None, None
        else:
            for f, tof, m in client.receive():
                if stop_event.is_set():
                    break
                yield f, tof, m

    # Inicjalizacja bufora okna przesuwnego
    win_size = max(1, args.window_size)
    min_frames = min(max(1, args.min_frames), win_size)
    gesture_window = deque(maxlen=win_size)
    last_activation_time = 0.0
    last_activated_gesture = None
    current_action = "Czeka na start (OPENHAND / 't')"
    window_status_str = f"0/{win_size}"

    print("\n" + "=" * 65)
    print("🚁 [KONTROLER CRAZYFLIE - TRYB BEZPIECZNY]")
    print(f"   Prędkość ruchu: {VELOCITY*100:.0f} cm/s | Krok: {STEP_DIST*100:.0f} cm")
    print(f"   Okno przesuwne: {min_frames} z {win_size} klatek | Cooldown: {args.cooldown}s")
    if not args.no_telemetry:
        print(f"   Telemetria UDP: {args.telemetry_host}:{args.telemetry_port}")
    if args.webcam:
        print("   Tryb wideo:     Lokalna kamerka (testy offline)")
    print("   [t] - Start z ziemi | [q] / [ESC] - Bezpieczne lądowanie i wyjście")
    print("=" * 65 + "\n")

    try:
        for frame, tof_frame, meta in frame_stream():
            if stop_event.is_set():
                break
            if client and meta:
                client.send_reply(meta, None)
            last_frame_time = time.time()

            vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if len(frame.shape) == 2 else frame.copy()
            h, w = vis.shape[:2]

            # Detekcja AI
            info = detector.process_frame(vis)
            raw_gesture = info["gesture"]
            conf = info["gesture_conf"]

            # Kwalifikacja klatki: gest musi mieć zadowalającą pewność modelu
            valid_frame_gesture = raw_gesture if (info["detected"] and conf >= CONF_THRESH) else None
            gesture_window.append(valid_frame_gesture)

            # Analiza okna przesuwnego (liczymy tylko zidentyfikowane gesty w buforze)
            valid_gestures = [g for g in gesture_window if g is not None]
            gesture_counts = Counter(valid_gestures)
            top_gesture, count = gesture_counts.most_common(1)[0] if gesture_counts else (None, 0)
            window_status_str = f"{top_gesture.upper()}: {count}/{win_size}" if top_gesture else f"brak (0/{win_size})"

            # Publikacja stanu detekcji w telemetrii
            telemetry.update_vision(
                gesture=valid_frame_gesture,
                conf=conf if info["detected"] else 0.0,
                window_status=window_status_str,
            )

            # Sprawdzenie warunku aktywacji (M z N klatek)
            if top_gesture and count >= min_frames:
                now = time.time()
                is_new_gesture = (top_gesture != last_activated_gesture)
                cooldown_ok = (now - last_activation_time >= args.cooldown)

                if is_new_gesture or cooldown_ok:
                    last_activation_time = now
                    last_activated_gesture = top_gesture

                    # Jeśli dron jeszcze nie wystartował, 'openhand' uruchamia start
                    if not takeoff_event.is_set():
                        if top_gesture == "openhand":
                            print(f"🛫 [Start] Wykryto OPENHAND ({count}/{win_size} klatek) - startuję drona!")
                            takeoff_event.set()
                            current_action = "Startowanie..."
                            if args.dry_run:
                                telemetry.update_flight(flight_state="FLYING", command="hover")
                            gesture_window.clear()
                    else:
                        current_action = GESTURE_ACTIONS.get(top_gesture, top_gesture)
                        if args.dry_run:
                            print(f"🟡 [DRY-RUN]: {top_gesture.upper()} ({count}/{win_size} klatek) -> {current_action}")
                            telemetry.update_flight(flight_state="FLYING", command=top_gesture)
                        else:
                            if cmd_queue.empty():
                                cmd_queue.put(top_gesture)
                                print(f"🎮 [Aktywacja gestu]: {top_gesture.upper()} ({count}/{win_size} klatek) -> {current_action}")
                        # Czyścimy okno po aktywacji, aby wymagać nowego ustabilizowania gestu
                        gesture_window.clear()

            # --- PODGLĄD WIDEO ---
            # 1. Bounding box dłoni
            if info["detected"] and info["box"] is not None:
                x1, y1, x2, y2 = info["box"]
                color = GESTURE_COLORS.get(raw_gesture, (0, 255, 0))
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                label = f"{raw_gesture.upper()} {conf*100:.0f}%"
                cv2.putText(vis, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)

            # 2. Miniatura wycinka dłoni (PiP)
            if info.get("crop") is not None:
                crop = cv2.resize(info["crop"], (80, 80))
                vis[h - 85:h - 5, w - 85:w - 5] = crop
                cv2.rectangle(vis, (w - 86, h - 86), (w - 4, h - 4), (0, 255, 255), 1)

            # 3. Pasek statusu
            cv2.rectangle(vis, (0, 0), (w, 28), (20, 20, 20), -1)
            hud_text = f"AKCJA: {current_action} | OKNO [{min_frames}/{win_size}]: {window_status_str}"
            cv2.putText(vis, hud_text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1, cv2.LINE_AA)


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
                if args.dry_run:
                    telemetry.update_flight(flight_state="FLYING", command="hover")

    finally:
        stop_event.set()
        telemetry.close()
        if client:
            client.shutdown()
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        if drone_thread and drone_thread.is_alive():
            print("⏳ Czekam na wylądowanie drona i zamknięcie połączenia...")
            drone_thread.join(timeout=4.0)
        print("✅ Program zakończony.")


if __name__ == "__main__":
    main()
