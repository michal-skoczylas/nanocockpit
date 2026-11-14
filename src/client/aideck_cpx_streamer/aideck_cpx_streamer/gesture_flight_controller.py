#!/usr/bin/env python3
"""
gesture_flight_controller.py

Skrypt sterowania dronem Crazyflie za pomocą 8 gestów dłoni z podglądem z kamery AI-Deck,
zabezpieczeniami lotu oraz narzędziami do strojenia okna przesuwnego (sliding window) w laboratorium.

8 gestów i ich kierunki/zachowania:
  - openhand  -> Start (z ziemi) / Ruch w przód (w powietrzu)
  - shaka     -> Ruch w tył (mc.back)
  - left      -> Ruch w lewo (mc.left)
  - right     -> Ruch w prawo (mc.right)
  - v         -> Wznoszenie w górę (mc.up)
  - dislike   -> Opadanie w dół (mc.down)
  - ok        -> Bezpieczne lądowanie (mc.land)
  - closehand -> Zatrzymanie / zawis (mc.stop)
"""

import csv
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


# Bezpieczne, responsywne parametry lotu
URI = uri_helper.uri_from_env(default="radio://0/99/2M/E7E7E7E7E7")
DEFAULT_HEIGHT = 0.50   # wysokość bazowa [m] (50 cm)
STEP_DIST = 0.20        # krok ruchu [m] (20 cm - wyraźnie zauważalny ruch)
VELOCITY = 0.15         # prędkość ruchu [m/s] (15 cm/s)
CONF_THRESH = 0.40      # próg pewności ResNet (dla 8 klas: 0.40 to ponad 3x powyżej losowości 12.5%)
WATCHDOG_TIMEOUT = 2.0  # maks. czas [s] bez klatki z kamery przed lądowaniem awaryjnym
POST_TAKEOFF_LOCK = 0.8 # [s] blokada po starcie, aby gest 'openhand' nie wywołał od razu ruchu w przód

# ==============================================================================
# 🪟 OKNO PRZESUWNE DETEKCJI GESTÓW (Sliding Window)
# ==============================================================================
WINDOW_SIZE = 5            # N: rozmiar okna przesuwnego (liczba ostatnich klatek)
MIN_GESTURE_FRAMES = 3     # M: minimalna liczba klatek z danym gestem do wyzwolenia akcji
ACTIVATION_COOLDOWN = 0.5  # [s] minimalny odstęp czasu między kolejnymi aktywacjami


GESTURE_ACTIONS = {
    "openhand": "START / PRZÓD (forward)",
    "shaka": "TYŁ (back)",
    "left": "LEWO (left)",
    "right": "PRAWO (right)",
    "v": "GÓRA (up)",
    "dislike": "DÓŁ (down)",
    "closehand": "STOP / HOVER",
    "ok": "LĄDOWANIE (land)",
}

# Kolejki i zdarzenia synchronizujące
cmd_queue = queue.Queue(maxsize=1)
stop_event = threading.Event()
connected_event = threading.Event()
takeoff_event = threading.Event()
flying_event = threading.Event()
drone_busy_event = threading.Event()
first_frame_event = threading.Event()

_frame_time_lock = threading.Lock()
_last_frame_time = time.time()


def _touch_frame_time() -> None:
    global _last_frame_time
    with _frame_time_lock:
        _last_frame_time = time.time()


def _frame_age() -> float:
    with _frame_time_lock:
        return time.time() - _last_frame_time


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


def drone_worker(
    uri: str,
    telemetry: TelemetryPublisher = None,
    step_dist: float = STEP_DIST,
    velocity: float = VELOCITY,
):
    """Wątek drona: bezpieczny start, obsługa komend i lądowanie awaryjne."""
    if telemetry:
        telemetry.update_flight(flight_state="GROUND", command="idle")
    print(f"[Crazyflie] Inicjalizacja CRTP i łączenie z {uri}...")
    cflib.crtp.init_drivers()

    try:
        with SyncCrazyflie(uri, cf=Crazyflie(rw_cache="./cache")) as scf:
            connected_event.set()
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
                # Reset licznika klatek po starcie, aby sam czas wznoszenia nie wywołał watchdoga
                if first_frame_event.is_set():
                    _touch_frame_time()
                flying_event.set()
                print("🟢 [Crazyflie] Osiągnięto pułap! Dron w powietrzu - gotowy na komendy.")
                if telemetry:
                    telemetry.update_flight(flight_state="FLYING", command="hover")

                while not stop_event.is_set():
                    # Watchdog aktywny dopiero po otrzymaniu pierwszej klatki z kamery
                    if first_frame_event.is_set() and _frame_age() > WATCHDOG_TIMEOUT:
                        print(f"⚠️ [Watchdog Failsafe] Utrata klatek z kamery > {WATCHDOG_TIMEOUT}s! Ląduję...")
                        if telemetry:
                            telemetry.update_flight(flight_state="LANDING", command="failsafe_land")
                        stop_event.set()
                        break

                    try:
                        gesture = cmd_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    print(f"🎮 [Ruch drona]: {GESTURE_ACTIONS.get(gesture, gesture)}")

                    if gesture == "ok":
                        print("🛑 [Lądowanie] Gest OK - ląduję natychmiast.")
                        if telemetry:
                            telemetry.update_flight(flight_state="LANDING", command="land")
                        stop_event.set()
                        break

                    if gesture == "closehand":
                        print("✋ [Zawis / Stop] Pięść (closehand) - zatrzymanie w miejscu.")
                        if telemetry:
                            telemetry.update_flight(flight_state="FLYING", command="stop")
                        mc.stop()
                        if telemetry:
                            telemetry.update_flight(flight_state="FLYING", command="hover")
                        continue

                    move_map = {
                        "openhand": ("forward", mc.forward),
                        "shaka": ("back", mc.back),
                        "left": ("left", mc.left),
                        "right": ("right", mc.right),
                        "v": ("up", mc.up),
                        "dislike": ("down", mc.down),
                    }
                    if gesture in move_map:
                        cmd_name, move_fn = move_map[gesture]
                        drone_busy_event.set()
                        try:
                            if telemetry:
                                telemetry.update_flight(flight_state="FLYING", command=cmd_name)
                            move_fn(step_dist, velocity=velocity)
                        finally:
                            drone_busy_event.clear()
                            if telemetry and not stop_event.is_set():
                                telemetry.update_flight(flight_state="FLYING", command="hover")

            # MotionCommander.__exit__ wykonuje bezpieczne mc.land() przy wyjściu z bloku with
            if telemetry:
                telemetry.update_flight(flight_state="LANDED", command="idle")
    except Exception as e:
        print(f"❌ [Crazyflie Błąd]: {e}")
    finally:
        drone_busy_event.clear()
        flying_event.clear()
        connected_event.clear()
        if telemetry:
            telemetry.update_flight(flight_state="LANDED", command="idle")
        stop_event.set()
        print("[Crazyflie] Wątek drona zakończony bezpiecznie.")


def _resize_window_buffer(old_buf: deque, new_size: int) -> deque:
    """Tworzy nowy bufor o rozmiarze new_size zachowując najnowsze elementy."""
    items = list(old_buf)[-new_size:]
    return deque(items, maxlen=new_size)


def _draw_sliding_window_bar(vis, gesture_window: deque, win_size: int, min_frames: int) -> None:
    """Rysuje wizualizację zawartości okna przesuwnego u dołu podglądu."""
    h, w = vis.shape[:2]
    bar_h = 14
    margin_x = 8
    y2 = h - 8
    y1 = y2 - bar_h
    avail_w = max(40, w - 100)  # zostaw miejsce na PiP po prawej
    cell_gap = 3
    cell_w = max(8, min(28, (avail_w - margin_x - cell_gap * win_size) // max(1, win_size)))

    items = list(gesture_window)
    # Wyrównanie do prawej lub od lewej (najstarsza -> najnowsza)
    pad = win_size - len(items)
    padded = [None] * pad + items

    for idx, g in enumerate(padded):
        x1 = margin_x + idx * (cell_w + cell_gap)
        x2 = x1 + cell_w
        if x2 >= w - 92:
            break
        if g is None:
            color = (60, 60, 60)
        else:
            color = GESTURE_COLORS.get(g, (0, 255, 0))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (200, 200, 200), 1)

    label_x = min(w - 92, margin_x + min(win_size, len(padded)) * (cell_w + cell_gap) + 6)
    if label_x < w - 110:
        cv2.putText(
            vis,
            f"M={min_frames}/N={win_size}",
            (label_x, y2 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )


def main():
    parser = argparse.ArgumentParser(description="Sterowanie Crazyflie 8 gestami dłoni z podglądem wideo")
    parser.add_argument("-host", default="aideck.local", help="Host kamery AI-Deck")
    parser.add_argument("-port", type=int, default=5000, help="Port AI-Deck")
    parser.add_argument("--uri", default=URI, help="URI radia Crazyflie")
    parser.add_argument("--dry-run", action="store_true", help="Test wizji bez włączania drona (z symulacją maszyny stanów)")
    parser.add_argument("--mode", default="all", choices=["all", "binary"], help="Tryb gestów: 'all' (8 gestów) lub 'binary' (open/close)")
    parser.add_argument("--yolo-weights", default=None, help="Ścieżka do wag modelu YOLO (opcjonalnie)")
    parser.add_argument("--resnet-weights", default=None, help="Ścieżka do wag modelu ResNet (opcjonalnie)")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"], help="Urządzenie obliczeniowe")
    parser.add_argument("--conf", type=float, default=0.35, help="Próg detekcji dłoni YOLO")
    parser.add_argument("--resnet-conf", type=float, default=CONF_THRESH, help="Próg pewności klasyfikacji gestu ResNet (domyślnie 0.40)")
    parser.add_argument("--step-dist", type=float, default=STEP_DIST, help="Krok ruchu drona [m] (domyślnie 0.20m = 20cm)")
    parser.add_argument("--velocity", type=float, default=VELOCITY, help="Prędkość ruchu drona [m/s] (domyślnie 0.15m/s)")
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE, help="Rozmiar okna przesuwnego (N klatek)")
    parser.add_argument("--min-frames", type=int, default=MIN_GESTURE_FRAMES, help="Minimalna liczba klatek z gestem w oknie (M)")
    parser.add_argument("--cooldown", type=float, default=ACTIVATION_COOLDOWN, help="Minimalny odstęp [s] między kolejnymi aktywacjami")
    parser.add_argument("--webcam", action="store_true", help="Użyj kamerki laptopa zamiast Wi-Fi AI-Deck (testy offline)")
    parser.add_argument("--video", default=None, help="Ścieżka do nagranego pliku wideo do powtarzalnych testów sliding window")
    parser.add_argument("--record", default=None, help="Ścieżka do zapisu surowego wideo z kamery (np. lab_session.mp4)")
    parser.add_argument("--log-csv", default=None, help="Ścieżka do pliku CSV z logiem klatek i metryk sliding window")
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
        print(f"❌ [Błąd]: Plik modelu YOLO nie istnieje pod ścieżką:\n   {yolo_file}")
        sys.exit(1)
    if not resnet_all_file.exists():
        print(f"❌ [Błąd]: Plik modelu ResNet nie istnieje pod ścieżką:\n   {resnet_all_file}")
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

    # Inicjalizacja źródła wideo PRZED uruchomieniem wątku drona, aby błąd kamery nie zostawił wiszącego wątku
    cap = None
    client = None
    video_source_label = "AI-Deck Wi-Fi"
    if args.video:
        video_path = Path(args.video).expanduser().resolve()
        if not video_path.exists():
            print(f"❌ [Błąd]: Plik wideo nie istnieje: {video_path}")
            sys.exit(1)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ [Błąd]: Nie można otworzyć pliku wideo: {video_path}")
            sys.exit(1)
        video_source_label = f"Plik wideo ({video_path.name})"
    elif args.webcam:
        print("📷 [Kamera] Uruchamianie lokalnej kamerki...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ [Błąd]: Nie można otworzyć lokalnej kamerki.")
            sys.exit(1)
        video_source_label = "Lokalna kamerka"
    else:
        client = StreamerClient(host=args.host, port=args.port, udp_send=True)

    # Opcjonalny zapis CSV
    csv_file = None
    csv_writer = None
    if args.log_csv:
        csv_path = Path(args.log_csv).expanduser().resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "timestamp", "frame_idx", "fps", "latency_ms",
            "detected", "box_conf", "raw_gesture", "gesture_conf", "valid_gesture",
            "window_top", "window_count", "window_tie",
            "window_size", "min_frames", "resnet_conf", "cooldown",
            "activated_gesture", "paused", "drone_busy", "flight_state",
        ])
        print(f"📝 [Log CSV] Zapis metryk klatek do: {csv_path}")

    # Opcjonalny zapis wideo (inicjalizowany leniwie po pierwszej klatce, gdy znamy rozmiar)
    video_writer = None
    record_path = Path(args.record).expanduser().resolve() if args.record else None
    if record_path:
        record_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"🎥 [Nagrywanie] Zapis surowego wideo do: {record_path}")

    telemetry = TelemetryPublisher(
        host=args.telemetry_host,
        port=args.telemetry_port,
        enabled=not args.no_telemetry,
    )

    drone_thread = None
    if not args.dry_run:
        drone_thread = threading.Thread(
            target=drone_worker,
            args=(args.uri, telemetry, args.step_dist, args.velocity),
            daemon=False,
        )
        drone_thread.start()
    else:
        print("🟡 [TRYB DRY-RUN] Symulacja bez łączenia z dronem (pełna maszyna stanów).")
        connected_event.set()
        telemetry.update_flight(flight_state="GROUND", command="idle")

    def frame_stream():
        if cap is not None:
            while not stop_event.is_set():
                ret, f = cap.read()
                if not ret:
                    if args.video:
                        print("🎬 [Wideo] Koniec pliku wideo.")
                        break
                    time.sleep(0.03)
                    continue
                yield f, None, None
        else:
            for f, tof, m in client.receive():
                if stop_event.is_set():
                    break
                yield f, tof, m

    # Parametry sliding window (modyfikowalne na żywo w laboratorium)
    win_size = max(1, args.window_size)
    min_frames = min(max(1, args.min_frames), win_size)
    resnet_conf_thresh = max(0.05, min(0.99, args.resnet_conf))
    cooldown_sec = max(0.0, args.cooldown)

    gesture_window = deque(maxlen=win_size)
    last_activation_time = 0.0
    post_takeoff_lock_until = 0.0
    dry_run_takeoff_ready_at = None
    was_flying = False
    was_busy = False
    paused_commands = False

    current_action = "Czeka na start (OPENHAND / 't')"
    window_status_str = f"brak (0/{win_size})"

    # Statystyki sesji do analizy w laboratorium
    stats = {
        "frames": 0,
        "detected": 0,
        "accepted": 0,
        "rejected_conf": 0,
        "ties": 0,
        "activations": Counter(),
        "latencies_ms": [],
    }
    fps_ema = 0.0
    prev_frame_ts = time.time()

    print("\n" + "=" * 72)
    print("🚁 [KONTROLER CRAZYFLIE - TRYB LABORATORYJNY]")
    print(f"   Źródło wideo:   {video_source_label}")
    print(f"   Prędkość ruchu: {args.velocity*100:.0f} cm/s | Krok: {args.step_dist*100:.0f} cm")
    print(f"   Okno przesuwne: M={min_frames} z N={win_size} | Cooldown: {cooldown_sec:.1f}s | Próg ResNet: {resnet_conf_thresh*100:.0f}%")
    if not args.no_telemetry:
        print(f"   Telemetria UDP: {args.telemetry_host}:{args.telemetry_port}")
    print("   Klawisze lotu:  [t] Start | [l] Ląduj | [p] Pauza komend | [q/ESC] Wyjście")
    print("   Strojenie okna: [1/2] Conf -/+5% | [3/4] N -/+1 | [5/6] M -/+1 | [7/8] Cooldown -/+0.1s | [r] Reset")
    print("=" * 72 + "\n")

    try:
        for frame, tof_frame, meta in frame_stream():
            if stop_event.is_set():
                break
            if client and meta:
                client.send_reply(meta, None)

            now = time.time()
            dt = max(1e-4, now - prev_frame_ts)
            prev_frame_ts = now
            inst_fps = 1.0 / dt
            fps_ema = inst_fps if fps_ema == 0.0 else (0.85 * fps_ema + 0.15 * inst_fps)

            _touch_frame_time()
            first_frame_event.set()

            vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if len(frame.shape) == 2 else frame.copy()
            h, w = vis.shape[:2]

            # Zapis surowej klatki do wideo (jeśli włączono --record)
            if record_path is not None:
                if video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    rec_fps = 15.0 if args.webcam or args.video else 10.0
                    video_writer = cv2.VideoWriter(str(record_path), fourcc, rec_fps, (w, h))
                video_writer.write(vis)

            # Obsługa przejścia stanu TAKING_OFF -> FLYING w trybie dry-run
            if args.dry_run and takeoff_event.is_set() and not flying_event.is_set():
                if dry_run_takeoff_ready_at is not None and now >= dry_run_takeoff_ready_at:
                    flying_event.set()
                    telemetry.update_flight(flight_state="FLYING", command="hover")

            # Wykrycie momentu zakończenia startu (zapobiega natychmiastowemu 'forward' po starcie gestem 'openhand')
            is_flying = flying_event.is_set()
            if is_flying and not was_flying:
                gesture_window.clear()
                post_takeoff_lock_until = now + POST_TAKEOFF_LOCK
                last_activation_time = 0.0
                current_action = "ZAWIS (HOVER) - Gotowy na gesty"
            was_flying = is_flying

            # Wykrycie zakończenia kroku lotu -> czyścimy okno, aby nie wykorzystywać klatek z trakcie ruchu
            is_busy = drone_busy_event.is_set()
            if was_busy and not is_busy:
                gesture_window.clear()
            was_busy = is_busy

            # Detekcja AI
            info = detector.process_frame(vis)
            raw_gesture = info["gesture"]
            conf = float(info["gesture_conf"])
            box_conf = float(info.get("box_conf", 0.0))
            latency_ms = float(info.get("latency_ms", 0.0))

            stats["frames"] += 1
            stats["latencies_ms"].append(latency_ms)
            if info["detected"]:
                stats["detected"] += 1

            # Kwalifikacja klatki
            has_valid_prediction = info["detected"] and raw_gesture != "Brak dłoni"
            if has_valid_prediction and conf >= resnet_conf_thresh:
                valid_frame_gesture = raw_gesture
                stats["accepted"] += 1
            else:
                valid_frame_gesture = None
                if has_valid_prediction:
                    stats["rejected_conf"] += 1

            # Jeśli dron wykonuje krok, nie zbieramy klatek do okna (chyba że trzeba wykryć pilne 'ok')
            if is_busy and valid_frame_gesture != "ok":
                gesture_window.append(None)
            else:
                gesture_window.append(valid_frame_gesture)

            # Analiza okna przesuwnego z wykrywaniem remisów
            valid_gestures = [g for g in gesture_window if g is not None]
            gesture_counts = Counter(valid_gestures)
            most_common = gesture_counts.most_common(2)
            is_tie = len(most_common) == 2 and most_common[0][1] == most_common[1][1]
            top_gesture, count = most_common[0] if most_common else (None, 0)

            if is_tie and count >= min_frames:
                stats["ties"] += 1

            if is_tie:
                window_status_str = f"REMIS ({count}/{win_size})"
            elif top_gesture:
                window_status_str = f"{top_gesture.upper()}: {count}/{win_size}"
            else:
                window_status_str = f"brak (0/{win_size})"

            # Publikacja stanu detekcji w telemetrii
            telemetry.update_vision(
                gesture=valid_frame_gesture,
                conf=conf if info["detected"] else 0.0,
                window_status=window_status_str,
            )

            activated_gesture = None

            # Sprawdzenie warunku aktywacji (M z N klatek, brak remisu, brak blokady)
            if top_gesture and not is_tie and count >= min_frames:
                cooldown_ok = (now - last_activation_time >= cooldown_sec)
                post_takeoff_ok = (now >= post_takeoff_lock_until)

                if cooldown_ok and post_takeoff_ok:
                    if paused_commands:
                        # W trybie pauzy testujemy samo okno, bez wysyłania komend do drona
                        activated_gesture = f"{top_gesture}(PAUSED)"
                        last_activation_time = now
                        stats["activations"][top_gesture] += 1
                        print(f"⏸️ [PAUZA - TEST OKNA]: {top_gesture.upper()} ({count}/{win_size}, conf={conf*100:.0f}%)")
                        gesture_window.clear()
                    elif not takeoff_event.is_set():
                        if not connected_event.is_set():
                            current_action = "Łączenie z Crazyflie..."
                            gesture_window.clear()
                        elif top_gesture == "openhand":
                            activated_gesture = "openhand(TAKEOFF)"
                            last_activation_time = now
                            stats["activations"]["takeoff"] += 1
                            print(f"🛫 [Start] Wykryto OPENHAND ({count}/{win_size} klatek) - startuję drona!")
                            takeoff_event.set()
                            current_action = "Wznoszenie na 50 cm..."
                            if args.dry_run:
                                telemetry.update_flight(flight_state="TAKING_OFF", command="takeoff")
                                dry_run_takeoff_ready_at = now + 0.6
                            gesture_window.clear()
                    elif not is_flying:
                        current_action = "Wznoszenie na 50 cm..."
                        gesture_window.clear()
                    elif is_busy and top_gesture != "ok":
                        # Dron jest w trakcie ruchu - ignorujemy inne gesty niż awaryjne lądowanie 'ok'
                        pass
                    else:
                        activated_gesture = top_gesture
                        last_activation_time = now
                        stats["activations"][top_gesture] += 1
                        current_action = GESTURE_ACTIONS.get(top_gesture, top_gesture)

                        if args.dry_run:
                            print(f"🟡 [DRY-RUN]: {top_gesture.upper()} ({count}/{win_size} klatek, conf={conf*100:.0f}%) -> {current_action}")
                            telemetry.update_flight(flight_state="FLYING", command=top_gesture)
                            if top_gesture == "ok":
                                telemetry.update_flight(flight_state="LANDED", command="idle")
                                stop_event.set()
                        else:
                            try:
                                cmd_queue.put_nowait(top_gesture)
                                print(f"🎮 [Aktywacja gestu]: {top_gesture.upper()} ({count}/{win_size} klatek, conf={conf*100:.0f}%) -> {current_action}")
                            except queue.Full:
                                if top_gesture == "ok":
                                    try:
                                        cmd_queue.get_nowait()
                                    except queue.Empty:
                                        pass
                                    cmd_queue.put_nowait(top_gesture)
                                    print("🛑 [Priorytet OK] Nadpisano kolejkę komendą lądowania!")
                        gesture_window.clear()

            # Zapis wiersza do CSV
            if csv_writer:
                csv_writer.writerow([
                    f"{now:.4f}", stats["frames"], f"{fps_ema:.2f}", f"{latency_ms:.2f}",
                    int(info["detected"]), f"{box_conf:.4f}",
                    raw_gesture if has_valid_prediction else "", f"{conf:.4f}",
                    valid_frame_gesture or "",
                    top_gesture or "", count, int(is_tie),
                    win_size, min_frames, f"{resnet_conf_thresh:.2f}", f"{cooldown_sec:.2f}",
                    activated_gesture or "", int(paused_commands), int(is_busy),
                    telemetry.state["flight_state"],
                ])

            # --- PODGLĄD WIDEO ---
            if info["detected"] and info["box"] is not None:
                x1, y1, x2, y2 = info["box"]
                color = GESTURE_COLORS.get(raw_gesture, (0, 255, 0)) if has_valid_prediction else (140, 140, 140)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                label = f"{raw_gesture.upper()} {conf*100:.0f}%" if has_valid_prediction else "DLON (brak klasy)"
                cv2.putText(vis, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)

            if info.get("crop") is not None:
                crop = cv2.resize(info["crop"], (80, 80))
                vis[h - 85:h - 5, w - 85:w - 5] = crop
                cv2.rectangle(vis, (w - 86, h - 86), (w - 4, h - 4), (0, 255, 255), 1)

            # Pasek statusu (góra)
            cv2.rectangle(vis, (0, 0), (w, 46), (20, 20, 20), -1)
            status_flags = []
            if paused_commands:
                status_flags.append("PAUZA")
            if is_busy:
                status_flags.append("RUCH")
            flag_str = f" [{' '.join(status_flags)}]" if status_flags else ""

            det_label = f"{raw_gesture.upper()} ({conf*100:.0f}%)" if has_valid_prediction else "brak dloni"
            hud_line1 = f"AKCJA: {current_action}{flag_str} | DET: {det_label}"
            hud_line2 = f"OKNO: {window_status_str} | N={win_size} M={min_frames} Conf>={resnet_conf_thresh*100:.0f}% CD={cooldown_sec:.1f}s | {fps_ema:.1f} FPS ({latency_ms:.0f}ms)"
            cv2.putText(vis, hud_line1, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(vis, hud_line2, (8, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 220, 255), 1, cv2.LINE_AA)

            # Wizualizacja bufora okna przesuwnego (dół)
            _draw_sliding_window_bar(vis, gesture_window, win_size, min_frames)

            cv2.imshow("NanoCockpit AI-Deck: Podglad Kamery Drona", vis)

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
                if not connected_event.is_set():
                    print("⏳ [Klawiatura] Dron jeszcze się łączy - poczekaj na komunikat 'Połączono!'.")
                    current_action = "Łączenie z Crazyflie..."
                else:
                    print("🛫 [Klawiatura] Start drona klawiszem 't'")
                    takeoff_event.set()
                    current_action = "Wznoszenie na 50 cm..."
                    gesture_window.clear()
                    try:
                        cmd_queue.get_nowait()
                    except queue.Empty:
                        pass
                    if args.dry_run:
                        telemetry.update_flight(flight_state="TAKING_OFF", command="takeoff")
                        dry_run_takeoff_ready_at = time.time() + 0.6
            elif key == ord("l") and flying_event.is_set():
                print("🛑 [Klawiatura] Lądowanie awaryjne klawiszem 'l'")
                if args.dry_run:
                    telemetry.update_flight(flight_state="LANDED", command="idle")
                    stop_event.set()
                else:
                    try:
                        cmd_queue.get_nowait()
                    except queue.Empty:
                        pass
                    cmd_queue.put_nowait("ok")
            elif key == ord("p"):
                paused_commands = not paused_commands
                gesture_window.clear()
                print(f"{'⏸️ [PAUZA]' if paused_commands else '▶️ [WZNOWIONO]'} Wysyłanie komend do drona: {'WSTRZYMANE (test okna)' if paused_commands else 'AKTYWNE'}")
            elif key == ord("r"):
                gesture_window.clear()
                print("🔄 [Reset] Wyczyszczono bufor okna przesuwnego.")
            elif key == ord("1"):
                resnet_conf_thresh = max(0.05, round(resnet_conf_thresh - 0.05, 2))
                print(f"🎚️ Próg ResNet: {resnet_conf_thresh*100:.0f}%")
            elif key == ord("2"):
                resnet_conf_thresh = min(0.95, round(resnet_conf_thresh + 0.05, 2))
                print(f"🎚️ Próg ResNet: {resnet_conf_thresh*100:.0f}%")
            elif key == ord("3"):
                win_size = max(1, win_size - 1)
                min_frames = min(min_frames, win_size)
                gesture_window = _resize_window_buffer(gesture_window, win_size)
                print(f"🎚️ Rozmiar okna N={win_size}, M={min_frames}")
            elif key == ord("4"):
                win_size = min(30, win_size + 1)
                gesture_window = _resize_window_buffer(gesture_window, win_size)
                print(f"🎚️ Rozmiar okna N={win_size}, M={min_frames}")
            elif key == ord("5"):
                min_frames = max(1, min_frames - 1)
                print(f"🎚️ Próg klatek M={min_frames}/{win_size}")
            elif key == ord("6"):
                min_frames = min(win_size, min_frames + 1)
                print(f"🎚️ Próg klatek M={min_frames}/{win_size}")
            elif key == ord("7"):
                cooldown_sec = max(0.0, round(cooldown_sec - 0.1, 1))
                print(f"🎚️ Cooldown: {cooldown_sec:.1f}s")
            elif key == ord("8"):
                cooldown_sec = min(5.0, round(cooldown_sec + 0.1, 1))
                print(f"🎚️ Cooldown: {cooldown_sec:.1f}s")

    finally:
        stop_event.set()
        telemetry.close()
        if csv_file:
            csv_file.close()
        if video_writer:
            video_writer.release()
        if client:
            client.shutdown()
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        if drone_thread and drone_thread.is_alive():
            print("⏳ Czekam na wylądowanie drona i zamknięcie połączenia...")
            drone_thread.join(timeout=10.0)

        # Podsumowanie sesji laboratoryjnej
        if stats["frames"] > 0:
            avg_lat = sum(stats["latencies_ms"]) / len(stats["latencies_ms"])
            det_pct = (stats["detected"] / stats["frames"]) * 100.0
            acc_pct = (stats["accepted"] / stats["frames"]) * 100.0
            print("\n📊 [Podsumowanie sesji laboratoryjnej]")
            print(f"   Klatki: {stats['frames']} | Wykryta dłoń: {det_pct:.1f}% | Zaakceptowane (>={resnet_conf_thresh*100:.0f}%): {acc_pct:.1f}%")
            print(f"   Odrzucone przez próg pewności: {stats['rejected_conf']} | Remisy w oknie: {stats['ties']}")
            print(f"   Średnia latencja AI: {avg_lat:.1f} ms | Końcowe okno: N={win_size}, M={min_frames}, CD={cooldown_sec:.1f}s")
            if stats["activations"]:
                print(f"   Aktywacje gestów: {dict(stats['activations'])}")

        print("✅ Program zakończony.")


if __name__ == "__main__":
    main()
