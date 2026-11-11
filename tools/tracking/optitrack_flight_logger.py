#!/usr/bin/env python3
"""
optitrack_flight_logger.py

Skrypt do akwizycji danych lotu drona Crazyflie w przestrzeni OptiTrack:
1. Odbiera w czasie rzeczywistym pozycję i orientację 3D z OptiTrack (UDP JSON).
2. Odbiera równolegle telemetrię z gesture_flight_controller.py (komendy ruchu, wykryty gest, stan drona).
3. Synchronizuje strumienie za pomocą precyzyjnych timestampów (czas absolutny Unix oraz względny time_rel).
4. Zapisuje dane do struktury CSV (analiza w Pandas/Matplotlib) oraz podsumowania JSON.

Autor: Praca Inżynierska - System Sterowania Dronem Gestami Dłoni
"""

import os
import sys
import csv
import time
import json
import socket
import signal
import argparse
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List


class TelemetryReceiver(threading.Thread):
    """Wątek odbierający pakiety telemetryczne z kontrolera gestów (UDP)."""

    def __init__(self, port: int = 12112, host: str = "127.0.0.1"):
        super().__init__(daemon=True)
        self.port = port
        self.host = host
        self.running = True
        self._lock = threading.Lock()

        self.last_packet_time = 0.0
        self.packet_count = 0
        self.latest_state = {
            "flight_state": "DISCONNECTED",
            "active_command": "none",
            "command_timestamp": 0.0,
            "detected_gesture": "none",
            "gesture_conf": 0.0,
            "gesture_timestamp": 0.0,
            "window_status": "brak",
            "sender_timestamp": 0.0,
        }

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except (AttributeError, OSError):
            pass
        self.sock.bind((self.host, self.port))
        self.sock.settimeout(0.5)

    def run(self):
        while self.running:
            try:
                data, _ = self.sock.recvfrom(2048)
                payload = json.loads(data.decode("utf-8"))
                if not isinstance(payload, dict):
                    continue
                now = time.time()
                with self._lock:
                    self.last_packet_time = now
                    self.packet_count += 1
                    self.latest_state["flight_state"] = str(payload.get("flight_state", "UNKNOWN"))
                    self.latest_state["active_command"] = str(payload.get("active_command", "none"))
                    self.latest_state["command_timestamp"] = float(payload.get("command_timestamp", now))
                    self.latest_state["detected_gesture"] = str(payload.get("detected_gesture") or "none")
                    self.latest_state["gesture_conf"] = float(payload.get("gesture_conf", 0.0))
                    self.latest_state["gesture_timestamp"] = float(payload.get("gesture_timestamp", now))
                    self.latest_state["window_status"] = str(payload.get("window_status", ""))
                    self.latest_state["sender_timestamp"] = float(payload.get("timestamp", now))
            except socket.timeout:
                continue
            except Exception:
                continue

    def get_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            snap = dict(self.latest_state)
            snap["telemetry_age_s"] = (time.time() - self.last_packet_time) if self.last_packet_time > 0 else 999.0
            snap["telemetry_connected"] = snap["telemetry_age_s"] < 2.5
            if not snap["telemetry_connected"]:
                snap["flight_state"] = "DISCONNECTED"
            return snap

    def stop(self):
        self.running = False
        try:
            self.sock.close()
        except Exception:
            pass


class OptiTrackFlightLogger:
    """Główny rejestrator danych OptiTrack oraz telemetrii lotu."""

    CSV_HEADER = [
        "timestamp",           # Czas rejestracji próbki (Unix timestamp [s])
        "time_rel",            # Czas od startu rejestracji [s]
        "ot_visible",          # Czy marker/rigid body widoczny w OptiTrack (1/0)
        "pos_x",               # Pozycja X po uwzględnieniu wybranej osi pionowej [m]
        "pos_y",               # Pozycja Y po uwzględnieniu wybranej osi pionowej [m]
        "pos_z",               # Wysokość Z (wysokość nad podłogą) [m]
        "raw_x",               # Surowa współrzędna X z OptiTrack [m]
        "raw_y",               # Surowa współrzędna Y z OptiTrack [m]
        "raw_z",               # Surowa współrzędna Z z OptiTrack [m]
        "roll",                # Kąt roll [rad lub deg wg serwera OT]
        "pitch",               # Kąt pitch [rad lub deg wg serwera OT]
        "yaw",                 # Kąt yaw [rad lub deg wg serwera OT]
        "flight_state",        # Stan operacyjny drona (GROUND, TAKING_OFF, FLYING, LANDING itp.)
        "command",             # Wykonywana komenda ruchu (forward, back, left, hover itp.)
        "gesture",             # Aktualnie wykryty gest (openhand, left, v, none itp.)
        "gesture_conf",        # Pewność klasyfikacji gestu (0.0 - 1.0)
        "window_status",       # Stan okna przesuwnego detektora
        "cmd_age_s",           # Czas [s] od wydania aktualnej komendy
        "gesture_age_s",       # Czas [s] od ostatniej detekcji gestu
        "telemetry_connected", # Flaga połączenia z kontrolerem gestów (1/0)
    ]

    def __init__(
        self,
        ot_port: int = 12111,
        ot_host: str = "",
        body_name: Optional[str] = None,
        up_axis: str = "z",
        telemetry_port: int = 12112,
        telemetry_host: str = "127.0.0.1",
        output_dir: str = "flight_logs",
    ):
        self.ot_port = ot_port
        self.ot_host = ot_host
        self.body_name = body_name
        self.up_axis = up_axis.lower()
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.running = True
        self.stopped = False
        self.detected_body_name = None

        # Odbiornik telemetrii z kontrolera gestów
        self.telemetry = TelemetryReceiver(port=telemetry_port, host=telemetry_host)

        # Gniazdo OptiTrack UDP
        self.ot_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.ot_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.ot_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except (AttributeError, OSError):
            pass
        self.ot_sock.bind((self.ot_host, self.ot_port))
        self.ot_sock.settimeout(0.5)

        # Pliki zapisu
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.output_dir / f"flight_session_{self.session_id}.csv"
        self.meta_path = self.output_dir / f"flight_session_{self.session_id}_meta.json"
        self.csv_file = None
        self.csv_writer = None

        # Statystyki sesji
        self.start_time = None
        self.sample_count = 0
        self.command_history: List[Dict[str, Any]] = []
        self.last_logged_command = None
        self.gesture_counts: Dict[str, int] = {}

    def _parse_optitrack_packet(self, data: bytes) -> Optional[Dict[str, Any]]:
        """Parsuje pakiet JSON z OptiTracka."""
        try:
            msg = json.loads(data.decode("utf-8"))
            if not isinstance(msg, dict):
                return None
        except Exception:
            return None

        # Wykrycie nazwy rigid body
        target_name = self.body_name
        if not target_name:
            if self.detected_body_name:
                target_name = self.detected_body_name
            else:
                for key, val in msg.items():
                    if key != "rigid_bodies" and isinstance(val, list) and len(val) >= 7:
                        self.detected_body_name = key
                        target_name = key
                        print(f"🎯 [OptiTrack] Autowykryto rigid body: '{target_name}'")
                        break

        if not target_name or target_name not in msg:
            return None

        arr = msg[target_name]
        if not isinstance(arr, list) or len(arr) < 7:
            return None

        try:
            visible, raw_x, raw_y, raw_z, roll, pitch, yaw = arr[:7]
            vis_int = int(visible)
            rx, ry, rz = float(raw_x), float(raw_y), float(raw_z)
            r, p, y = float(roll), float(pitch), float(yaw)
        except (ValueError, TypeError):
            return None

        # Konwersja osi (standardowo w robotyce Z to wysokość)
        if self.up_axis == "y":
            # OptiTrack standard Y-up -> mapujemy: X=X, Y=Z (poziom), Z=Y (wysokość)
            pos_x, pos_y, pos_z = rx, rz, ry
        else:
            # OptiTrack Z-up (standard robotyki ENU)
            pos_x, pos_y, pos_z = rx, ry, rz

        return {
            "body_name": target_name,
            "visible": vis_int,
            "pos_x": pos_x,
            "pos_y": pos_y,
            "pos_z": pos_z,
            "raw_x": rx,
            "raw_y": ry,
            "raw_z": rz,
            "roll": r,
            "pitch": p,
            "yaw": y,
        }

    def start(self):
        """Główna pętla akwizycji i rejestracji danych."""
        print("=" * 70)
        print("📊 [REJESTRATOR DANYCH LOTU I OPTITRACK]")
        print(f"   Port OptiTrack (UDP):  {self.ot_port}")
        print(f"   Port Telemetrii (UDP): {self.telemetry.port}")
        print(f"   Orientacja osi pion:   {self.up_axis.upper()}-UP")
        print(f"   Plik CSV:              {self.csv_path}")
        print("   Wciśnij Ctrl+C, aby zatrzymać i wygenerować raport sesji.")
        print("=" * 70)

        # Uruchomienie wątku telemetrii
        self.telemetry.start()

        # Otwarcie pliku CSV do buforowanego zapisu (zgodny z RFC 4180 csv.writer)
        self.csv_file = open(self.csv_path, mode="w", newline="", buffering=1024 * 64, encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(self.CSV_HEADER)

        self.start_time = time.time()
        last_print_time = self.start_time

        try:
            while self.running:
                try:
                    data, _ = self.ot_sock.recvfrom(2048)
                except socket.timeout:
                    continue
                except Exception:
                    break

                now = time.time()
                ot_data = self._parse_optitrack_packet(data)
                if not ot_data:
                    continue

                telem = self.telemetry.get_snapshot()

                time_rel = now - self.start_time
                cmd_age = now - telem["command_timestamp"] if telem["command_timestamp"] > 0 else 0.0
                gesture_age = now - telem["gesture_timestamp"] if telem["gesture_timestamp"] > 0 else 0.0

                # Statystyki gestów i komend
                cur_cmd = telem["active_command"]
                if cur_cmd != self.last_logged_command:
                    self.command_history.append({
                        "time_rel": round(time_rel, 3),
                        "command": cur_cmd,
                        "flight_state": telem["flight_state"],
                        "gesture": telem["detected_gesture"],
                    })
                    self.last_logged_command = cur_cmd

                cur_gest = telem["detected_gesture"]
                if cur_gest and cur_gest != "none":
                    self.gesture_counts[cur_gest] = self.gesture_counts.get(cur_gest, 0) + 1

                # Zapis wiersza CSV przez csv_writer
                row = [
                    f"{now:.6f}",
                    f"{time_rel:.4f}",
                    ot_data["visible"],
                    f"{ot_data['pos_x']:.5f}",
                    f"{ot_data['pos_y']:.5f}",
                    f"{ot_data['pos_z']:.5f}",
                    f"{ot_data['raw_x']:.5f}",
                    f"{ot_data['raw_y']:.5f}",
                    f"{ot_data['raw_z']:.5f}",
                    f"{ot_data['roll']:.5f}",
                    f"{ot_data['pitch']:.5f}",
                    f"{ot_data['yaw']:.5f}",
                    telem["flight_state"],
                    cur_cmd,
                    cur_gest,
                    f"{telem['gesture_conf']:.4f}",
                    telem["window_status"],
                    f"{cmd_age:.3f}",
                    f"{gesture_age:.3f}",
                    1 if telem["telemetry_connected"] else 0,
                ]
                self.csv_writer.writerow(row)
                self.sample_count += 1

                # Konsola statusu (co ~0.5s)
                if now - last_print_time >= 0.5:
                    hz = self.sample_count / (time_rel + 1e-5)
                    t_status = "🟢 POŁĄCZONY" if telem["telemetry_connected"] else "🔴 BRAK KONTROLERA"
                    print(
                        f"\r⏱️ {time_rel:5.1f}s | OT: {hz:5.1f}Hz | "
                        f"XYZ: [{ot_data['pos_x']:+.2f}, {ot_data['pos_y']:+.2f}, {ot_data['pos_z']:+.2f}]m | "
                        f"Komenda: {cur_cmd:<8} | Gest: {cur_gest:<8} | Telemetria: {t_status}",
                        end="",
                        flush=True,
                    )
                    last_print_time = now

        finally:
            self.stop()

    def stop(self):
        """Zatrzymanie zapisu i eksport metadanych JSON."""
        if self.stopped:
            return
        self.stopped = True
        self.running = False
        print("\n\n🛑 [Zatrzymano rejestrację]")

        if self.csv_file and not self.csv_file.closed:
            self.csv_file.flush()
            self.csv_file.close()

        self.telemetry.stop()
        try:
            self.ot_sock.close()
        except Exception:
            pass

        duration = (time.time() - self.start_time) if self.start_time else 0.0
        avg_hz = (self.sample_count / duration) if duration > 0 else 0.0

        meta = {
            "session_id": self.session_id,
            "created_at": datetime.now().isoformat(),
            "duration_seconds": round(duration, 3),
            "sample_count": self.sample_count,
            "avg_optitrack_hz": round(avg_hz, 1),
            "rigid_body_name": self.detected_body_name or self.body_name or "unknown",
            "up_axis": self.up_axis,
            "ot_port": self.ot_port,
            "telemetry_port": self.telemetry.port,
            "csv_path": str(self.csv_path),
            "command_history": self.command_history,
            "gesture_occurrences": self.gesture_counts,
        }

        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        print("=" * 70)
        print(f"✅ Zapisano pomyślnie sesję:")
        print(f"   • Plik CSV:       {self.csv_path} ({self.sample_count} próbek)")
        print(f"   • Metadane JSON:  {self.meta_path}")
        print(f"   • Średnia częst.: {avg_hz:.1f} Hz | Czas: {duration:.1f} s")
        print(f"   • Uruchom analizę wykresów:")
        print(f"     python3 plot_flight_data.py --csv {self.csv_path}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Rejestrator pozycji drona OptiTrack i telemetrii gestów")
    parser.add_argument("--ot-port", type=int, default=12111, help="Port UDP danych OptiTrack (domyślnie 12111)")
    parser.add_argument("--ot-host", default="", help="Host/interfejs dla OptiTrack UDP (domyślnie pusty = wszystkie)")
    parser.add_argument("--body-name", default=None, help="Nazwa rigid body w OptiTrack (np. QDrone, domyślnie autodetekcja)")
    parser.add_argument("--up-axis", choices=["z", "y"], default="z", help="Oś pionowa wysokości w sali: 'z' (standard robotyki) lub 'y' (domyślna OptiTrack)")
    parser.add_argument("--telemetry-port", type=int, default=12112, help="Port UDP telemetrii z gesture_flight_controller (domyślnie 12112)")
    parser.add_argument("--telemetry-host", default="127.0.0.1", help="Interfejs nasłuchu telemetrii UDP (domyślnie 127.0.0.1 localhost)")
    parser.add_argument("--output-dir", default=str(Path(__file__).parent / "flight_logs"), help="Katalog zapisu plików CSV/JSON")
    args = parser.parse_args()

    logger = OptiTrackFlightLogger(
        ot_port=args.ot_port,
        ot_host=args.ot_host,
        body_name=args.body_name,
        up_axis=args.up_axis,
        telemetry_port=args.telemetry_port,
        telemetry_host=args.telemetry_host,
        output_dir=args.output_dir,
    )

    def _sig_handler(sig, frame):
        logger.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    logger.start()


if __name__ == "__main__":
    main()
