#!/usr/bin/env python3
"""
gesture_detector_viewer.py

Zaawansowany klient strumienia AI-deck (NanoCockpit / Crazyflie) z detekcją dłoni YOLO
oraz klasyfikacją gestów ResNet w czasie rzeczywistym.

Główne funkcje:
1. Odbiór klatek z kamery AI-deck (oraz danych z czujnika TOF i telemetrii) przez CPX Streamer.
2. Detekcja dłoni w kadrze przy użyciu dedykowanego modelu YOLO (weights/yolo_hand_detector.pt).
3. Klasyfikacja gestu za pomocą zoptymalizowanej sieci ResNet (TorchScript / PyTorch).
4. Przełączanie trybów:
   - Tryb binarny: 'closehand' (pięść) vs 'openhand' (otwarta dłoń)
   - Tryb pełny: 8 gestów (closehand, dislike, left, ok, openhand, right, shaka, v)
   - Dynamiczne przełączanie w locie klawiszem 'm' (w trybie CV2) lub parametrem --mode.
5. Wizualizacja:
   - Ramka detekcji YOLO wokół dłoni wraz z etykietą gestu i pewnością (%).
   - Miniatura wyciętej dłoni w rogu (Picture-in-Picture).
   - Pomiary opóźnień, FPS oraz filtracja czasowa (temporal smoothing).

Użycie:
    # 1. Tryb binarny (open / closed):
    python gesture_detector_viewer.py --mode binary

    # 2. Tryb wszystkich 8 gestów:
    python gesture_detector_viewer.py --mode all

    # 3. Z szybkim oknem OpenCV zamiast Matplotlib:
    python gesture_detector_viewer.py --viewer cv2 --mode binary
"""

import os
import sys
import time
import argparse
from pathlib import Path
from collections import deque
from typing import Optional, Tuple, List, Dict, Any

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO

# Upewnij się, że lokalny moduł cpx oraz utils są w sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from cpx import StreamerClient, StreamerMetadata
    from utils import create_dataset_dir, FrameSaver
except ImportError:
    # Fallback jeśli uruchamiane z innego katalogu roboczego
    try:
        from aideck_cpx_streamer.cpx import StreamerClient, StreamerMetadata
        from aideck_cpx_streamer.utils import create_dataset_dir, FrameSaver
    except ImportError:
        raise ImportError(
            "Nie można zaimportować modułów 'cpx' i 'utils'. "
            "Upewnij się, że skrypt jest uruchamiany w środowisku NanoCockpit."
        )

# Domyślne etykiety klas
BINARY_CLASSES = ["closehand", "openhand"]
ALL_CLASSES = ["closehand", "dislike", "left", "ok", "openhand", "right", "shaka", "v"]

# Paleta kolorów BGR dla poszczególnych gestów
GESTURE_COLORS = {
    "openhand": (40, 200, 40),      # Zielony (Start / Otwarcie)
    "closehand": (0, 70, 230),      # Czerwony / Pomarańcz (Stop / Pięść)
    "dislike": (0, 165, 255),       # Pomarańczowy (Lądowanie)
    "left": (230, 180, 0),          # Błękit / Cyjan (Lewo)
    "right": (200, 50, 200),        # Fiolet (Prawo)
    "ok": (255, 255, 0),            # Żółty (Hover / Potwierdzenie)
    "shaka": (0, 230, 230),         # Żółto-zielony (RTH)
    "v": (220, 120, 50),            # Niebieski (Foto / Peace)
}


def find_weights_file(candidates: List[Path]) -> Path:
    """Szuka pliku wag pośród listy potencjalnych ścieżek względnych i bezwzględnych."""
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    # Jeśli nie znaleziono, zwróć pierwszą jako domyślną
    return candidates[0]


class DroneGestureDetector:
    """Klasa zarządzająca modelami YOLO oraz ResNet do inferencji w czasie rzeczywistym."""

    def __init__(
        self,
        yolo_path: Path,
        classifier_bin_path: Path,
        classifier_all_path: Path,
        mode: str = "binary",
        conf_thresh: float = 0.35,
        device_str: str = "auto",
        smooth_frames: int = 5,
    ):
        self.mode = mode.lower()  # "binary" lub "all"
        self.conf_thresh = conf_thresh
        self.smooth_frames = smooth_frames

        # Dobór urządzenia
        if device_str == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device_str)

        print(f"🍎 [Inference Engine] Urządzenie obliczeniowe: {self.device.type.upper()}")

        # 1. Załadowanie detektora YOLO
        print(f"📦 [YOLO] Ładowanie wag: {yolo_path}")
        if not yolo_path.exists():
            raise FileNotFoundError(f"Nie znaleziono wag detektora YOLO: {yolo_path}")
        self.yolo_model = YOLO(str(yolo_path))

        # 2. Załadowanie klasyfikatorów ResNet
        print(f"📦 [ResNet Binary] Ładowanie wag: {classifier_bin_path}")
        self.model_bin = self._load_classifier(classifier_bin_path, num_classes=2)
        self.classes_bin = BINARY_CLASSES

        print(f"📦 [ResNet All-8] Ładowanie wag: {classifier_all_path}")
        self.model_all = self._load_classifier(classifier_all_path, num_classes=8)
        self.classes_all = ALL_CLASSES

        # Transformacja wejściowa ImageNet
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Bufor do wygładzania predykcji (Temporal Smoothing)
        self.history = deque(maxlen=self.smooth_frames)

    def _load_classifier(self, path: Path, num_classes: int) -> torch.nn.Module:
        """Ładuje model ze skryptu TorchScript lub standardowego pliku wag state_dict."""
        if not path.exists():
            raise FileNotFoundError(f"Nie znaleziono wag klasyfikatora: {path}")

        try:
            # Próba wczytania jako TorchScript
            model = torch.jit.load(str(path), map_location=self.device)
            model.eval()
            return model
        except Exception:
            # Fallback dla czystych wag state_dict torchvision ResNet
            from torchvision import models
            arch = models.resnet34 if "res34" in str(path) else models.resnet18
            model = arch(weights=None)
            in_f = model.fc.in_features
            model.fc = torch.nn.Linear(in_f, num_classes)
            state_dict = torch.load(str(path), map_location=self.device)
            # Obsługa wag pochodzących z PyTorch Lightning (prefix 'model.')
            cleaned_state = {}
            for k, v in state_dict.items():
                new_k = k.replace("model.", "").replace("backbone.", "")
                cleaned_state[new_k] = v
            try:
                model.load_state_dict(cleaned_state, strict=False)
            except Exception:
                model.load_state_dict(state_dict, strict=False)
            model.to(self.device)
            model.eval()
            return model

    def toggle_mode(self) -> str:
        """Przełącza w locie tryb pracy: binary <-> all."""
        self.mode = "all" if self.mode == "binary" else "binary"
        self.history.clear()
        print(f"🔄 [Tryb zmieniony]: {'WSZYSTKIE 8 GESTÓW' if self.mode == 'all' else 'TYLKO OPEN/CLOSE HAND'}")
        return self.mode

    def process_frame(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Główny potok przetwarzania pojedynczej klatki:
        1. Detekcja dłoni YOLO -> bounding box
        2. Wycięcie i normalizacja kadru dłoni
        3. Klasyfikacja gestu przez model ResNet
        """
        t0 = time.time()
        h, w = frame_bgr.shape[:2]

        # Wybór aktywnego klasyfikatora i etykiet
        if self.mode == "binary":
            active_model = self.model_bin
            active_classes = self.classes_bin
        else:
            active_model = self.model_all
            active_classes = self.classes_all

        # 1. Detekcja dłoni YOLO
        yolo_res = self.yolo_model.predict(
            frame_bgr,
            conf=self.conf_thresh,
            verbose=False,
            device=0 if self.device.type == "cuda" else self.device.type,
        )[0]

        detected_hand = False
        box_coords = None
        box_conf = 0.0
        gesture_name = "Brak dłoni"
        gesture_conf = 0.0
        hand_crop = None

        if len(yolo_res.boxes) > 0:
            detected_hand = True
            # Wybierz box o najwyższej pewności detekcji
            best_box = max(yolo_res.boxes, key=lambda b: b.conf[0].item())
            box_conf = best_box.conf[0].item()
            xyxy = best_box.xyxy[0].cpu().numpy().astype(int)

            # Dodaj 10% marginesu wokół dłoni (padding)
            bw = xyxy[2] - xyxy[0]
            bh = xyxy[3] - xyxy[1]
            pad_x = int(bw * 0.10)
            pad_y = int(bh * 0.10)

            x1 = max(0, xyxy[0] - pad_x)
            y1 = max(0, xyxy[1] - pad_y)
            x2 = min(w, xyxy[2] + pad_x)
            y2 = min(h, xyxy[3] + pad_y)
            box_coords = (int(x1), int(y1), int(x2), int(y2))

            # Wycięcie dłoni
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size > 0 and crop.shape[0] >= 10 and crop.shape[1] >= 10:
                hand_crop = crop.copy()
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                input_tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)

                # 2. Klasyfikacja ResNet
                with torch.no_grad():
                    logits = active_model(input_tensor)
                    probs = torch.softmax(logits, dim=1)[0]
                    conf, pred_idx = torch.max(probs, dim=0)

                raw_gesture = active_classes[pred_idx.item()]
                raw_conf = conf.item()

                # Temporal smoothing (filtracja większościowa z bufora)
                self.history.append(raw_gesture)
                # Wybierz najczęstszą predykcję z ostatnich N klatek
                voted_gesture = max(set(self.history), key=self.history.count)
                gesture_name = voted_gesture
                gesture_conf = raw_conf
        else:
            self.history.clear()

        latency_ms = (time.time() - t0) * 1000.0

        return {
            "detected": detected_hand,
            "box": box_coords,
            "box_conf": box_conf,
            "gesture": gesture_name,
            "gesture_conf": gesture_conf,
            "crop": hand_crop,
            "mode": self.mode,
            "latency_ms": latency_ms,
        }


class GestureDetectorViewer:
    """Główna aplikacja integrująca odbiór strumienia z drona i wyświetlanie predykcji."""

    def __init__(self) -> None:
        parser = argparse.ArgumentParser(description="AI-deck Streamer z Detekcją i Klasyfikacją Gestów")
        parser.add_argument("-host", default="aideck.local", metavar="host", help="Adres IP / host drona AI-deck")
        parser.add_argument("-port", type=int, default=5000, metavar="port", help="Port AI-deck (domyślnie 5000)")
        parser.add_argument("--no-udp-send", action="store_false", dest="udp_send", help="Wyłącz odpowiedzi UDP")
        parser.add_argument("-save", type=str, default=None, metavar="save", help="Zapis klatek do katalogu")

        # Parametry AI
        parser.add_argument("--mode", type=str, default="binary", choices=["binary", "all"],
                            help="Tryb klasyfikacji gestów: 'binary' (close/open) lub 'all' (8 gestów)")
        parser.add_argument("--conf", type=float, default=0.35, help="Próg pewności detektora YOLO (domyślnie 0.35)")
        parser.add_argument("--viewer", type=str, default="cv2", choices=["cv2", "plt"],
                            help="Silnik wyświetlania okna: 'cv2' (szybki OpenCV) lub 'plt' (Matplotlib)")
        parser.add_argument("--smooth", type=int, default=5, help="Rozmiar bufora wygładzania czasowego gestów (domyślnie 5)")
        parser.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cuda", "cpu"],
                            help="Urządzenie akceleracji inferencji (domyślnie auto = MPS na Macu)")

        # Ścieżki do wag modeli
        parser.add_argument("--yolo-weights", type=str, default=None, help="Ścieżka do wag YOLO (opcjonalna)")
        parser.add_argument("--classifier-binary", type=str, default=None, help="Ścieżka do wag ResNet Binary")
        parser.add_argument("--classifier-all", type=str, default=None, help="Ścieżka do wag ResNet All-8")
        parser.add_argument("--demo", action="store_true", help="Tryb demo bez podłączonego drona (odtwarza klatki z drona)")
        parser.add_argument("--source", type=str, default="aideck", help="Źródło obrazu: 'aideck' (dron), 'demo' (zbiór drona), 'webcam' (kamera Maca), lub ścieżka do pliku wideo")

        self.args = parser.parse_args()

        # Automatyczne odnajdywanie wag modeli w strukturze projektu
        base_dir = Path(__file__).resolve().parent
        thesis_dir = base_dir.parents[4] if len(base_dir.parents) >= 5 else base_dir
        self.dane_weights = thesis_dir / "dane" / "weights"
        dane_weights = self.dane_weights

        yolo_candidates = [
            Path(self.args.yolo_weights) if self.args.yolo_weights else None,
            dane_weights / "yolo_hand_detector.pt",
            base_dir / "weights" / "yolo_hand_detector.pt",
            Path("weights/yolo_hand_detector.pt"),
        ]
        bin_candidates = [
            Path(self.args.classifier_binary) if self.args.classifier_binary else None,
            dane_weights / "resnet18_binary_best.torchscript.pt",
            dane_weights / "exp_v2_resnet18_binary_hires_drop_best.pt",
            Path("weights/resnet18_binary_best.torchscript.pt"),
        ]
        all_candidates = [
            Path(self.args.classifier_all) if self.args.classifier_all else None,
            dane_weights / "resnet_all_gestures" / "resnet_all8_best.torchscript.pt",
            dane_weights / "resnet_all_gestures" / "resnet18_all8_best.torchscript.pt",
            dane_weights / "resnet_all_gestures" / "exp_v4_res34_all8_hires_best.pt",
            Path("weights/resnet_all_gestures/resnet_all8_best.torchscript.pt"),
        ]

        yolo_path = find_weights_file([p for p in yolo_candidates if p is not None])
        bin_path = find_weights_file([p for p in bin_candidates if p is not None])
        all_path = find_weights_file([p for p in all_candidates if p is not None])

        # Inicjalizacja silnika detekcji i klasyfikacji
        self.detector = DroneGestureDetector(
            yolo_path=yolo_path,
            classifier_bin_path=bin_path,
            classifier_all_path=all_path,
            mode=self.args.mode,
            conf_thresh=self.args.conf,
            device_str=self.args.device,
            smooth_frames=self.args.smooth,
        )

        # Klient AI-deck streamer (tylko w trybie aideck)
        self.is_demo = self.args.demo or self.args.source in ("demo", "dataset", "webcam", "0") or Path(self.args.source).exists()
        if not self.is_demo:
            self.client = StreamerClient(
                host=self.args.host, port=self.args.port, udp_send=self.args.udp_send
            )
        else:
            self.client = None

        self.frame_saver = None
        if self.args.save is not None:
            dataset_dir = create_dataset_dir(self.args.save)
            self.frame_saver = FrameSaver(dataset_dir)

        # Licznik FPS
        self.fps_tracker = deque(maxlen=20)
        self.last_time = time.time()

        # Inicjalizacja wybranego widoku
        self.viewer_type = self.args.viewer
        if self.viewer_type == "plt":
            import matplotlib.pyplot as plt
            plt.ion()
            self.fig, (self.ax, self.tof_ax) = plt.subplots(1, 2, figsize=(11, 5))
            self.fig.canvas.mpl_connect("close_event", self.on_close)
            self.fig.canvas.mpl_connect("key_press_event", self.on_plt_key)
            self.ax.axis("off")
            self.tof_ax.axis("off")
            self.im = None
            self.tof = None

    def main(self):
        print("\n" + "=" * 70)
        print("🚁 [AI-Deck Real-Time Gesture & Hand Detector]")
        print(f"   Host drona  : {self.args.host}:{self.args.port}")
        print(f"   Tryb gestów : {'TYLKO CLOSE/OPEN HAND' if self.detector.mode == 'binary' else 'WSZYSTKIE 8 GESTÓW'}")
        print(f"   Silnik GUI  : {self.viewer_type.upper()} (Naciśnij 'm' aby zmienić tryb, 'q' aby wyjść)")
        print("=" * 70 + "\n")

        try:
            if self.frame_saver:
                self.frame_saver.open()

            def frame_generator():
                if not self.is_demo:
                    for f, t, m in self.client.receive():
                        self.client.send_reply(m, None)
                        yield f, t, m
                elif self.args.source in ("webcam", "0"):
                    cap = cv2.VideoCapture(0)
                    while cap.isOpened():
                        ret, f = cap.read()
                        if not ret: break
                        yield f, None, None
                    cap.release()
                else:
                    # Odtwarzaj klatki ze zbioru drona lub pliku
                    val_dir = self.dane_weights.parent / "data" / "yolo_dataset" / "images" / "val"
                    img_files = sorted(list(val_dir.glob("*.jpg"))) if val_dir.exists() else []
                    if not img_files and Path(self.args.source).exists():
                        cap = cv2.VideoCapture(self.args.source)
                        while cap.isOpened():
                            ret, f = cap.read()
                            if not ret: break
                            yield f, None, None
                        cap.release()
                        return
                    print(f"🎬 [Tryb Demo] Odtwarzanie {len(img_files)} klatek z kamery drona...")
                    while True:
                        for p in img_files:
                            img = cv2.imread(str(p))
                            time.sleep(0.04) # ~25 FPS
                            yield img, None, None

            for frame, tof_frame, metadata in frame_generator():

                # Konwersja surowej klatki do obrazu 3-kanałowego BGR
                if len(frame.shape) == 2 or frame.shape[2] == 1:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                else:
                    frame_bgr = frame

                # Przetwarzanie klatki przez potok YOLO + ResNet
                ai_info = self.detector.process_frame(frame_bgr)

                # Aktualizacja FPS
                now = time.time()
                dt = now - self.last_time
                self.last_time = now
                if dt > 0:
                    self.fps_tracker.append(1.0 / dt)
                avg_fps = np.mean(self.fps_tracker) if self.fps_tracker else 0.0

                # Rysowanie na klatce
                annotated_frame = self.draw_annotations(frame_bgr, ai_info, avg_fps, metadata)

                # Wyświetlanie
                if self.viewer_type == "cv2":
                    should_exit = self.display_cv2(annotated_frame, tof_frame)
                    if should_exit:
                        break
                else:
                    self.display_plt(annotated_frame, tof_frame, metadata)

                if self.frame_saver:
                    self.frame_saver.save(frame, tof_frame, metadata)
        finally:
            if self.frame_saver:
                self.frame_saver.close()
            if self.viewer_type == "cv2":
                cv2.destroyAllWindows()
            if self.client:
                self.client.shutdown()

    def draw_annotations(
        self,
        frame_bgr: np.ndarray,
        ai_info: Dict[str, Any],
        fps: float,
        metadata: Optional[StreamerMetadata],
    ) -> np.ndarray:
        """Rysuje ramkę detekcji YOLO, etykietę gestu, miniaturę oraz parametry telemetryczne."""
        vis = frame_bgr.copy()
        h, w = vis.shape[:2]

        mode_text = "[MODE: OPEN/CLOSED]" if ai_info["mode"] == "binary" else "[MODE: ALL 8 GESTURES]"
        mode_color = (0, 220, 220) if ai_info["mode"] == "binary" else (255, 120, 255)

        # 1. Pasek informacyjny u góry
        cv2.rectangle(vis, (0, 0), (w, 36), (25, 25, 25), -1)
        cv2.putText(vis, mode_text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color, 2, cv2.LINE_AA)
        fps_text = f"FPS: {fps:4.1f} | {ai_info['latency_ms']:3.0f}ms"
        cv2.putText(vis, fps_text, (w - 145, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)

        # 2. Ramka detekcji YOLO i etykieta gestu
        if ai_info["detected"] and ai_info["box"] is not None:
            x1, y1, x2, y2 = ai_info["box"]
            gesture = ai_info["gesture"]
            color = GESTURE_COLORS.get(gesture, (0, 255, 0))

            # Ramka YOLO wokół dłoni
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)

            # Etykieta gestu z tłem
            label_text = f"{gesture.upper()}: {ai_info['gesture_conf']*100:.1f}% (YOLO: {ai_info['box_conf']*100:.0f}%)"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
            bg_y1 = max(0, y1 - th - 10)
            cv2.rectangle(vis, (x1, bg_y1), (x1 + tw + 10, y1), color, -1)
            cv2.putText(vis, label_text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            # Komunikat o braku dłoni
            cv2.putText(vis, "Brak dloni w kadrze", (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 240), 2, cv2.LINE_AA)

        # 3. Miniatura wycinka dłoni (Picture-in-Picture) w prawym dolnym rogu
        if ai_info["crop"] is not None:
            crop = ai_info["crop"]
            crop_h, crop_w = 90, 90
            resized_crop = cv2.resize(crop, (crop_w, crop_h), interpolation=cv2.INTER_AREA)

            # Pozycja w prawym dolnym rogu
            px1, py1 = w - crop_w - 10, h - crop_h - 10
            px2, py2 = px1 + crop_w, py1 + crop_h

            if py1 >= 0 and px1 >= 0:
                color = GESTURE_COLORS.get(ai_info["gesture"], (0, 255, 0))
                # Tło / ramka miniatury
                cv2.rectangle(vis, (px1 - 3, py1 - 3), (px2 + 3, py2 + 3), color, 2)
                vis[py1:py2, px1:px2] = resized_crop
                cv2.putText(vis, "CROP", (px1 + 4, py1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1, cv2.LINE_AA)

        # 4. Telemetria drona (jeśli dostępna)
        if metadata and hasattr(metadata, "state"):
            sx = metadata.state.x / 1000.0
            sy = metadata.state.y / 1000.0
            sz = metadata.state.z / 1000.0
            telemetry = f"Pos: [{sx:+.1f}, {sy:+.1f}, {sz:+.1f}] m"
            cv2.putText(vis, telemetry, (10, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

        return vis

    def display_cv2(self, annotated_bgr: np.ndarray, tof_frame: Optional[np.ndarray]) -> bool:
        """Płynne wyświetlanie okna OpenCV z obsługą skrótów klawiszowych."""
        cv2.imshow("NanoCockpit AI-Deck: YOLO Hand & ResNet Gestures", annotated_bgr)

        # Wyświetlanie TOF jeśli dostępne
        if tof_frame is not None:
            tof_norm = cv2.normalize(tof_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            tof_color = cv2.applyColorMap(tof_norm, cv2.COLORMAP_TURBO)
            cv2.imshow("NanoCockpit: TOF Depth Sensor", cv2.resize(tof_color, (240, 240), interpolation=cv2.INTER_NEAREST))

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):  # 'q' lub ESC
            return True
        elif key == ord('m'):     # 'm' przełącza tryb
            self.detector.toggle_mode()
        return False

    def display_plt(self, annotated_bgr: np.ndarray, tof_frame: Optional[np.ndarray], metadata: Optional[StreamerMetadata]):
        """Wyświetlanie kompatybilne z oryginalnym plt_viewer.py w Matplotlib."""
        import matplotlib.pyplot as plt

        frame_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB) / 255.0

        if self.im is None:
            self.im = self.ax.imshow(frame_rgb)
        else:
            self.im.set_data(frame_rgb)

        if self.tof is None and tof_frame is not None:
            self.tof = self.tof_ax.imshow(tof_frame, cmap="turbo", vmin=0, vmax=255)
        elif self.tof is not None and tof_frame is not None:
            self.tof.set_data(tof_frame)

        mode_title = "OPEN/CLOSED" if self.detector.mode == "binary" else "ALL 8 GESTURES"
        self.ax.set_title(f"AI-deck Camera + YOLO & ResNet [{mode_title}]", fontsize=11, fontweight="bold")
        if tof_frame is not None:
            self.tof_ax.set_title(f"TOF Sensor ({tof_frame.shape[0]}x{tof_frame.shape[1]})", fontsize=11)

        self.fig.canvas.start_event_loop(0.005)

    def on_plt_key(self, event):
        if event.key == "m":
            self.detector.toggle_mode()
        elif event.key in ("q", "escape"):
            self.on_close(None)

    def on_close(self, _event):
        self.client.shutdown()
        sys.exit(0)


def main():
    viewer = GestureDetectorViewer()
    viewer.main()


if __name__ == "__main__":
    main()
