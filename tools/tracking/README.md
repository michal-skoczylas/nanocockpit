# OptiTrack Flight Tracker & Telemetry Logger 🚁📍
### Moduł akwizycji danych drona Crazyflie i analizy trajektorii w przestrzeni OptiTrack

Kompleksowy moduł do synchronizacji i rejestracji danych trajektorii 3D z systemu **OptiTrack** z telemetrią kontrolera gestów dłoni (**YOLOv8 + ResNet**). Przygotowany na potrzeby pracy inżynierskiej do precyzyjnej analizy dynamiki lotu, trajektorii, stabilizacji i opóźnień sterowania gestami.

---

## ✋ Gesty dłoni i przypisane akcje

* `openhand` $\rightarrow$ **Start z ziemi** (gdy stoi) / **Ruch w przód** (w locie)
* `shaka` $\rightarrow$ **Ruch w tył**
* `left` $\rightarrow$ **Ruch w lewo**
* `right` $\rightarrow$ **Ruch w prawo**
* `v` $\rightarrow$ **Wznoszenie w górę**
* `dislike` $\rightarrow$ **Opadanie w dół**
* `closehand` $\rightarrow$ **Zatrzymanie w miejscu / Zawis (HOVER / STOP)**
* `ok` $\rightarrow$ **Bezpieczne lądowanie drona (LAND)**

---

## 📁 Zawartość katalogu

* [`optitrack_flight_logger.py`](optitrack_flight_logger.py) – Główny rejestrator danych (OptiTrack UDP + Telemetria UDP $\rightarrow$ CSV/JSON).
* [`plot_flight_data.py`](plot_flight_data.py) – Narzędzie analityczne generujące ciągłą trajektorię 3D, przebiegi osi X, Y, Z w czasie i wykresy gestów.
* [`dummy_optitrack_replay.py`](dummy_optitrack_replay.py) – Symulator/odtwarzacz ramek z pliku `logger_QDrone.txt` przez UDP.
* [`run_offline_test.py`](run_offline_test.py) – Skrypt automatycznego testu offline wykonujący pełną misję lotu i generujący wykresy.

---

## 🚀 1. Uruchamianie w Laboratorium (Fizyczny Dron + OptiTrack)

### Krok 1: Włączenie rejestratora danych
W pierwszym terminalu:
```bash
cd tools/tracking
python3 optitrack_flight_logger.py
```
* **Parametry opcjonalne**:
  * `--up-axis y` – jeśli w laboratorium Motive jest skonfigurowany z osią Y w pionie (domyślnie `z`).
  * `--body-name QDrone` – jeśli chcesz wymusić konkretną nazwę obiektu (domyślnie autodetekcja).
  * `--ot-port 12111` – port OptiTrack UDP (domyślnie 12111).
  * `--telemetry-port 12112` – port telemetrii drona (domyślnie 12112).

### Krok 2: Uruchomienie kontrolera lotu gestami
W drugim terminalu:
```bash
cd src/client/aideck_cpx_streamer/aideck_cpx_streamer
python3 gesture_flight_controller.py
```
1. Wciśnij `'t'` na klawiaturze (lub pokaż `openhand`).
2. Dron zasygnalizuje procedurę wznoszenia: `AKCJA: Wznoszenie na 50 cm...`. Podczas wznoszenia gesty są celowo ignorowane, aby uniknąć fałszywych komend.
3. Gdy dron osiągnie pułap 50 cm, na ekranie pojawi się: `AKCJA: ZAWIS (HOVER) - Gotowy na gesty`.
4. Możesz teraz sterować dronem gestami (ruch w przód: `openhand`, zawis: `closehand`, lądowanie: `ok`).

### Krok 3: Zakończenie sesji
Wciśnij `Ctrl+C` w oknie rejestratora. W katalogu `flight_logs/` powstaną dwa pliki:
- `flight_session_YYYYMMDD_HHMMSS.csv` (pełna trajektoria 3D z komendami i gestami ~100–200 Hz)
- `flight_session_YYYYMMDD_HHMMSS_meta.json` (podsumowanie sesji, histogram komend i gestów)

---

## 🧪 2. Testowanie Offline w Domu (Bez Drona i Bez Kamer)

### Opcja A: Test w 100% automatyczny (1 komenda)
```bash
cd tools/tracking
python3 run_offline_test.py
```
Skrypt przeprowadza symulację misji z kompletem gestów (`takeoff`, `hover`, `forward`, `up`, `land`), zapisuje dane i generuje komplet wykresów PNG.

### Opcja B: Test interaktywny z kamerką laptopa (`--webcam`)
1. **Terminal 1 – Rejestrator:**
   ```bash
   cd tools/tracking
   python3 optitrack_flight_logger.py
   ```
2. **Terminal 2 – Replay OptiTrack z pliku laboratoryjnego:**
   ```bash
   cd tools/tracking
   python3 dummy_optitrack_replay.py --rate 100
   ```
3. **Terminal 3 – Kontroler gestów z kamerką Maca:**
   ```bash
   cd src/client/aideck_cpx_streamer/aideck_cpx_streamer
   python3 gesture_flight_controller.py --dry-run --webcam
   ```

---

## 📊 3. Generowanie Wykresów i Analiza

Dla dowolnej sesji lotu uruchom:
```bash
python3 plot_flight_data.py --csv flight_logs/flight_session_XXXXXXXX_XXXXXX.csv --save --show
```
