# OptiTrack Flight Tracker & Telemetry Logger 🚁📍
### Moduł akwizycji danych drona Crazyflie i analizy trajektorii w przestrzeni OptiTrack

Kompleksowy moduł do synchronizacji i rejestracji danych trajektorii 3D z systemu **OptiTrack** z telemetrią kontrolera gestów dłoni (**YOLOv8 + ResNet**). Przygotowany na potrzeby pracy inżynierskiej do precyzyjnej analizy dynamiki lotu, trajektorii, stabilizacji i opóźnień sterowania gestami.

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
Kontroler automatycznie nawiąże bezkontaktowe nadawanie telemetrii UDP. Jeśli rejestrator nie jest włączony, dron leci normalnie.

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
Skrypt przeprowadza symulację 17-sekundowej misji z kompletem gestów (`takeoff`, `hover`, `forward`, `up`, `land`), zapisuje dane i generuje komplet wykresów PNG.

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
Pokazuj gesty do wbudowanej kamery i obserwuj w Terminalu 1, jak stan drona i gesty synchronizują się z trajektorią OptiTracka.

---

## 📊 3. Generowanie Wykresów i Analiza

Dla dowolnej sesji lotu uruchom:
```bash
python3 plot_flight_data.py --csv flight_logs/flight_session_XXXXXXXX_XXXXXX.csv --save --show
```

### Otrzymasz:
1. `*_traj3d.png` – Ciągła linia trajektorii 3D ze zmieniającymi się kolorami komend, rzutem na podłogę ($Z=0$) oraz punktami startu i lądowania.
2. `*_timeseries.png` – Przebiegi współrzędnych $X(t)$, $Y(t)$, $Z(t)$ z zaznaczonymi pasmami czasowymi komend w tle.
3. `*_gestures.png` – Dynamika klasyfikacji gestów w czasie z nałożoną granicą pewności progu aktywacji ($65\%$).

---

## 📋 4. Struktura Kolumn w Pliku CSV

| Kolumna | Typ | Opis |
|---|---|---|
| `timestamp` | `float` | Czas uniksowy rejestracji próbki $[s]$ |
| `time_rel` | `float` | Czas od startu nagrywania sesji $[s]$ |
| `ot_visible` | `int` | $1$ = marker widoczny w OptiTrack, $0$ = przesłonięty |
| `pos_x`, `pos_y`, `pos_z` | `float` | Zorientowane współrzędne pozycji drona (oś $Z$ zawsze pion w górę) $[m]$ |
| `raw_x`, `raw_y`, `raw_z` | `float` | Surowe współrzędne bezpośrednio z pakietu serwera OptiTrack $[m]$ |
| `roll`, `pitch`, `yaw` | `float` | Orientacja kątowa drona $[rad / deg]$ |
| `flight_state` | `string` | Stan operacyjny (`GROUND`, `TAKING_OFF`, `FLYING`, `LANDING`, `LANDED`) |
| `command` | `string` | Wykonywana akcja (`idle`, `takeoff`, `forward`, `back`, `left`, `right`, `up`, `down`, `stop`, `hover`, `land`) |
| `gesture` | `string` | Rozpoznany gest dłoni (`openhand`, `closehand`, `v`, `shaka`, `left`, `right`, `dislike`, `ok`, `none`) |
| `gesture_conf` | `float` | Pewność modelu ResNet ($0.00 - 1.00$) |
| `window_status` | `string` | Status bufora okna przesuwnego (np. `OPENHAND: 8/10`) |
| `cmd_age_s` | `float` | Czas w sekundach od momentu wydania aktualnej komendy |
| `gesture_age_s` | `float` | Czas w sekundach od ostatniej klatki detekcji gestu |
| `telemetry_connected` | `int` | Flaga połączenia z kontrolerem drona ($1$ = aktywny, $0$ = brak pakietów $>2.5s$) |
