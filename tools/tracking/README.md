# OptiTrack Flight Tracker & Telemetry Logger 🚁📍

Moduł akwizycji i synchronizacji danych trajektorii 3D drona z systemu **OptiTrack** oraz telemetrii z kontrolera gestów (**YOLOv8 + ResNet**).
Przygotowany na potrzeby pracy inżynierskiej do precyzyjnej analizy dynamiki lotu, trajektorii i opóźnień sterowania gestami.

---

## 📁 Zawartość katalogu

* [`optitrack_flight_logger.py`](file:///Users/michalskoczylas/Documents/studia/thesis/nanocockpit/tools/tracking/optitrack_flight_logger.py) – Główny skrypt rejestrujący pozycję z OptiTracka oraz komendy i gesty z kontrolera lotu do plików CSV i JSON.
* [`plot_flight_data.py`](file:///Users/michalskoczylas/Documents/studia/thesis/nanocockpit/tools/tracking/plot_flight_data.py) – Narzędzie analityczne: generuje wykresy 3D trajektorii, przebiegi osi X, Y, Z w czasie oraz dynamikę detekcji gestów.
* [`dummy_optitrack_replay.py`](file:///Users/michalskoczylas/Documents/studia/thesis/nanocockpit/tools/tracking/dummy_optitrack_replay.py) – Symulator / odtwarzacz ramek z pliku (np. `logger_QDrone.txt`) do testów offline bez obecności w laboratorium.

---

## 🚀 Jak uruchamiać w laboratorium (Testy z dronem)

### Krok 1: Uruchomienie rejestratora danych
W pierwszym terminalu:
```bash
cd /Users/michalskoczylas/Documents/studia/thesis/nanocockpit/tools/tracking
python3 optitrack_flight_logger.py
```
*Domyślnie nasłuchuje OptiTrack na porcie UDP `12111` i telemetrię na porcie `12112`.*
*Jeśli w laboratorium oś pionowa to Y (zamiast Z), dodaj flagę `--up-axis y`.*
*Jeśli chcesz wymusić nazwę rigid body, dodaj `--body-name QDrone` (domyślnie skrypt autowykrywa nazwę).*

### Krok 2: Uruchomienie kontrolera lotu gestami
W drugim terminalu:
```bash
cd /Users/michalskoczylas/Documents/studia/thesis/nanocockpit/src/client/aideck_cpx_streamer/aideck_cpx_streamer
python3 gesture_flight_controller.py
```
*Kontroler automatycznie nadaje pakiety UDP z aktualnym stanem drona, komendą i wykrytym gestem.*

### Krok 3: Zakończenie sesji
Wciśnij `Ctrl+C` w oknie `optitrack_flight_logger.py`. Skrypt natychmiast zamknie i zrzuci pliki:
- `flight_logs/flight_session_YYYYMMDD_HHMMSS.csv` (kompletny zbiór próbek ~100-200 Hz)
- `flight_logs/flight_session_YYYYMMDD_HHMMSS_meta.json` (podsumowanie sesji, statystyki gestów)

---

## 🧪 Jak testować offline w domu (Symulacja / Replay)

1. **Terminal 1 – Symulator OptiTrack**:
   ```bash
   python3 dummy_optitrack_replay.py --rate 100
   ```
2. **Terminal 2 – Rejestrator**:
   ```bash
   python3 optitrack_flight_logger.py
   ```
3. **Terminal 3 – Kontroler w trybie wizji (Dry-Run)**:
   ```bash
   python3 gesture_flight_controller.py --dry-run
   ```

---

## 📊 Generowanie wykresów do pracy inżynierskiej

Aby wygenerować wykresy dla wybranej sesji:
```bash
python3 plot_flight_data.py --csv flight_logs/flight_session_XXXXXXXX_XXXXXX.csv --save --show
```
Skrypt automatycznie tworzy:
1. `*_traj3d.png` – Trajektoria 3D w przestrzeni laboratoryjnej z kolorami oznaczającymi wykonywane komendy.
2. `*_timeseries.png` – Przebiegi czasowe X(t), Y(t), Z(t) z nałożonymi pasmami czasowymi poszczególnych komend ruchu.
3. `*_gestures.png` – Rozkład wykrytych gestów w czasie i ich pewność klasyfikacji.

---

## 📋 Struktura danych w pliku CSV

| Kolumna | Opis |
|---|---|
| `timestamp` | Czas uniksowy rejestracji próbki [s] |
| `time_rel` | Czas od startu nagrywania [s] (np. `0.0000`, `0.0050`...) |
| `ot_visible` | Flaga widoczności w OptiTrack (1 = widoczny, 0 = przesłonięty) |
| `pos_x`, `pos_y`, `pos_z` | Pozycja drona w układzie zorientowanym do robotyki (Z to wysokość) [m] |
| `raw_x`, `raw_y`, `raw_z` | Surowe współrzędne z serwera OptiTrack [m] |
| `roll`, `pitch`, `yaw` | Kąty orientacji drona [rad / deg] |
| `flight_state` | Stan drona (`GROUND`, `TAKING_OFF`, `FLYING`, `LANDING`, `LANDED`) |
| `command` | Aktualna komenda (`hover`, `forward`, `back`, `left`, `right`, `up`, `down`, `stop`, `land`, `takeoff`) |
| `gesture` | Aktualnie wykryty gest (`openhand`, `closehand`, `v`, `shaka`, `left`, `right`, `dislike`, `ok`, `none`) |
| `gesture_conf` | Pewność predykcji ResNet (0.00 - 1.00) |
| `window_status` | Status okna przesuwnego (np. `OPENHAND: 8/10`) |
| `cmd_age_s` | Czas jaki upłynął od wydania obecnej komendy [s] |
| `gesture_age_s` | Czas jaki upłynął od ostatniej klatki detekcji gestu [s] |
| `telemetry_connected` | Status połączenia z kontrolerem gestów (1 / 0) |

