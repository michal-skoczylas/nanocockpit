# NanoCockpit - Dokumentacja i Flowcharty

Harmonogram zadań.

## Zadania

### 1. [x] Przeczytaj pozostałe pliki źródłowe do pełnego zrozumienia
Makefile STM32, transport layer CPX, ros_viewer.py, frame_saver.py, Docker README.
Pliki do przeczytania:
- `src/stm32/app/Makefile`
- `src/client/aideck_cpx_streamer/aideck_cpx_streamer/cpx/transport/tcp.py`
- `src/client/aideck_cpx_streamer/aideck_cpx_streamer/cpx/transport/udp.py`
- `src/client/aideck_cpx_streamer/aideck_cpx_streamer/cpx/streamer.py`
- `src/client/aideck_cpx_streamer/aideck_cpx_streamer/ros_viewer.py`
- `src/client/aideck_cpx_streamer/aideck_cpx_streamer/utils/frame_saver.py`
- `src/docker/gapsdk/README.md`
- `src/docker/espidf/README.md`
- `src/client/aideck_cpx_streamer/launch/ros_viewer_launch.xml`
- `src/client/aideck_cpx_streamer/aideck_cpx_streamer/cpx/cpx.py`
- `src/client/aideck_cpx_streamer/aideck_cpx_streamer/cpx/transport/__init__.py`
- `src/gap/lib/uart_protocol.h` (baud rate verification)
- `src/stm32/crazyflie-firmware/src/deck/drivers/src/aideck.c` (baud rate cross-check)

### 2. [x] Stwórz plik docs/NANOCKPIT_ARCHITECTURE.md - sekcja 1: Przegląd architektury
Opis 4 węzłów systemu (STM32, GAP8, ESP32, Host PC).

### 3. [x] Stwórz flowcharty komunikacji w Mermaid
Diagram ogólny architektury + diagram sekwencji dla pipeline PULP-Frontnet.

### 4. [x] Dokumentacja GAP8
Opis lib/, dokumentacja wszystkich 6 przykładów z parametrami.

### 5. [x] Dokumentacja STM32
Opis aplikacji frontnet, parametry compile-time i runtime (CFclient), log group.

### 6. [x] Dokumentacja ESP32/NINA
Opis CPX bridge, konfiguracja Wi-Fi, GPIO, taski FreeRTOS.

### 7. [x] Dokumentacja Client
plt_viewer, ros_viewer, frame_saver, parametry CLI, ROS2 launch.

### 8. [x] Dokumentacja Docker
Obrazy GAP SDK i ESP-IDF, instrukcje użytkowania.

### 9. [x] Sekcja "Jak odpalić cały system od zera"
Kompletna instrukcja end-to-end od venv do lotu.

### 10. [x] Tabela zbiorcza wszystkich parametrów

### 11. [x] Przegląd końcowy i weryfikacja kompletności
