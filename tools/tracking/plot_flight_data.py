#!/usr/bin/env python3
"""
plot_flight_data.py

Narzędzie analityczne i generator wykresów do pracy inżynierskiej:
1. Wizualizacja trajektorii 3D drona w przestrzeni z podziałem na komendy ruchu i gesty.
2. Przebiegi czasowe X(t), Y(t), Z(t) z zaznaczonymi przedziałami czasowymi poszczególnych komend.
3. Analiza wykrywania gestów (klasyfikacja, pewność modelu w czasie).
4. Wyznaczanie kluczowych metryk lotu (dystans, prędkość, stabilność wysokości).

Użycie:
    python3 plot_flight_data.py --csv flight_logs/flight_session_XXXXXXXX_XXXXXX.csv [--save] [--show]
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# Spójna paleta kolorów dla komend ruchu i gestów
CMD_COLORS = {
    "idle": "#7f7f7f",       # Szary
    "takeoff": "#17becf",    # Błękitny
    "hover": "#2ca02c",      # Zielony
    "forward": "#1f77b4",    # Niebieski
    "back": "#ff7f0e",       # Pomarańczowy
    "left": "#9467bd",       # Fioletowy
    "right": "#8c564b",      # Brązowy
    "up": "#e377c2",         # Różowy
    "down": "#bcbd22",       # Oliwkowy
    "stop": "#d62728",       # Czerwony
    "land": "#d62728",       # Czerwony
    "failsafe_land": "#800000",
}


def plot_trajectory_3d(df: pd.DataFrame, out_path: Path = None, show: bool = True):
    """Wykres 3D trajektorii lotu - ciągła linia ze zmianą kolorów wg komend."""
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 1. Rzut trajektorii na płaszczyznę podłogi (cień pomocniczy do percepcji 3D)
    z_floor = max(0.0, df["pos_z"].min() - 0.05)
    ax.plot(
        df["pos_x"],
        df["pos_y"],
        np.full_like(df["pos_z"], z_floor),
        color="#bbbbbb",
        linestyle=":",
        linewidth=1.2,
        alpha=0.6,
        label="Rzut na podłogę",
    )

    # 2. Rysujemy ciągłą linię trajektorii segmentami odpowiadającymi komendom ruchu
    change_indices = df.index[df["command"] != df["command"].shift()].tolist()
    if 0 not in change_indices:
        change_indices = [0] + change_indices
    change_indices.append(len(df))

    labeled_cmds = set()
    for i in range(len(change_indices) - 1):
        idx_start = change_indices[i]
        idx_end = change_indices[i + 1]
        # Bierzemy punkt styku z następnym segmentem (idx_end + 1), aby linia była idealnie ciągła
        slice_end = min(len(df), idx_end + 1)
        seg = df.iloc[idx_start:slice_end]

        cmd = str(df.loc[idx_start, "command"])
        color = CMD_COLORS.get(cmd, "#333333")

        label = f"Komenda: {cmd}" if cmd not in labeled_cmds else None
        if label:
            labeled_cmds.add(cmd)

        # Ciągła linia trajektorii
        ax.plot(
            seg["pos_x"],
            seg["pos_y"],
            seg["pos_z"],
            color=color,
            linewidth=2.8,
            alpha=0.9,
            solid_capstyle="round",
            label=label,
        )

    # 3. Zaznaczenie punktu startu i końca
    ax.scatter(
        df["pos_x"].iloc[0], df["pos_y"].iloc[0], df["pos_z"].iloc[0],
        color="#00ff00", s=140, marker="o", edgecolor="black", linewidth=1.5, label="Start lotu", zorder=10
    )
    ax.scatter(
        df["pos_x"].iloc[-1], df["pos_y"].iloc[-1], df["pos_z"].iloc[-1],
        color="#ff0000", s=140, marker="X", edgecolor="black", linewidth=1.5, label="Koniec lotu", zorder=10
    )

    ax.set_title("Trajektoria 3D Drona w Przestrzeni OptiTrack (Ciągła Trasa)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("X [m]", fontsize=11, labelpad=8)
    ax.set_ylabel("Y [m]", fontsize=11, labelpad=8)
    ax.set_zlabel("Wysokość Z [m]", fontsize=11, labelpad=8)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0), fontsize=9)
    plt.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"📊 Zapisano wykres 3D: {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_time_series(df: pd.DataFrame, out_path: Path = None, show: bool = True):
    """Przebiegi współrzędnych X, Y, Z w czasie z naniesionymi przedziałami komend."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    t = df["time_rel"]

    axes[0].plot(t, df["pos_x"], color="#1f77b4", linewidth=1.5, label="Pozycja X")
    axes[0].set_ylabel("X [m]", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(t, df["pos_y"], color="#2ca02c", linewidth=1.5, label="Pozycja Y")
    axes[1].set_ylabel("Y [m]", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    axes[2].plot(t, df["pos_z"], color="#d62728", linewidth=1.5, label="Wysokość Z")
    axes[2].set_ylabel("Z [m]", fontsize=11)
    axes[2].set_xlabel("Czas względny [s]", fontsize=11)
    axes[2].grid(True, linestyle="--", alpha=0.5)

    # Zaznaczenie przedziałów czasowych komend w tle
    change_indices = df.index[df["command"] != df["command"].shift()].tolist() + [len(df) - 1]
    for i in range(len(change_indices) - 1):
        idx_start = change_indices[i]
        idx_end = change_indices[i + 1]
        t_start = df.loc[idx_start, "time_rel"]
        t_end = df.loc[idx_end, "time_rel"]
        cmd = df.loc[idx_start, "command"]
        color = CMD_COLORS.get(cmd, "#ffffff")

        for ax in axes:
            ax.axvspan(t_start, t_end, color=color, alpha=0.15)

    axes[0].set_title("Przebieg Współrzędnych Lotu w Czasie (Zaznaczone Aktywne Komendy)", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"📊 Zapisano wykres czasowy: {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_gestures_and_confidence(df: pd.DataFrame, out_path: Path = None, show: bool = True):
    """Wykres detekcji gestów i pewności klasyfikacji w czasie."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    t = df["time_rel"]

    # Wykres pewności
    ax1.plot(t, df["gesture_conf"] * 100.0, color="purple", linewidth=1.3)
    ax1.axhline(65.0, color="red", linestyle="--", alpha=0.7, label="Próg aktywacji (65%)")
    ax1.set_ylabel("Pewność [%]", fontsize=11)
    ax1.set_title("Dynamika Detekcji Gestów w Czasie", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right")

    # Rozkład wykrytych gestów (scatter)
    valid_gestures = df[df["gesture"] != "none"]
    if not valid_gestures.empty:
        unique_g = sorted(valid_gestures["gesture"].unique())
        g_map = {name: i for i, name in enumerate(unique_g)}
        y_vals = [g_map[g] for g in valid_gestures["gesture"]]
        ax2.scatter(valid_gestures["time_rel"], y_vals, c="navy", s=15, alpha=0.6)
        ax2.set_yticks(list(g_map.values()))
        ax2.set_yticklabels(list(g_map.keys()))
    else:
        ax2.text(0.5, 0.5, "Brak zarejestrowanych gestów w sesji", ha="center", va="center")

    ax2.set_ylabel("Wykryty gest", fontsize=11)
    ax2.set_xlabel("Czas względny [s]", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"📊 Zapisano wykres gestów: {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def print_flight_statistics(df: pd.DataFrame):
    """Oblicza i drukuje kluczowe parametry telemetryczne lotu."""
    duration = df["time_rel"].iloc[-1] - df["time_rel"].iloc[0]
    total_samples = len(df)
    hz = total_samples / (duration + 1e-5)

    # Obliczenie długości trajektorii
    dx = np.diff(df["pos_x"])
    dy = np.diff(df["pos_y"])
    dz = np.diff(df["pos_z"])
    dist_segments = np.sqrt(dx**2 + dy**2 + dz**2)
    total_distance = np.sum(dist_segments)

    # Prędkości
    dt = np.diff(df["time_rel"])
    dt[dt == 0] = 1e-4
    speeds = dist_segments / dt
    max_speed = np.max(speeds)
    mean_speed = np.mean(speeds)

    print("\n" + "=" * 60)
    print("📈 [PODSUMOWANIE PARAMETRÓW LOTU]")
    print(f"   Czas trwania:             {duration:.2f} s")
    print(f"   Liczba próbek:            {total_samples} (śr. {hz:.1f} Hz)")
    print(f"   Długość trajektorii:      {total_distance:.2f} m")
    print(f"   Średnia prędkość:         {mean_speed:.2f} m/s ({mean_speed*100:.0f} cm/s)")
    print(f"   Maksymalna prędkość:      {max_speed:.2f} m/s ({max_speed*100:.0f} cm/s)")
    print(f"   Średnia wysokość Z:       {df['pos_z'].mean():.2f} m (odch. std: {df['pos_z'].std():.3f} m)")
    print(f"   Wykonane komendy:         {df['command'].value_counts().to_dict()}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generowanie wykresów trajektorii lotu drona i detekcji gestów")
    parser.add_argument("--csv", required=True, help="Ścieżka do pliku CSV z sesji lotu")
    parser.add_argument("--save", action="store_true", default=True, help="Zapisz wykresy do plików PNG")
    parser.add_argument("--show", action="store_true", help="Wyświetl interaktywne okna z wykresami")
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        print(f"❌ Plik nie istnieje: {csv_path}")
        sys.exit(1)

    print(f"📂 Wczytywanie danych z: {csv_path}...")
    df = pd.read_csv(csv_path)

    if df.empty or len(df) < 2:
        print("❌ Plik CSV zawiera zbyt mało danych do analizy.")
        sys.exit(1)

    print_flight_statistics(df)

    # Ścieżki wyjściowe dla wykresów
    base_dir = csv_path.parent
    stem = csv_path.stem

    p3d = base_dir / f"{stem}_traj3d.png" if args.save else None
    pts = base_dir / f"{stem}_timeseries.png" if args.save else None
    pgest = base_dir / f"{stem}_gestures.png" if args.save else None

    plot_trajectory_3d(df, out_path=p3d, show=args.show)
    plot_time_series(df, out_path=pts, show=args.show)
    plot_gestures_and_confidence(df, out_path=pgest, show=args.show)


if __name__ == "__main__":
    main()

