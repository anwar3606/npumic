"""Renders the charts in docs/RESEARCH.md as light and dark SVGs in docs/images/.

Run: uv run --with matplotlib python docs/charts.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).parent / "images"

THEMES = {
    "light": {
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "npu": "#2a78d6",
        "cpu": "#eb6834",
        "gpu": "#1baf7a",
        "old": "#b5b3ab",
        "limit": "#d03b3b",
    },
    "dark": {
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "npu": "#3987e5",
        "cpu": "#d95926",
        "gpu": "#199e70",
        "old": "#5c5b57",
        "limit": "#e66767",
    },
}

# Inference time as a share of the audio it processes (time per step / audio per step), in percent.
SPEED = [
    ("DPDFNet-2", {"NPU": 4.7 / 10, "CPU": 1.0 / 10, "NVIDIA GPU": 1.0 / 10}),
    ("DPDFNet-8", {"NPU": 13.5 / 10, "CPU": 2.9 / 10, "NVIDIA GPU": 2.9 / 10}),
    ("GTCRN", {"NPU": 1.98 / 16}),
    ("UL-UNAS", {"NPU": 4.1 / 16, "CPU": 1.8 / 16}),
    ("DTLN", {"NPU": 0.79 / 8}),
    ("DeepFilterNet3", {"NPU": 4.4 / 10, "CPU": 1.2 / 10}),
    ("VoxLock", {"NPU": 14.4 / 40, "NVIDIA GPU": 4.9 / 24}),
    ("Hush", {"CPU": 0.55 / 10}),
]
DEVICES = [("NPU", "npu"), ("CPU", "cpu"), ("NVIDIA GPU", "gpu")]

# NPU output compared with CPU output: how far below the CPU output the difference is, in dB.
FP16 = [("DPDFNet-2", 16.4, 38.0), ("UL-UNAS", 22.0, 43.0)]

# Late audio cycles in 4 minutes at a 5.3 ms PipeWire cycle, by VoxLock frames per NPU call.
QUEUE = [("Both models on the NPU", 811, 53), ("Noise removal on the CPU", 949, 5)]


def setup(theme: dict, width: float, height: float):
    fig, ax = plt.subplots(figsize=(width, height), dpi=100)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(theme["axis"])
    ax.tick_params(colors=theme["ink2"], length=0, labelsize=11)
    ax.xaxis.grid(True, color=theme["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def title(fig, theme: dict, text: str, sub: str) -> None:
    height = fig.get_figheight()
    fig.text(0.01, 1 - 0.15 / height, text, color=theme["ink"], fontsize=15, fontweight="bold", va="top")
    fig.text(0.01, 1 - 0.47 / height, sub, color=theme["ink2"], fontsize=11, va="top")


def legend(fig, theme: dict, handles, labels) -> None:
    height = fig.get_figheight()
    leg = fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.003, 1 - 0.75 / height),
        ncol=len(labels),
        frameon=False,
        fontsize=10.5,
        handlelength=1.0,
        columnspacing=1.6,
    )
    for text in leg.get_texts():
        text.set_color(theme["ink2"])


def speed_chart(theme: dict, path: Path) -> None:
    fig, ax = setup(theme, 9.2, 6.6)
    bar = 0.26
    names = [name for name, _ in SPEED]
    handles = {}
    for row, (_, values) in enumerate(SPEED):
        for slot, (device, key) in enumerate(DEVICES):
            if device not in values:
                continue
            y = row + (slot - 1) * (bar + 0.03)
            pct = values[device] * 100
            handles[device] = ax.barh(y, pct, height=bar, color=theme[key], linewidth=0)
            if pct > 100:
                ax.text(pct + 2, y, f"{pct:.0f}%", va="center", fontsize=10.5, color=theme["ink"])
    ax.axvline(100, color=theme["limit"], linewidth=1.5)
    ax.text(101.5, -0.55, "real-time limit", color=theme["ink2"], fontsize=10.5, va="center")
    ax.set_yticks(range(len(SPEED)), names)
    ax.set_ylim(len(SPEED) - 0.5, -0.8)
    ax.set_xlim(0, 150)
    ax.set_xticks(range(0, 151, 25), [f"{t}%" for t in range(0, 151, 25)])
    legend(fig, theme, [handles[d] for d, _ in DEVICES], [d for d, _ in DEVICES])
    title(fig, theme, "Time spent per second of audio", "Lower is better. Above 100% a model can't keep up with live audio.")
    fig.subplots_adjust(left=0.16, right=0.97, top=1 - 1.2 / 6.6, bottom=0.06)
    fig.savefig(path, transparent=True)
    plt.close(fig)


def paired_chart(theme: dict, path: Path, rows, labels, unit: str, head: str, sub: str, xmax: float) -> None:
    fig, ax = setup(theme, 9.2, 3.6)
    bar = 0.34
    handles = []
    for slot, key in enumerate(("old", "npu")):
        ys = [row + (slot - 0.5) * (bar + 0.04) for row in range(len(rows))]
        values = [r[1 + slot] for r in rows]
        handles.append(ax.barh(ys, values, height=bar, color=theme[key], linewidth=0))
        for y, value in zip(ys, values):
            ax.text(value + xmax * 0.012, y, f"{value:g}{unit}", va="center", fontsize=10.5, color=theme["ink"])
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, xmax)
    legend(fig, theme, handles, labels)
    title(fig, theme, head, sub)
    fig.subplots_adjust(left=0.24, right=0.97, top=1 - 1.2 / 3.6, bottom=0.1)
    fig.savefig(path, transparent=True)
    plt.close(fig)


def main() -> None:
    plt.rcParams["svg.fonttype"] = "path"
    plt.rcParams["font.family"] = "DejaVu Sans"
    OUT.mkdir(exist_ok=True)
    for mode, theme in THEMES.items():
        speed_chart(theme, OUT / f"speed-{mode}.svg")
        paired_chart(
            theme,
            OUT / f"fp16-{mode}.svg",
            FP16,
            ["Features computed on the NPU (FP16)", "Features computed on the CPU (FP32)"],
            " dB",
            "How close NPU output is to CPU output",
            "Higher is better. Moving the model's input features to the CPU made the NPU match the CPU.",
            55,
        )
        paired_chart(
            theme,
            OUT / f"queue-{mode}.svg",
            QUEUE,
            ["VoxLock: 3 frames per NPU call", "VoxLock: 5 frames per NPU call"],
            "",
            "Late audio cycles in 4 minutes",
            "Lower is better. Fewer, larger VoxLock calls leave room for noise removal on the NPU.",
            1100,
        )


if __name__ == "__main__":
    main()
