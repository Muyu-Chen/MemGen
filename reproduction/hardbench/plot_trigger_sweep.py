"""Render the single-candidate position/utility curve as a static PNG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH = 1600
HEIGHT = 1180
MARGIN_LEFT = 150
MARGIN_RIGHT = 70
PANEL_TOPS = (190, 500, 810)
PANEL_HEIGHT = 225
COLORS = {
    "background": "#ffffff",
    "foreground": "#17202a",
    "muted": "#68737d",
    "grid": "#dce1e5",
    "line": "#4c6ef5",
    "improved": "#208a57",
    "worsened": "#c43d4b",
    "equal": "#77838f",
    "baseline": "#d77b13",
    "human": "#7b3fc6",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def font(size: int, bold: bool = False):
    names = [
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def tick_values(maximum: float, count: int = 5) -> list[float]:
    if maximum <= 1:
        return [0.0, 0.25, 0.5, 0.75, 1.0]
    step = maximum / count
    return [step * index for index in range(count + 1)]


def main() -> None:
    args = parse_args()
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    human_ordinals = {
        "gsm8k_test_0710": 7,
        "gsm8k_test_0611": 1,
        "gsm8k_test_0754": 1,
    }
    sample_order = [
        "gsm8k_test_0710",
        "gsm8k_test_0611",
        "gsm8k_test_0754",
    ]
    samples = {sample["sample_id"]: sample for sample in analysis["samples"]}

    image = Image.new("RGB", (WIDTH, HEIGHT), COLORS["background"])
    draw = ImageDraw.Draw(image)
    draw.text(
        (MARGIN_LEFT, 45),
        "Single-invocation position utility",
        font=font(36, bold=True),
        fill=COLORS["foreground"],
    )
    draw.text(
        (MARGIN_LEFT, 95),
        "Absolute numeric error by candidate ordinal; lower is better",
        font=font(22),
        fill=COLORS["muted"],
    )

    plot_right = WIDTH - MARGIN_RIGHT
    plot_width = plot_right - MARGIN_LEFT
    for panel_index, sample_id in enumerate(sample_order):
        sample = samples[sample_id]
        variants = sample["variants"]
        top = PANEL_TOPS[panel_index]
        bottom = top + PANEL_HEIGHT
        errors = [float(variant["absolute_error"]) for variant in variants]
        baseline_error = float(sample["baseline_absolute_error"])
        raw_max = max(errors + [baseline_error, 1.0])
        y_max = raw_max * 1.08
        x_max = max(variant["candidate_ordinal"] for variant in variants)

        def x_pos(ordinal: int) -> float:
            if x_max == 1:
                return MARGIN_LEFT + plot_width / 2
            return MARGIN_LEFT + (ordinal - 1) / (x_max - 1) * plot_width

        def y_pos(value: float) -> float:
            return bottom - value / y_max * PANEL_HEIGHT

        draw.text(
            (MARGIN_LEFT, top - 40),
            f"{sample_id}   N={sample['baseline_prediction']}   gold={sample['gold']}",
            font=font(21, bold=True),
            fill=COLORS["foreground"],
        )
        for value in tick_values(y_max):
            y = y_pos(value)
            draw.line((MARGIN_LEFT, y, plot_right, y), fill=COLORS["grid"], width=1)
            label_value = value / 1_000_000 if y_max >= 1_000_000 else value
            suffix = "M" if y_max >= 1_000_000 else ""
            label = f"{label_value:.1f}{suffix}"
            bounds = draw.textbbox((0, 0), label, font=font(17))
            draw.text(
                (MARGIN_LEFT - 15 - (bounds[2] - bounds[0]), y - 10),
                label,
                font=font(17),
                fill=COLORS["muted"],
            )

        draw.rectangle(
            (MARGIN_LEFT, top, plot_right, bottom),
            outline=COLORS["grid"],
            width=2,
        )
        baseline_y = y_pos(baseline_error)
        for start in range(int(MARGIN_LEFT), int(plot_right), 18):
            draw.line(
                (start, baseline_y, min(start + 10, plot_right), baseline_y),
                fill=COLORS["baseline"],
                width=3,
            )
        human_x = x_pos(human_ordinals[sample_id])
        draw.line((human_x, top, human_x, bottom), fill=COLORS["human"], width=3)

        points = [
            (x_pos(variant["candidate_ordinal"]), y_pos(float(variant["absolute_error"])))
            for variant in variants
        ]
        if len(points) > 1:
            draw.line(points, fill=COLORS["line"], width=3)
        for variant, (x, y) in zip(variants, points):
            relation = variant["distance_relation_vs_baseline"]
            color = COLORS[relation]
            radius = 8
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

        tick_ordinals = sorted({1, x_max, human_ordinals[sample_id]} | {
            1 + round(index * (x_max - 1) / min(6, max(1, x_max - 1)))
            for index in range(min(6, max(1, x_max - 1)) + 1)
        })
        for ordinal in tick_ordinals:
            x = x_pos(ordinal)
            draw.line((x, bottom, x, bottom + 7), fill=COLORS["muted"], width=2)
            label = str(ordinal)
            bounds = draw.textbbox((0, 0), label, font=font(16))
            draw.text(
                (x - (bounds[2] - bounds[0]) / 2, bottom + 10),
                label,
                font=font(16),
                fill=COLORS["muted"],
            )
        human_label_x = human_x + 14 if human_x < (MARGIN_LEFT + plot_right) / 2 else plot_right - 170
        draw.text(
            (human_label_x, top + 10),
            f"human pick #{human_ordinals[sample_id]}",
            font=font(17),
            fill=COLORS["human"],
        )

    legend_y = 1100
    legend_items = [
        ("improved vs N", COLORS["improved"]),
        ("worsened vs N", COLORS["worsened"]),
        ("equal to N", COLORS["equal"]),
        ("N baseline", COLORS["baseline"]),
        ("human pick", COLORS["human"]),
    ]
    cursor = MARGIN_LEFT
    for label, color in legend_items:
        draw.ellipse((cursor, legend_y, cursor + 14, legend_y + 14), fill=color)
        draw.text((cursor + 22, legend_y - 4), label, font=font(17), fill=COLORS["foreground"])
        cursor += 255

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output, format="PNG", optimize=True)
    print(f"plot={args.output.resolve()}")


if __name__ == "__main__":
    main()
