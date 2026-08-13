import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets" / "model_complexity_performance.svg"


def polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def scale_points(
    x_values: list[float],
    y_values: list[float],
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    left: float,
    top: float,
    width: float,
    height: float,
) -> list[tuple[float, float]]:
    return [
        (
            left + ((x - x_min) / (x_max - x_min)) * width,
            top + height - ((y - y_min) / (y_max - y_min)) * height,
        )
        for x, y in zip(x_values, y_values)
    ]


def main() -> None:
    complexity = [i / 399 for i in range(400)]
    performance = [
        0.22 + 0.68 * math.exp(-((x - 0.5) ** 2) / 0.055) + 0.1 * x
        for x in complexity
    ]

    optimal_idx = max(range(len(performance)), key=performance.__getitem__)
    optimal_complexity = complexity[optimal_idx]
    optimal_performance = performance[optimal_idx]

    canvas_width = 980
    canvas_height = 560
    left = 95
    top = 88
    plot_width = 790
    plot_height = 340
    x_min = min(complexity)
    x_max = max(complexity)
    y_min = 0.0
    y_max = 1.08

    curve_points = scale_points(
        complexity,
        performance,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        left=left,
        top=top,
        width=plot_width,
        height=plot_height,
    )
    optimal_x, optimal_y = scale_points(
        [optimal_complexity],
        [optimal_performance],
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        left=left,
        top=top,
        width=plot_width,
        height=plot_height,
    )[0]

    grid_lines = "\n".join(
        f'<line x1="{left}" y1="{y}" x2="{left + plot_width}" y2="{y}" class="grid" />'
        for y in [top + (plot_height * i / 5) for i in range(6)]
    )
    band_width = plot_width / 3

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_width}" height="{canvas_height}" viewBox="0 0 {canvas_width} {canvas_height}" role="img" aria-labelledby="title desc">
  <title id="title">Model Complexity and Performance</title>
  <desc id="desc">Line chart showing performance rising from underfitting, peaking at an optimal fit, and falling as the model overfits.</desc>
  <style>
    text {{ font-family: Inter, Arial, sans-serif; fill: #24292f; }}
    .title {{ font-size: 30px; font-weight: 700; }}
    .axis-label {{ font-size: 18px; font-weight: 600; }}
    .band-label {{ font-size: 18px; font-weight: 700; }}
    .note {{ font-size: 16px; fill: #57606a; }}
    .grid {{ stroke: #eaeef2; stroke-width: 1; }}
    .axis {{ stroke: #8c959f; stroke-width: 2; }}
    .divider {{ stroke: #d0d7de; stroke-width: 2; stroke-dasharray: 8 8; }}
    .curve {{ fill: none; stroke: #2f80ed; stroke-width: 6; stroke-linecap: round; stroke-linejoin: round; }}
    .optimal-line {{ stroke: #27ae60; stroke-width: 3; stroke-dasharray: 9 9; }}
    .sweet {{ fill: #1f7a3f; font-size: 19px; font-weight: 700; }}
  </style>
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="{canvas_width / 2}" y="44" text-anchor="middle" class="title">Model Complexity and Performance</text>
  <rect x="{left}" y="{top}" width="{band_width}" height="{plot_height}" fill="#eaf3ff" opacity="0.55" />
  <rect x="{left + band_width}" y="{top}" width="{band_width}" height="{plot_height}" fill="#eafaf0" opacity="0.62" />
  <rect x="{left + 2 * band_width}" y="{top}" width="{band_width}" height="{plot_height}" fill="#fff3e6" opacity="0.65" />
  {grid_lines}
  <line x1="{left + band_width}" y1="{top}" x2="{left + band_width}" y2="{top + plot_height}" class="divider" />
  <line x1="{left + 2 * band_width}" y1="{top}" x2="{left + 2 * band_width}" y2="{top + plot_height}" class="divider" />
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" class="axis" />
  <line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" class="axis" />
  <line x1="{optimal_x:.2f}" y1="{top}" x2="{optimal_x:.2f}" y2="{top + plot_height}" class="optimal-line" />
  <polyline points="{polyline(curve_points)}" class="curve" />
  <circle cx="{optimal_x:.2f}" cy="{optimal_y:.2f}" r="9" fill="#27ae60" />
  <text x="{optimal_x + 22:.2f}" y="{optimal_y - 20:.2f}" class="sweet">Best generalization</text>
  <text x="{left - 52}" y="{top + plot_height / 2}" text-anchor="middle" class="axis-label" transform="rotate(-90 {left - 52},{top + plot_height / 2})">Performance</text>
  <text x="{left + plot_width / 2}" y="{top + plot_height + 82}" text-anchor="middle" class="axis-label">Model Complexity</text>
  <text x="{left + band_width / 2}" y="{top + 32}" text-anchor="middle" class="band-label">Underfitting</text>
  <text x="{left + band_width / 2}" y="{top + plot_height + 32}" text-anchor="middle" class="note">Too simple</text>
  <text x="{left + band_width * 1.5}" y="{top + 32}" text-anchor="middle" class="band-label">Optimal</text>
  <text x="{left + band_width * 1.5}" y="{top + plot_height + 32}" text-anchor="middle" class="note">Good fit</text>
  <text x="{left + band_width * 2.5}" y="{top + 32}" text-anchor="middle" class="band-label">Overfitting</text>
  <text x="{left + band_width * 2.5}" y="{top + plot_height + 32}" text-anchor="middle" class="note">Too complex</text>
</svg>
"""

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    main()
