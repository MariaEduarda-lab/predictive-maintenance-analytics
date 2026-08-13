import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets" / "logistic_sigmoid.svg"


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
    x_values = [-7 + (14 * i / 499) for i in range(500)]
    probabilities = [1 / (1 + math.exp(-x)) for x in x_values]

    canvas_width = 980
    canvas_height = 560
    left = 105
    top = 86
    plot_width = 775
    plot_height = 350

    curve_points = scale_points(
        x_values,
        probabilities,
        x_min=-7,
        x_max=7,
        y_min=0,
        y_max=1,
        left=left,
        top=top,
        width=plot_width,
        height=plot_height,
    )
    threshold_x, threshold_y = scale_points(
        [0],
        [0.5],
        x_min=-7,
        x_max=7,
        y_min=0,
        y_max=1,
        left=left,
        top=top,
        width=plot_width,
        height=plot_height,
    )[0]

    horizontal_grid = "\n".join(
        f'<line x1="{left}" y1="{top + plot_height * i / 5}" x2="{left + plot_width}" y2="{top + plot_height * i / 5}" class="grid" />'
        for i in range(6)
    )
    y_labels = "\n".join(
        f'<text x="{left - 16}" y="{top + plot_height - plot_height * i / 5 + 6}" text-anchor="end" class="tick">{i / 5:.1f}</text>'
        for i in range(6)
    )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_width}" height="{canvas_height}" viewBox="0 0 {canvas_width} {canvas_height}" role="img" aria-labelledby="title desc">
  <title id="title">Logistic Sigmoid Function</title>
  <desc id="desc">Sigmoid curve converting input scores into probability outputs between zero and one, with a decision threshold at zero point five.</desc>
  <style>
    text {{ font-family: Inter, Arial, sans-serif; fill: #24292f; }}
    .title {{ font-size: 30px; font-weight: 700; }}
    .axis-label {{ font-size: 18px; font-weight: 600; }}
    .tick {{ font-size: 15px; fill: #57606a; }}
    .note {{ font-size: 17px; fill: #57606a; }}
    .curve {{ fill: none; stroke: #2f80ed; stroke-width: 7; stroke-linecap: round; stroke-linejoin: round; }}
    .axis {{ stroke: #8c959f; stroke-width: 2; }}
    .grid {{ stroke: #eaeef2; stroke-width: 1; }}
    .threshold {{ stroke: #27ae60; stroke-width: 3; stroke-dasharray: 9 9; }}
    .marker {{ fill: #27ae60; }}
    .highlight {{ fill: #1f7a3f; font-size: 18px; font-weight: 700; }}
  </style>
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="{canvas_width / 2}" y="44" text-anchor="middle" class="title">Logistic Sigmoid Function</text>
  {horizontal_grid}
  {y_labels}
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" class="axis" />
  <line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" class="axis" />
  <line x1="{left}" y1="{threshold_y:.2f}" x2="{left + plot_width}" y2="{threshold_y:.2f}" class="threshold" />
  <line x1="{threshold_x:.2f}" y1="{top}" x2="{threshold_x:.2f}" y2="{top + plot_height}" class="threshold" />
  <polyline points="{polyline(curve_points)}" class="curve" />
  <circle cx="{threshold_x:.2f}" cy="{threshold_y:.2f}" r="9" class="marker" />
  <text x="{threshold_x + 20:.2f}" y="{threshold_y - 18:.2f}" class="highlight">Decision threshold</text>
  <text x="{threshold_x + 20:.2f}" y="{threshold_y + 10:.2f}" class="note">p = 0.5 when input = 0</text>
  <text x="{left - 62}" y="{top + plot_height / 2}" text-anchor="middle" class="axis-label" transform="rotate(-90 {left - 62},{top + plot_height / 2})">Probability Output</text>
  <text x="{left + plot_width / 2}" y="{top + plot_height + 76}" text-anchor="middle" class="axis-label">Input Score</text>
  <text x="{left}" y="{top + plot_height + 34}" text-anchor="middle" class="tick">negative</text>
  <text x="{threshold_x:.2f}" y="{top + plot_height + 34}" text-anchor="middle" class="tick">0</text>
  <text x="{left + plot_width}" y="{top + plot_height + 34}" text-anchor="middle" class="tick">positive</text>
</svg>
"""

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    main()
