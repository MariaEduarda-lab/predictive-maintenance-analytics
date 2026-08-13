import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets" / "bias_variance_tradeoff.svg"


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
    complexity = [0.05 + (0.95 * i / 399) for i in range(400)]
    bias = [0.9 * math.exp(-3.1 * x) + 0.08 for x in complexity]
    variance = [0.08 + 0.9 * x**2.6 for x in complexity]
    irreducible_error = 0.13
    total_error = [b + v + irreducible_error for b, v in zip(bias, variance)]

    optimal_idx = min(range(len(total_error)), key=total_error.__getitem__)
    optimal_complexity = complexity[optimal_idx]
    optimal_error = total_error[optimal_idx]

    canvas_width = 980
    canvas_height = 580
    left = 95
    top = 88
    plot_width = 790
    plot_height = 355
    x_min = min(complexity)
    x_max = max(complexity)
    y_min = 0.0
    y_max = max(total_error) + 0.18

    bias_points = scale_points(
        complexity,
        bias,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        left=left,
        top=top,
        width=plot_width,
        height=plot_height,
    )
    variance_points = scale_points(
        complexity,
        variance,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        left=left,
        top=top,
        width=plot_width,
        height=plot_height,
    )
    total_points = scale_points(
        complexity,
        total_error,
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
        [optimal_error],
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
    optimal_line = (
        f'<line x1="{optimal_x:.2f}" y1="{top}" x2="{optimal_x:.2f}" '
        f'y2="{top + plot_height}" class="optimal-line" />'
    )
    arrow_path = (
        f"M {optimal_x + 130:.2f} {optimal_y - 90:.2f} "
        f"C {optimal_x + 92:.2f} {optimal_y - 82:.2f}, "
        f"{optimal_x + 48:.2f} {optimal_y - 47:.2f}, "
        f"{optimal_x + 10:.2f} {optimal_y - 12:.2f}"
    )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_width}" height="{canvas_height}" viewBox="0 0 {canvas_width} {canvas_height}" role="img" aria-labelledby="title desc">
  <title id="title">Bias-Variance Tradeoff</title>
  <desc id="desc">Line chart showing bias decreasing, variance increasing, and total error forming a U shape as model complexity grows.</desc>
  <style>
    text {{ font-family: Inter, Arial, sans-serif; fill: #24292f; }}
    .title {{ font-size: 30px; font-weight: 700; }}
    .axis-label {{ font-size: 18px; font-weight: 600; }}
    .note {{ font-size: 17px; fill: #57606a; }}
    .legend {{ font-size: 17px; font-weight: 600; }}
    .grid {{ stroke: #eaeef2; stroke-width: 1; }}
    .axis {{ stroke: #8c959f; stroke-width: 2; }}
    .bias {{ fill: none; stroke: #2f80ed; stroke-width: 5; stroke-linecap: round; stroke-linejoin: round; }}
    .variance {{ fill: none; stroke: #f2994a; stroke-width: 5; stroke-linecap: round; stroke-linejoin: round; }}
    .total {{ fill: none; stroke: #24292f; stroke-width: 6; stroke-linecap: round; stroke-linejoin: round; }}
    .optimal-line {{ stroke: #27ae60; stroke-width: 3; stroke-dasharray: 9 9; }}
    .sweet {{ fill: #1f7a3f; font-size: 19px; font-weight: 700; }}
  </style>
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="{canvas_width / 2}" y="44" text-anchor="middle" class="title">Bias-Variance Tradeoff</text>
  {grid_lines}
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" class="axis" />
  <line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" class="axis" />
  {optimal_line}
  <polyline points="{polyline(total_points)}" class="total" />
  <polyline points="{polyline(bias_points)}" class="bias" />
  <polyline points="{polyline(variance_points)}" class="variance" />
  <circle cx="{optimal_x:.2f}" cy="{optimal_y:.2f}" r="8" fill="#27ae60" />
  <path d="{arrow_path}" fill="none" stroke="#27ae60" stroke-width="3" marker-end="url(#arrow)" />
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L9,3 z" fill="#27ae60" />
    </marker>
  </defs>
  <text x="{optimal_x + 140:.2f}" y="{optimal_y - 94:.2f}" class="sweet">Sweet spot</text>
  <text x="{left - 52}" y="{top + plot_height / 2}" text-anchor="middle" class="axis-label" transform="rotate(-90 {left - 52},{top + plot_height / 2})">Error</text>
  <text x="{left + plot_width / 2}" y="{top + plot_height + 82}" text-anchor="middle" class="axis-label">Model Complexity</text>
  <text x="{left + 85}" y="{top + plot_height + 32}" text-anchor="middle" class="note">Underfitting</text>
  <text x="{left + 85}" y="{top + plot_height + 55}" text-anchor="middle" class="note">simple models</text>
  <text x="{left + plot_width - 82}" y="{top + plot_height + 32}" text-anchor="middle" class="note">Overfitting</text>
  <text x="{left + plot_width - 82}" y="{top + plot_height + 55}" text-anchor="middle" class="note">complex models</text>
  <g transform="translate(280 68)">
    <line x1="0" y1="0" x2="44" y2="0" class="bias" />
    <text x="56" y="6" class="legend">Bias</text>
    <line x1="160" y1="0" x2="204" y2="0" class="variance" />
    <text x="216" y="6" class="legend">Variance</text>
    <line x1="375" y1="0" x2="419" y2="0" class="total" />
    <text x="431" y="6" class="legend">Total error</text>
  </g>
</svg>
"""

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    main()
