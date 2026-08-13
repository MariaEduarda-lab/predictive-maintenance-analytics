from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"


def write_svg(filename: str, body: str, width: int = 980, height: int = 560) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">
  <style>
    text {{ font-family: Inter, Arial, sans-serif; fill: #24292f; }}
    .title {{ font-size: 30px; font-weight: 700; }}
    .label {{ font-size: 18px; font-weight: 700; }}
    .small {{ font-size: 15px; fill: #57606a; }}
    .axis {{ stroke: #8c959f; stroke-width: 2; }}
    .grid {{ stroke: #eaeef2; stroke-width: 1; }}
    .line {{ fill: none; stroke: #2f80ed; stroke-width: 6; stroke-linecap: round; stroke-linejoin: round; }}
    .dash {{ stroke: #27ae60; stroke-width: 3; stroke-dasharray: 9 9; }}
    .box {{ fill: #ffffff; stroke: #d0d7de; stroke-width: 2; rx: 8; }}
    .blue {{ fill: #2f80ed; }}
    .orange {{ fill: #f2994a; }}
    .green {{ fill: #27ae60; }}
    .red {{ fill: #eb5757; }}
    .muted {{ fill: #8c959f; }}
  </style>
  <rect width="100%" height="100%" fill="#ffffff" />
  {body}
</svg>
"""
    (ASSETS / filename).write_text(svg, encoding="utf-8")


def arrow(x1: int, y1: int, x2: int, y2: int, color: str = "#8c959f") -> str:
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{color}" stroke-width="3" marker-end="url(#arrow)" />'
    )


def defs() -> str:
    return """<defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L9,3 z" fill="#8c959f" />
    </marker>
  </defs>"""


def point(x: int, y: int, cls: str, r: int = 8) -> str:
    return f'<circle cx="{x}" cy="{y}" r="{r}" class="{cls}" />'


def main() -> None:
    write_svg(
        "unsupervised_workflow.svg",
        defs()
        + """
  <text x="490" y="48" text-anchor="middle" class="title">Unsupervised Learning Workflow</text>
  <rect x="75" y="118" width="150" height="86" class="box" /><text x="150" y="151" text-anchor="middle" class="label">Prepare Data</text><text x="150" y="178" text-anchor="middle" class="small">clean and scale</text>
  <rect x="285" y="118" width="150" height="86" class="box" /><text x="360" y="151" text-anchor="middle" class="label">Choose Model</text><text x="360" y="178" text-anchor="middle" class="small">clustering, PCA</text>
  <rect x="495" y="118" width="150" height="86" class="box" /><text x="570" y="151" text-anchor="middle" class="label">Fit Model</text><text x="570" y="178" text-anchor="middle" class="small">learn patterns</text>
  <rect x="705" y="118" width="150" height="86" class="box" /><text x="780" y="151" text-anchor="middle" class="label">Interpret</text><text x="780" y="178" text-anchor="middle" class="small">visualize quality</text>
  """
        + arrow(225, 161, 285, 161)
        + arrow(435, 161, 495, 161)
        + arrow(645, 161, 705, 161)
        + """
  <rect x="335" y="310" width="310" height="96" class="box" /><text x="490" y="346" text-anchor="middle" class="label">Use Results</text><text x="490" y="374" text-anchor="middle" class="small">segment, reduce features, detect anomalies</text>
  """
        + arrow(780, 204, 575, 310),
    )

    before_points = [(170, 160), (220, 130), (265, 180), (200, 225), (300, 235), (410, 165), (455, 125), (510, 190), (470, 250), (610, 155), (680, 145), (720, 220), (650, 265), (755, 260)]
    after_blue = [(170, 160), (220, 130), (265, 180), (200, 225), (300, 235)]
    after_orange = [(410, 165), (455, 125), (510, 190), (470, 250)]
    after_green = [(610, 155), (680, 145), (720, 220), (650, 265), (755, 260)]
    write_svg(
        "clustering_before_after.svg",
        """
  <text x="490" y="48" text-anchor="middle" class="title">Before and After Clustering</text>
  <text x="260" y="96" text-anchor="middle" class="label">Before</text>
  <text x="720" y="96" text-anchor="middle" class="label">After</text>
  <rect x="80" y="115" width="360" height="300" fill="#f6f8fa" stroke="#d0d7de" rx="8" />
  <rect x="540" y="115" width="360" height="300" fill="#f6f8fa" stroke="#d0d7de" rx="8" />
  """
        + "".join(point(x, y, "muted") for x, y in before_points)
        + "".join(point(x + 460, y, "blue") for x, y in after_blue)
        + "".join(point(x + 460, y, "orange") for x, y in after_orange)
        + "".join(point(x + 460, y, "green") for x, y in after_green)
        + '<text x="260" y="455" text-anchor="middle" class="small">Unlabeled points</text><text x="720" y="455" text-anchor="middle" class="small">Similar points grouped together</text>',
    )

    write_svg(
        "kmeans_algorithm_steps.svg",
        defs()
        + """
  <text x="490" y="48" text-anchor="middle" class="title">K-Means Algorithm</text>
  <rect x="75" y="118" width="170" height="95" class="box" /><text x="160" y="151" text-anchor="middle" class="label">1. Initialize</text><text x="160" y="178" text-anchor="middle" class="small">place K centroids</text>
  <rect x="300" y="118" width="170" height="95" class="box" /><text x="385" y="151" text-anchor="middle" class="label">2. Assign</text><text x="385" y="178" text-anchor="middle" class="small">nearest centroid</text>
  <rect x="525" y="118" width="170" height="95" class="box" /><text x="610" y="151" text-anchor="middle" class="label">3. Update</text><text x="610" y="178" text-anchor="middle" class="small">move to mean</text>
  <rect x="750" y="118" width="170" height="95" class="box" /><text x="835" y="151" text-anchor="middle" class="label">4. Repeat</text><text x="835" y="178" text-anchor="middle" class="small">until stable</text>
  """
        + arrow(245, 166, 300, 166)
        + arrow(470, 166, 525, 166)
        + arrow(695, 166, 750, 166)
        + """
  <circle cx="275" cy="335" r="70" fill="#eaf3ff" stroke="#2f80ed" stroke-width="2" />
  <circle cx="500" cy="335" r="70" fill="#fff3e6" stroke="#f2994a" stroke-width="2" />
  <circle cx="725" cy="335" r="70" fill="#eafaf0" stroke="#27ae60" stroke-width="2" />
  <text x="275" y="340" text-anchor="middle" class="label">C1</text><text x="500" y="340" text-anchor="middle" class="label">C2</text><text x="725" y="340" text-anchor="middle" class="label">C3</text>
  <text x="490" y="475" text-anchor="middle" class="small">Centroids shift until point assignments stop changing.</text>
  """,
    )

    write_svg(
        "elbow_method.svg",
        """
  <text x="490" y="48" text-anchor="middle" class="title">Elbow Method</text>
  <line x1="115" y1="420" x2="870" y2="420" class="axis" /><line x1="115" y1="95" x2="115" y2="420" class="axis" />
  <path d="M125 120 C210 180, 275 255, 345 315 C450 365, 620 386, 850 398" class="line" />
  <line x1="350" y1="95" x2="350" y2="420" class="dash" /><circle cx="350" cy="318" r="9" class="green" />
  <text x="380" y="295" class="label">Elbow</text><text x="380" y="322" class="small">good tradeoff</text>
  <text x="493" y="485" text-anchor="middle" class="label">K (number of clusters)</text>
  <text x="45" y="258" text-anchor="middle" class="label" transform="rotate(-90 45,258)">Inertia</text>
  <text x="245" y="230" text-anchor="middle" class="small">large improvement</text>
  <text x="665" y="370" text-anchor="middle" class="small">small improvement</text>
  """,
    )

    write_svg(
        "elbow_scenarios.svg",
        """
  <text x="490" y="48" text-anchor="middle" class="title">Interpreting Elbow Plots</text>
  <text x="265" y="98" text-anchor="middle" class="label">Clear Elbow</text><text x="715" y="98" text-anchor="middle" class="label">No Clear Elbow</text>
  <line x1="100" y1="360" x2="430" y2="360" class="axis" /><line x1="100" y1="135" x2="100" y2="360" class="axis" />
  <path d="M105 145 C160 205, 210 275, 265 315 C315 342, 365 350, 425 354" class="line" />
  <circle cx="265" cy="315" r="8" class="green" /><text x="285" y="302" class="small">K=3</text>
  <line x1="550" y1="360" x2="880" y2="360" class="axis" /><line x1="550" y1="135" x2="550" y2="360" class="axis" />
  <path d="M555 150 C625 215, 685 270, 750 314 C790 338, 830 350, 875 355" class="line" />
  <text x="265" y="420" text-anchor="middle" class="small">Use the visible bend as a candidate K.</text>
  <text x="715" y="420" text-anchor="middle" class="small">Use silhouette scores or domain knowledge.</text>
  """,
    )

    write_svg(
        "silhouette_concept.svg",
        """
  <text x="490" y="48" text-anchor="middle" class="title">Silhouette Score Intuition</text>
  <ellipse cx="300" cy="285" rx="150" ry="95" fill="#eaf3ff" stroke="#2f80ed" stroke-width="2" />
  <ellipse cx="690" cy="285" rx="150" ry="95" fill="#fff3e6" stroke="#f2994a" stroke-width="2" />
  <text x="300" y="155" text-anchor="middle" class="label">Cluster A</text><text x="690" y="155" text-anchor="middle" class="label">Cluster B</text>
  """
        + "".join(point(x, y, "blue", 7) for x, y in [(245, 265), (285, 320), (330, 260), (355, 310), (275, 235)])
        + "".join(point(x, y, "orange", 7) for x, y in [(640, 270), (690, 235), (730, 305), (665, 325), (745, 255)])
        + point(450, 290, "green", 9)
        + '<text x="450" y="260" text-anchor="middle" class="label">point</text><text x="375" y="398" text-anchor="middle" class="small">a: close to own cluster</text><text x="595" y="398" text-anchor="middle" class="small">b: far from nearest other cluster</text>',
    )

    write_svg(
        "silhouette_plots.svg",
        """
  <text x="490" y="48" text-anchor="middle" class="title">Silhouette Plot Examples</text>
  <text x="265" y="96" text-anchor="middle" class="label">Good Clustering</text><text x="715" y="96" text-anchor="middle" class="label">Poor Clustering</text>
  <line x1="115" y1="385" x2="430" y2="385" class="axis" /><line x1="115" y1="125" x2="115" y2="385" class="axis" />
  <rect x="190" y="150" width="205" height="38" class="blue" opacity="0.85" /><rect x="175" y="220" width="220" height="38" class="orange" opacity="0.85" /><rect x="205" y="290" width="190" height="38" class="green" opacity="0.85" />
  <line x1="365" y1="125" x2="365" y2="385" class="dash" />
  <line x1="565" y1="385" x2="880" y2="385" class="axis" /><line x1="680" y1="125" x2="680" y2="385" class="axis" />
  <rect x="610" y="150" width="155" height="38" class="blue" opacity="0.85" /><rect x="625" y="220" width="105" height="38" class="orange" opacity="0.85" /><rect x="590" y="290" width="85" height="38" class="red" opacity="0.85" />
  <line x1="735" y1="125" x2="735" y2="385" class="dash" />
  <text x="265" y="430" text-anchor="middle" class="small">Most values are positive and wide.</text>
  <text x="715" y="430" text-anchor="middle" class="small">Some values are near zero or negative.</text>
  """,
    )

    write_svg(
        "hierarchical_clustering.svg",
        defs()
        + """
  <text x="490" y="48" text-anchor="middle" class="title">Hierarchical Clustering</text>
  <text x="265" y="100" text-anchor="middle" class="label">Agglomerative</text><text x="715" y="100" text-anchor="middle" class="label">Divisive</text>
  <rect x="140" y="140" width="250" height="55" class="box" /><text x="265" y="174" text-anchor="middle" class="small">5 points</text>
  <rect x="140" y="230" width="250" height="55" class="box" /><text x="265" y="264" text-anchor="middle" class="small">merge nearest clusters</text>
  <rect x="140" y="320" width="250" height="55" class="box" /><text x="265" y="354" text-anchor="middle" class="small">1 final hierarchy</text>
  """
        + arrow(265, 195, 265, 230)
        + arrow(265, 285, 265, 320)
        + """
  <rect x="590" y="140" width="250" height="55" class="box" /><text x="715" y="174" text-anchor="middle" class="small">1 large cluster</text>
  <rect x="590" y="230" width="250" height="55" class="box" /><text x="715" y="264" text-anchor="middle" class="small">split recursively</text>
  <rect x="590" y="320" width="250" height="55" class="box" /><text x="715" y="354" text-anchor="middle" class="small">many smaller clusters</text>
  """
        + arrow(715, 195, 715, 230)
        + arrow(715, 285, 715, 320),
    )

    write_svg(
        "dbscan_density.svg",
        """
  <text x="490" y="48" text-anchor="middle" class="title">DBSCAN: Density and Noise</text>
  <circle cx="340" cy="275" r="118" fill="#eaf3ff" stroke="#2f80ed" stroke-width="2" />
  <circle cx="640" cy="275" r="118" fill="#eafaf0" stroke="#27ae60" stroke-width="2" />
  """
        + "".join(point(x, y, "blue", 7) for x, y in [(285, 235), (330, 220), (365, 250), (305, 300), (355, 315), (395, 285), (320, 270), (375, 225)])
        + "".join(point(x, y, "green", 7) for x, y in [(590, 235), (625, 220), (680, 250), (610, 300), (655, 320), (705, 285), (640, 270), (690, 225)])
        + point(820, 395, "red", 8)
        + '<text x="340" y="430" text-anchor="middle" class="label">dense region = cluster</text><text x="640" y="430" text-anchor="middle" class="label">dense region = cluster</text><text x="820" y="430" text-anchor="middle" class="label">noise</text>',
    )

    write_svg(
        "distance_metrics.svg",
        """
  <text x="490" y="48" text-anchor="middle" class="title">Euclidean vs Manhattan Distance</text>
  <line x1="220" y1="390" x2="760" y2="390" class="axis" /><line x1="220" y1="390" x2="220" y2="120" class="axis" />
  <circle cx="300" cy="330" r="9" class="blue" /><text x="278" y="358" class="label">A</text>
  <circle cx="620" cy="150" r="9" class="orange" /><text x="635" y="145" class="label">B</text>
  <path d="M300 330 L620 150" stroke="#2f80ed" stroke-width="5" fill="none" />
  <path d="M300 330 L620 330 L620 150" stroke="#f2994a" stroke-width="5" fill="none" stroke-dasharray="12 10" />
  <text x="455" y="218" text-anchor="middle" class="small">Euclidean: straight line</text>
  <text x="500" y="360" text-anchor="middle" class="small">Manhattan: grid path</text>
  """,
    )

    write_svg(
        "clustering_algorithm_flowchart.svg",
        defs()
        + """
  <text x="490" y="48" text-anchor="middle" class="title">Choosing a Clustering Algorithm</text>
  <rect x="325" y="95" width="330" height="64" class="box" /><text x="490" y="134" text-anchor="middle" class="label">Do you know K?</text>
  <rect x="115" y="220" width="250" height="64" class="box" /><text x="240" y="259" text-anchor="middle" class="label">K-Means</text>
  <rect x="615" y="220" width="250" height="64" class="box" /><text x="740" y="250" text-anchor="middle" class="label">Elbow + Silhouette</text><text x="740" y="274" text-anchor="middle" class="small">or Hierarchical/DBSCAN</text>
  <rect x="115" y="365" width="250" height="64" class="box" /><text x="240" y="404" text-anchor="middle" class="label">DBSCAN</text>
  <rect x="615" y="365" width="250" height="64" class="box" /><text x="740" y="404" text-anchor="middle" class="label">GMM</text>
  """
        + arrow(405, 159, 285, 220)
        + arrow(575, 159, 695, 220)
        + arrow(240, 284, 240, 365)
        + arrow(740, 284, 740, 365)
        + '<text x="318" y="200" class="small">yes</text><text x="650" y="200" class="small">no</text><text x="257" y="336" class="small">outliers?</text><text x="758" y="336" class="small">probabilities?</text>',
    )


if __name__ == "__main__":
    main()
