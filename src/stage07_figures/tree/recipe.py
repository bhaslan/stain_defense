import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.stage07_figures import config_figures as cfg
from src.stage07_figures.tree import prepare
from src.stage07_figures.tree.prepare import ANCHOR_TIERS, ATTRIBUTES, FIGURES

CODE_DIR = Path(__file__).resolve().parent
CLADE_COLOUR_FORMATTER = CODE_DIR / "clade_colour.cs"
CLADE_COLOURS_TEMPLATE = CODE_DIR / "clade_colours.cs"

CUSTOM_SCRIPT_ID = "a76d00d2-95e0-4274-a77d-1439a013e3d9"

TREE_RADIUS = 1500
RING_START, RING_HEIGHT, RING_GAP = 1560, 64, 10
PNG_WIDTH = 4000
LABEL_STEP = 60
ANCHOR_FONT = 20

ZERO_COLOUR = "#F4F4F4"
BRANCH_WEIGHT = 3

@dataclass(frozen=True)
class CategoryRing:

    attribute: str
    label: str
    colours: dict

    @property
    def formatter(self): return cfg.TREE_BUILD_DIR / f"formatter_{self.attribute}.cs"

    @property
    def legend(self): return f"**{self.label}** " + " ".join(f"![](square://14,{c}) {name}" for name, c in self.colours.items())

    def write_formatter(self):

        colours = ",\n".join(f"    Colour.FromRgba({rgba(colour)})" for colour in self.colours.values())
        self.formatter.write_text(CATEGORY_TEMPLATE.replace("COLOURS", colours))

@dataclass(frozen=True)
class LinearRing:

    attribute: str
    label: str
    colours: tuple
    top: float
    unit: str = ""

    @property
    def formatter(self): return cfg.TREE_BUILD_DIR / f"formatter_{self.attribute}.cs"

    @property
    def legend(self):

        labels = (">0", f"{self.top / 2:g}", f"{self.top:g}")
        swatches = " → ".join(f"![](square://14,{c}) {label}" for c, label in zip(self.colours, labels))

        return f"**{self.label}**{self.unit} ![](square://14,{ZERO_COLOUR}) 0 · {swatches}"

    def write_formatter(self):

        stops = ",\n".join(f"    new TreeViewer.GradientStop({position}, Colour.FromRgba({rgba(colour)}))"
                           for position, colour in zip((0.0, 0.5, 1.0), self.colours))
        self.formatter.write_text(FORMATTER_TEMPLATE.replace("STOPS", stops).replace("ZERO", rgba(ZERO_COLOUR))
                                  .replace("TOP", f"{self.top}d"))

def rgba(colour): return ", ".join(f"{int(colour[i:i + 2], 16) / 255:.4f}d" for i in (1, 3, 5)) + ", 1d"

FORMATTER_TEMPLATE = """static Gradient gradient = new Gradient(new List<TreeViewer.GradientStop>()
{
STOPS
});

public static Colour? Format(object attribute)
{
    if (attribute is double value && !double.IsNaN(value))
    {
        if (value == 0) return Colour.FromRgba(ZERO);
        return gradient.GetColour(Math.Min(value / TOP, 1));
    }

    return null;
}
"""

CATEGORY_TEMPLATE = """static Colour[] colours = new Colour[]
{
COLOURS
};

public static Colour? Format(object attribute)
{
    if (attribute is double code && !double.IsNaN(code)) return colours[(int)code];

    return null;
}
"""

RINGS = (CategoryRing("hth_class", "HTH class", cfg.HTH_CLASS_COLOURS),
         LinearRing("repeats", "Repeats", ("#C8C8C8", "#787878", "#000000"), 1, f" (share of flanks with any repeat within {cfg.ENRICHMENT_WINDOW} bp)"),
         CategoryRing("phylum", "Host phylum", cfg.PHYLUM_COLOURS))

def clade_legend(figure):

    clades = pd.read_csv(figure.clades, keep_default_na=False)
    swatches = [f"![](square://14,{cfg.CLADE_COLOURS[row.clade]}) {row.tips} tips (UFBoot {round(float(row.support.split('/')[1]))}){': ' + row.anchors if row.anchors else ''}"
                for row in clades.itertuples()]

    return "**Clades** (SH-aLRT ≥ 80 and UFBoot ≥ 95; grey: backbone)  \n" + "  \n".join(swatches)

def legend(figure): return "\n\n".join(["### **Rings, inside out**", *(ring.legend for ring in RINGS), clade_legend(figure)]) + "\n"

def enable(module):

    return [f"module enable {module}", "y"]

def options(*pairs):

    return [line for name, value in pairs for line in (f"option select {name}", f"option set {value}")]

def ring_lines(ring, radius, n_tips):

    span = 2 * math.pi * radius / n_tips * 1.02

    return (enable("Node states")
            + options(("Show on:", "Tips"), ("Anchor:", "Origin"), ("Orientation reference:", "Branch"),
                      ("Position:", f"{radius}, 0"), ("Width:", RING_HEIGHT), ("Height:", f"{span:.2f}"),
                      ("Stroke thickness:", 0), ("Attribute:", ring.attribute), ("Attribute type:", "Number"),
                      ("Missing data:", "Unknown state"))
            + ["option select State colours:", f"option set source {ring.formatter}", "option set #FFFFFF00"]
            + options(("Type:", "Rectangle")))

def recipe(figure):

    radii = [RING_START + i * (RING_HEIGHT + RING_GAP) for i in range(len(RINGS))]
    n_tips = len(pd.read_csv(figure.tips))
    outer = radii[-1] + RING_HEIGHT / 2
    lines = [f"open {figure.display_tree}", "y", "module enable Circular tree style", "module disable Labels",
             "module select Circular", "option select Outer radius:", f"option set {TREE_RADIUS}", "update all",
             ]
    for attribute in ATTRIBUTES:

        lines += [f"attachment add {figure.attribute_path(attribute)}", attribute, "y", "y"]
        lines += enable("Parse node states") + options(("Data file", attribute), ("Separator", ","), ("Regex", "false"), ("Use first row as header", "true"))

    lines += ["update all"]
    lines += (enable(CUSTOM_SCRIPT_ID) + ["option select Source code:", f"option set source {figure.build_dir / 'clade_colours.cs'}", "update all"]
              + ["module select Branches", "option select Line weight:", f"option set {BRANCH_WEIGHT}",
                 "option select Auto colour by node", "option set false", "option select Colour:",
                 "option set attribute string clade_colour", f"option set source {CLADE_COLOUR_FORMATTER}"])
    lines += (enable("Branch extensions")
              + options(("End anchor:", "Origin"), ("End:", f"{RING_START - 4}, 0"), ("Line colour:", "#E4E4E4"),
                        ("Line weight:", 0.25)))
    lines += [line for ring, radius in zip(RINGS, radii) for line in ring_lines(ring, radius, n_tips)]
    for tier in range(ANCHOR_TIERS):

        lines += (enable("Labels")
                  + options(("Show on:", "Leaves"), ("Attribute:", f"anchor_{tier}"), ("Anchor:", "Origin"),
                            ("Position:", f"{outer + 15 + tier * LABEL_STEP}, 0"), ("Reference:", "Branch"), ("Font:", f"Helvetica-Bold {ANCHOR_FONT}")))

    lines += (enable("Labels")
              + options(("Show on:", "Internal nodes"), ("Attribute:", "support_label"), ("Reference:", "Horizontal"),
                        ("Font:", f"Helvetica-Bold {2 * ANCHOR_FONT}"), ("Background colour:", "#FFFFFF")))

    lines += (enable("Scale bar")
              + options(("Scale size:", "0.5"), ("Line thickness:", 3), ("Font:", "Helvetica 18"), ("Position:", f"{-0.82 * outer}, {0.22 * outer}")))
    lines += (enable("Legend")
              + ["option select Markdown source", f"option set source {figure.build_dir / 'legend.md'}"]
              + options(("Width", 640), ("Font size", 15), ("Anchor", "Node"), ("Alignment", "Top-left"), ("Position", f"{-outer - 60}, {-outer - 60}")))
    lines += ["update all", f"pdf {figure.build_dir / 'rings.pdf'}", f"binary modules loaded {figure.stem}.tbi", "n", "y", "exit"]

    return lines

def draw():

    prepare.write_inputs()

    for ring in RINGS: ring.write_formatter()

    for figure in FIGURES:

        (figure.build_dir / "legend.md").write_text(legend(figure))
        palette = ", ".join(f'"{colour}"' for colour in cfg.CLADE_COLOURS)
        (figure.build_dir / "clade_colours.cs").write_text(CLADE_COLOURS_TEMPLATE.read_text().replace("PALETTE", palette))
        recipe_path, log_path = figure.build_dir / "rebuild.txt", figure.build_dir / "rebuild.log"
        recipe_path.write_text("\n".join(recipe(figure)) + "\n")

        with open(recipe_path) as stdin, open(log_path, "w") as log:

            subprocess.run(["TreeViewerCommandLine"], stdin=stdin, stdout=log, stderr=subprocess.STDOUT,
                           env={**os.environ, "NO_COLOR": "true"}, timeout=300, check=True)

        subprocess.run(["pdftoppm", "-png", "-singlefile", "-scale-to", str(PNG_WIDTH), figure.build_dir / "rings.pdf", figure.stem], check=True)

    return [figure.stem.with_suffix(".png") for figure in FIGURES]
