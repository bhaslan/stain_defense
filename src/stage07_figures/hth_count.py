import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.protein_tree import config_tree as tree_cfg
from src.stage07_figures import config_figures as cfg

ROWS = ("4 HTH", "3 HTH", "2 HTH", "1 HTH", "Schlafen + RecG only")
BAR_COLOUR = "#465566"

def draw():

    table = pd.read_csv(tree_cfg.PROTEIN_STATUS_PATH, sep="\t")
    intact = table.loc[table["status"] == "intact"]
    counts = intact["architecture"].map(_row).value_counts().reindex(ROWS, fill_value=0)

    figure, axis = plt.subplots(figsize=cfg.HTH_COUNT_SIZE)
    axis.barh(range(len(ROWS)), counts.to_numpy(), color=BAR_COLOUR)
    axis.set_yticks(range(len(ROWS)), ROWS)
    axis.invert_yaxis()

    for position, count in enumerate(counts): axis.text(count, position, f" {count:,}", va="center", fontsize=8)

    axis.set_xlabel("Intact proteins")
    axis.set_xlim(0, counts.max() * 1.25)
    figure.tight_layout()

    cfg.HTH_COUNT_DIR.mkdir(parents=True, exist_ok=True)
    path = cfg.HTH_COUNT_DIR / "MAIN_hth_count.png"
    figure.savefig(path, dpi=cfg.DPI)
    plt.close(figure)
    counts.rename("proteins").to_csv(cfg.HTH_COUNT_DIR / "hth_count.tsv", sep="\t", index_label="row")

    return [path]

def _row(architecture):

    tokens = architecture.split(" -- ")
    hth = tokens.count("HTH")

    if hth: return f"{hth} HTH"
    if tokens == ["Schlafen", "RecG"]: return "Schlafen + RecG only"

    raise ValueError(f"An intact protein without an HTH has neither Schlafen nor RecG alone: {architecture}")
