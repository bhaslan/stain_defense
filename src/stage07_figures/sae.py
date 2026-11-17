import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from src.stage04_sae import config_sae as sae_cfg
from src.stage07_figures import config_figures as cfg

def plot_sae_summary(joined_scores):

    import matplotlib.pyplot as plt

    required = {"local_HTH_status", "length", "sae_hth_support_score",
                "evidence_tier"}
    missing = sorted(required - set(joined_scores.columns))

    if missing:

        raise ValueError(f"Joined SAE score table is missing columns: {missing}")

    frame = joined_scores.copy()
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    order = [status for status in ("HTH_positive", "HTH_negative")
             if status in set(frame["local_HTH_status"])]

    if not order:

        raise ValueError("Joined SAE score table contains no recognized HTH status.")

    data = [frame.loc[frame["local_HTH_status"] == status,
                      "sae_hth_support_score"].dropna()
            for status in order]
    axes[0].boxplot(data, tick_labels=[status.replace("_", " ") for status in order],
                    showfliers=False)
    axes[0].set_title("SAE HTH score by local HTH status")
    axes[0].set_ylabel("Selected-feature weighted bin score")
    axes[1].scatter(frame["length"], frame["sae_hth_support_score"],
                    s=12, alpha=0.45)
    axes[1].set_title("SAE HTH score versus length")
    axes[1].set_xlabel("Protein length")
    axes[1].set_ylabel("SAE HTH weighted bin score")
    medians = (frame.groupby("evidence_tier")["sae_hth_support_score"]
               .median()
               .sort_values(ascending=False))
    axes[2].barh(np.arange(len(medians)), medians.values)
    axes[2].set_yticks(np.arange(len(medians)))
    axes[2].set_yticklabels([str(value).replace("_", " ") for value in medians.index],
                            fontsize=8)
    axes[2].invert_yaxis()
    axes[2].set_title("Median SAE HTH score by evidence tier")
    axes[2].set_xlabel("Median score")
    figure.tight_layout()

    return figure, axes

def draw():

    import matplotlib.pyplot as plt

    figure, _ = plot_sae_summary(pd.read_csv(sae_cfg.JOINED_SCORES_PATH, sep="\t"))
    path = cfg.SAE_DIR / f"sae_hth_binned_summary.{sae_cfg.MODEL_ID.replace('-', '_')}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    return [path]
