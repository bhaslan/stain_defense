from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from src.stage05_C_vmatch import config_vmatch as vmatch_cfg
from src.stage06_repeat_enrichment import config_enrichment as cfg

CATEGORIES = ("none", "some", "all")
PARTNERS = ("same cluster, other genus", "same cluster, same genus", "other cluster, same genus")

@dataclass(frozen=True)
class CodedFlanks:

    cluster: np.ndarray
    genus: np.ndarray
    cell: np.ndarray
    cell_cluster: np.ndarray
    cell_genus: np.ndarray
    by_genus: np.ndarray

    @classmethod
    def of(cls, flanks):

        cluster, genus = pd.factorize(flanks["group_accession"])[0], pd.factorize(flanks["genus"])[0]
        cell, cells = pd.factorize(pd.MultiIndex.from_arrays([cluster, genus]))

        return cls(cluster, genus, cell, cells.get_level_values(0).to_numpy(), cells.get_level_values(1).to_numpy(), np.argsort(genus, kind="stable"))

    def shuffled(self, outcome, rng):

        shuffled = np.empty_like(outcome)
        shuffled[self.by_genus] = outcome[np.lexsort((rng.random(len(outcome)), self.genus))]

        return shuffled

    def categories(self, outcome):

        flanks, positive = np.bincount(self.cluster), np.bincount(self.cluster, weights=outcome)
        share = (positive / flanks)[flanks >= cfg.LINEAGE_MIN_FLANKS]

        return dict(zip(CATEGORIES, (np.mean(share == 0), np.mean((share > 0) & (share < 1)), np.mean(share == 1))))

    def partners(self, outcome):

        flanks, positive = np.bincount(self.cell), np.bincount(self.cell, weights=outcome)
        in_cluster = [np.bincount(self.cell_cluster, weights=values)[self.cell_cluster] for values in (flanks, positive)]
        in_genus = [np.bincount(self.cell_genus, weights=values)[self.cell_genus] for values in (flanks, positive)]
        kinds = dict(zip(PARTNERS, ((in_cluster[0] - flanks, in_cluster[1] - positive), (flanks - 1, positive - 1), (in_genus[0] - flanks, in_genus[1] - positive))))
        found = {}

        for kind, (partners, positive_partners) in kinds.items():

            pairs = positive @ partners
            clusters = len(np.unique(self.cell_cluster[(positive > 0) & (partners > 0)]))
            found[kind] = (positive @ positive_partners / pairs if pairs else np.nan, int(pairs), clusters)

        return found

def _against_shuffles(observed, shuffles):

    shuffles = shuffles[~np.isnan(shuffles)]
    p_value = min(max(2 * min((shuffles >= observed).mean(), (shuffles <= observed).mean()), 1 / len(shuffles)), 1.0)

    return {"observed": round(observed, 4), "shuffled": round(shuffles.mean(), 4), "shuffled_ci": np.percentile(shuffles, [2.5, 97.5]).round(4).tolist(), "p_value": p_value}

def lineages(flanks):

    outcome_column = f"occupied_{cfg.LINEAGE_WINDOW}"
    used = flanks.loc[(flanks["exclusion"] == "") & (flanks["repeat_set"] == vmatch_cfg.STEP5_PRIMARY_REPEAT_SET)
                      & (flanks[f"searched_bases_{cfg.LINEAGE_WINDOW}"] == cfg.LINEAGE_WINDOW)]
    rng = np.random.default_rng(cfg.BACKGROUND_SEED)
    category_rows, partner_rows = [], []

    for name, side in product(cfg.HTH_CLASSES, ("left", "right")):

        chosen = used.loc[(used["hth_class"] == name) & (used["side"] == side)]
        coded = CodedFlanks.of(chosen)
        outcome = chosen[outcome_column].to_numpy(dtype=float)
        shuffles = [coded.shuffled(outcome, rng) for _ in range(cfg.LINEAGE_PERMUTATIONS)]
        read = int((np.bincount(coded.cluster) >= cfg.LINEAGE_MIN_FLANKS).sum())

        if read:

            null = pd.DataFrame([coded.categories(shuffle) for shuffle in shuffles])
            category_rows.extend({"hth_class": name, "side": side, "category": category, "clusters": read, **_against_shuffles(observed, null[category].to_numpy())}
                                 for category, observed in coded.categories(outcome).items())

        observed = coded.partners(outcome)
        null = [coded.partners(shuffle) for shuffle in shuffles]

        for kind, (rate, pairs, clusters) in observed.items():

            if not pairs: continue
            partner_rows.append({"hth_class": name, "side": side, "partner": kind, "pairs": pairs, "clusters": clusters, "base_rate": round(outcome.mean(), 4),
                                 **_against_shuffles(rate, np.array([found[kind][0] for found in null]))})

    tables = pd.DataFrame(category_rows), pd.DataFrame(partner_rows)
    tables[0].to_csv(cfg.ALL_OR_NONE_PATH, sep="\t", index=False)
    tables[1].to_csv(cfg.FOLLOWS_PATH, sep="\t", index=False)

    return tables
