from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import Phylo

from src import config_main as main_cfg
from src.protein_tree import config_tree as tree_cfg
from src.protein_tree import samples
from src.stage05_C_vmatch import config_vmatch as vmatch_cfg
from src.stage05_C_vmatch import config_vmatch
from src.stage06_repeat_enrichment import config_enrichment as enrichment_cfg
from src.stage07_figures import config_figures as cfg

ANCHOR_CLASH, ANCHOR_TIERS = 15, 3

ATTRIBUTES = ("hth_class", "repeats", "phylum", "clade", *(f"anchor_{tier}" for tier in range(ANCHOR_TIERS)))

CLADES, CLADE_MIN_TIPS = 6, 20
CLADE_SUPPORT = (80, 95)
MAD_MIN_DISTANCE = 0.01
DISPLAY_MIN_TIP_LENGTH = 1e-6

@dataclass(frozen=True)
class Figure:

    name: str
    source_tree: Path
    title: str

    @property
    def build_dir(self): return cfg.TREE_BUILD_DIR / self.name

    @property
    def display_tree(self): return self.build_dir / f"{self.name}.display.treefile"

    @property
    def tips(self): return self.build_dir / f"{self.name}_tips.csv"

    @property
    def clades(self): return self.build_dir / f"{self.name}_clades.csv"

    @property
    def stem(self): return cfg.TREE_DIR / f"MAIN_{self.name}_rings"

    def attribute_path(self, attribute): return self.build_dir / f"tips_{attribute}.csv"

FIGURES = (Figure(samples.pruned_from("full"), tree_cfg.PRUNED_TREE_PATHS["full"], "the pruned full-protein tree"),)

TAXONOMY_DIR = Path(main_cfg.p_data_db) / "Kraken2" / "Standard"
HTH_CLASSES = tuple(cfg.HTH_CLASS_COLOURS)
PHYLA = tuple(cfg.PHYLUM_COLOURS)

def phylum_names(taxids):

    parent, rank = {}, {}

    with open(TAXONOMY_DIR / "nodes.dmp") as f:

        for line in f:

            taxid, up, level = line.split("\t|\t", 3)[:3]
            parent[taxid], rank[taxid] = up, level

    def phylum(taxid):

        while taxid in parent and rank[taxid] != "phylum" and taxid != "1": taxid = parent[taxid]

        return taxid if rank.get(taxid) == "phylum" else None

    phyla = {taxid: phylum(taxid) for taxid in taxids}
    wanted, names = set(phyla.values()) - {None}, {}

    with open(TAXONOMY_DIR / "names.dmp") as f:

        for line in f:

            taxid, name, _, kind = line.rstrip("\t|\n").split("\t|\t")
            if taxid in wanted and kind == "scientific name": names[taxid] = name

    return {taxid: names[p] if p else None for taxid, p in phyla.items()}

def sample_phyla():

    samples = pd.read_csv(tree_cfg.SAMPLES_PATH, sep="\t", dtype=str)
    contexts = pd.read_csv(config_vmatch.STEP2_CONTEXTS_PATH, sep="\t", usecols=["protein_accession", "taxid"], dtype=str)
    contexts = samples.merge(contexts, on="protein_accession")
    names = phylum_names(set(contexts["taxid"]))
    contexts["phylum"] = contexts["taxid"].map(names)

    return contexts.dropna(subset=["phylum"]).groupby("sample_accession")["phylum"].agg(lambda x: Counter(x).most_common(1)[0][0])

def mad_root(tree):

    tips = tree.get_terminals()
    nodes = list(tree.find_clades(order="preorder"))
    column = {id(node): k for k, node in enumerate(nodes)}
    parent = {id(child): node for node in nodes for child in node.clades}
    below = np.zeros((len(tips), len(nodes)), bool)
    row = {id(tip): i for i, tip in enumerate(tips)}

    for k, node in enumerate(nodes):

        for tip in node.get_terminals(): below[row[id(tip)], k] = True

    depth = tree.depths()
    u = np.zeros(below.shape)
    u[:, 0] = [depth[tip] for tip in tips]

    for k, node in enumerate(nodes[1:], 1):

        length = node.branch_length or 0
        u[:, k] = u[:, column[id(parent[id(node)])]] + np.where(below[:, k], -length, length)

    distance = u[:, [column[id(tip)] for tip in tips]]
    kept = distance >= MAD_MIN_DISTANCE
    w = np.where(kept, 1 / np.where(kept, distance, 1) ** 2, 0)
    s = 2 * below - 1.0
    w1, wu, ws = w.sum(1), w @ u, w @ s
    a = (u ** 2 * w1[:, None]).sum(0) - (u * wu).sum(0)
    b = (u * s * w1[:, None]).sum(0) - (u * ws).sum(0)
    c = (s * s * w1[:, None]).sum(0) - (s * ws).sum(0)
    best = None

    for k in range(1, len(nodes)):

        x = float(np.clip(-b[k] / c[k], 0, nodes[k].branch_length or 0))
        deviation = a[k] + 2 * b[k] * x + c[k] * x * x
        if best is None or deviation < best[0]: best = (deviation, nodes[k], x)

    return best[1], best[2]

def write_display_tree(figure):

    tree = Phylo.read(figure.source_tree, "newick")
    labels = splits(tree)
    clade, height = mad_root(tree)
    tree.root_with_outgroup(clade, outgroup_branch_length=height)

    for node, split in _sides(tree): node.name = labels[split]

    tree.ladderize()

    for tip in tree.get_terminals(): tip.branch_length = max(tip.branch_length or 0, DISPLAY_MIN_TIP_LENGTH)

    Phylo.write(tree, figure.display_tree, "newick", format_branch_length="%.10f")

    return tree

def _sides(tree):

    tips = frozenset(tip.name for tip in tree.get_terminals())
    first = tree.get_terminals()[0].name

    for node in tree.find_clades():

        if node.clades and node is not tree.root:

            side = frozenset(tip.name for tip in node.get_terminals())
            yield node, side if first not in side else tips - side

def splits(tree): return {split: node.name for node, split in _sides(tree)}

def supported(node):

    try: return all(float(value) >= bar for value, bar in zip(node.name.split("/"), CLADE_SUPPORT))
    except (AttributeError, ValueError): return False

def clades(tree):

    def qualifying(groups): return [g for g in groups if supported(g) and g.count_terminals() >= CLADE_MIN_TIPS]

    groups = list(tree.root.clades)

    while True:

        weak = [g for g in groups if g.clades and not supported(g)]
        large = [g for g in qualifying(groups) if g.clades]
        if weak: split = max(weak, key=lambda g: g.count_terminals())
        elif len(qualifying(groups)) < CLADES and large: split = max(large, key=lambda g: g.count_terminals())
        else: break
        groups.remove(split)
        groups += split.clades

    return sorted(qualifying(groups), key=lambda g: -g.count_terminals())

def sample_repeats(repeat_set="all"):

    window = cfg.ENRICHMENT_WINDOW
    flanks = pd.read_csv(enrichment_cfg.BACKGROUND_FLANKS_PATH, sep="\t", dtype={"exclusion": str})
    flanks = flanks.loc[flanks["exclusion"].isna() & (flanks["repeat_set"] == repeat_set) & (flanks[f"searched_bases_{window}"] == window)]

    return flanks.groupby("group_accession")[f"occupied_{window}"].mean()

def anchor_tiers(anchors):

    tiers, previous, tier = [], None, 0

    for position in anchors.dropna().index:

        tier = (tier + 1) % ANCHOR_TIERS if previous is not None and position - previous < ANCHOR_CLASH else 0
        tiers.append(tier)
        previous = position

    return pd.Series(tiers, index=anchors.dropna().index).reindex(anchors.index)

def write_tips(figure, tree, phyla, repeats):

    tips = [tip.name for tip in tree.get_terminals()]
    summary = pd.read_csv(vmatch_cfg.STEP6_SAMPLE_SUMMARY_PATH, sep="\t").set_index("protein_accession")
    clade_of, rows = {}, []

    for code, clade in enumerate(clades(tree)):

        members = [tip.name for tip in clade.get_terminals()]
        clade_of.update(dict.fromkeys(members, code))
        rows.append({"clade": code, "tips": len(members), "support": clade.name,
                     "anchors": ", ".join(summary.loc[members, "anchor_name"].dropna().sort_values())})

    pd.DataFrame(rows).to_csv(figure.clades, index=False)
    phyla = phyla.reindex(tips)
    phyla = phyla.where(phyla.isin(PHYLA) | phyla.isna(), "other")

    table = pd.DataFrame({"Name": tips,
                          "hth_class": [HTH_CLASSES.index(c) for c in summary.loc[tips, "hth_class"]],
                          "repeats": repeats.reindex(tips).to_numpy(),
                          "phylum": phyla.map(PHYLA.index, na_action="ignore").astype("Int64").to_numpy(),
                          "clade": pd.Series(tips).map(clade_of).astype("Int64").to_numpy(),
                          "anchor": summary.loc[tips, "anchor_name"].to_numpy()})
    tiers = anchor_tiers(table["anchor"])
    for tier in range(ANCHOR_TIERS): table[f"anchor_{tier}"] = table["anchor"].where(tiers == tier)

    table.to_csv(figure.tips, index=False)
    for attribute in ATTRIBUTES: table[["Name", attribute]].dropna().to_csv(figure.attribute_path(attribute), index=False)

    return table

def write_inputs():

    phyla, repeats = sample_phyla(), sample_repeats()

    for figure in FIGURES:

        figure.build_dir.mkdir(parents=True, exist_ok=True)
        table = write_tips(figure, write_display_tree(figure), phyla, repeats)

        print(f"{figure.name}: {len(table):,} tips -> {figure.tips}")
