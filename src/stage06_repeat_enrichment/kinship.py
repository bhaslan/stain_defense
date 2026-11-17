import json

import numpy as np
import pandas as pd
from Bio import Phylo

from src.protein_tree import config_tree as tree_cfg
from src.protein_tree import samples
from src.stage06_repeat_enrichment import config_enrichment as cfg

def _depth(clade, parents):

    distance = 0.0

    while clade in parents:

        if clade.branch_length is not None: distance = distance + clade.branch_length
        clade = parents[clade]

    return distance

def covariance(tree):

    parents = {child: clade for clade in tree.find_clades() for child in clade.clades}
    tips = tree.get_terminals()
    index = {tip: position for position, tip in enumerate(tips)}
    span = {}

    for clade in tree.find_clades(order="postorder"):

        if clade.is_terminal(): span[clade] = (index[clade], index[clade] + 1)
        else: span[clade] = (span[clade.clades[0]][0], span[clade.clades[-1]][1])

    matrix = np.empty((len(tips), len(tips)))

    for clade in tree.find_clades():

        shared = _depth(clade, parents)

        if clade.is_terminal():

            matrix[index[clade], index[clade]] = shared
            continue

        children = [span[child] for child in clade.clades]

        for position, (first, last) in enumerate(children):

            for other_first, other_last in children[position + 1:]:

                matrix[first:last, other_first:other_last] = shared
                matrix[other_first:other_last, first:last] = shared

    names = [tip.name for tip in tips]

    return pd.DataFrame(matrix, index=names, columns=names)

def build_kinship(name):

    tree_path = tree_cfg.PRUNED_TREE_PATHS[name]
    tree = Phylo.read(tree_path, "newick")

    if sorted(tip.name for tip in tree.get_terminals()) != samples.accessions():

        raise ValueError(f"{tree_path} has tips other than the samples in {tree_cfg.SAMPLES_PATH}; rerun the protein tree's prune")

    source = json.loads(samples.MANIFEST_PATH.read_text())["trees"][name]
    destination = cfg.KINSHIP_DIR / f"{name}.tsv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    matrix = covariance(tree)
    matrix.to_csv(destination, sep="\t")

    metadata = {"pruned_tree_sha256": samples.sha256(tree_path), "source_tree_sha256": source["source_tree_sha256"],
                "matrix_sha256": samples.sha256(destination),
                "rooting": "as serialized; association covariance is projected against an intercept"}
    destination.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"Stage 06 kinship ({name} tree): {len(matrix):,} samples")
    print(destination)

    return destination

def verified(path):

    metadata = json.loads(path.with_suffix(".json").read_text())
    tree_path = tree_cfg.PRUNED_TREE_PATHS[path.stem]
    tree = samples.sha256(tree_path)
    pruned = json.loads(samples.MANIFEST_PATH.read_text())["trees"][path.stem]["pruned_tree_sha256"]

    if not metadata["pruned_tree_sha256"] == tree == pruned:

        raise RuntimeError(f"{path} was built from another pruned tree than {tree_path}; "
                           "rerun step2_build_kinship after the protein tree's prune.")

    if metadata["matrix_sha256"] != samples.sha256(path):

        raise RuntimeError(f"{path} changed after it was built; rerun step2_build_kinship.")

    return path

def frozen(path):

    metadata = json.loads(path.with_suffix(".json").read_text())
    tree_path = tree_cfg.PRUNED_TREE_PATHS[path.stem]

    if path.stem not in tree_cfg.FROZEN_TREES: raise ValueError(f"The {path.stem} tree is not frozen; read its kinship through verified().")

    if metadata["pruned_tree_sha256"] != samples.sha256(tree_path) or metadata["matrix_sha256"] != samples.sha256(path):

        raise RuntimeError(f"{path} or {tree_path} changed after the matrix was built, though the tree is frozen.")

    return path
