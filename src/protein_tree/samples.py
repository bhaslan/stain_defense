import hashlib
import json
from pathlib import Path

import pandas as pd
from Bio import Phylo, SeqIO

from src.protein_tree import config_tree as cfg
from src.protein_tree import cores

MANIFEST_PATH = cfg.SAMPLES_DIR / "samples.json"

def sha256(path):

    digest = hashlib.sha256()

    with open(path, "rb") as handle:

        for chunk in iter(lambda: handle.read(1 << 20), b""): digest.update(chunk)

    return digest.hexdigest()

def anchors():

    return {record.id for record in SeqIO.parse(cores.SOURCES["anchors"], "fasta")}

def clusters():

    table = pd.read_csv(cfg.PROTEIN_ARCHITECTURES_PATH, usecols=["protein_accession", "representative_accession"])

    return table.set_index("protein_accession", verify_integrity=True)["representative_accession"]

def assign(cluster, status, anchored):

    intact = status.loc[status["status"] == "intact", "length"]
    members = cluster.loc[intact.index]
    sample_of = {}

    for representative, group in members.groupby(members):

        held = set(group.index)
        chosen = held & anchored

        if len(chosen) > 1: raise ValueError(f"Cluster {representative} holds anchors {', '.join(sorted(chosen))}; only one of them can be its sample.")

        if chosen: sample_of[representative] = chosen.pop()
        elif representative in held: sample_of[representative] = representative
        else: sample_of[representative] = min(held, key=lambda accession: (-intact[accession], accession))

    table = pd.DataFrame({"protein_accession": members.index, "sample_accession": members.map(sample_of).to_numpy()})
    replaced = {representative: sample for representative, sample in sample_of.items() if sample != representative}

    return table.sort_values("protein_accession", ignore_index=True), replaced

def write_samples():

    status = pd.read_csv(cfg.PROTEIN_STATUS_PATH, sep="\t").set_index("protein_accession")
    anchored = anchors()
    table, replaced = assign(clusters(), status, anchored)
    cfg.SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(cfg.SAMPLES_PATH, sep="\t", index=False)

    manifest = {"samples_sha256": sha256(cfg.SAMPLES_PATH), "protein_status_sha256": sha256(cfg.PROTEIN_STATUS_PATH),
                "replaced_representatives": {representative: {"sample": sample, "why": "anchor" if sample in anchored else "representative_not_intact"}
                                             for representative, sample in sorted(replaced.items())},
                "counts": {"samples": int(table["sample_accession"].nunique()), "proteins": len(table)}}
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    return manifest

def prune():

    manifest = json.loads(MANIFEST_PATH.read_text())

    if manifest["samples_sha256"] != sha256(cfg.SAMPLES_PATH): raise RuntimeError(f"{cfg.SAMPLES_PATH} changed after its manifest was written; rerun samples.")

    wanted = set(accessions())
    pruned = {}

    for name in cfg.TREES:

        if name in cfg.FROZEN_TREES: continue
        tree = Phylo.read(cfg.TREE_PATHS[name], "newick")
        tips = {tip.name for tip in tree.get_terminals()}
        if wanted - tips: raise ValueError(f"The {name} tree lacks {len(wanted - tips):,} samples, e.g. {', '.join(sorted(wanted - tips)[:5])}.")
        root = tree.root
        for tip in sorted(tips - wanted): tree.prune(tip)
        if tree.root is not root: raise RuntimeError(f"Pruning moved the {name} tree's root, which would change the remaining tips' distances")

        Phylo.write(tree, cfg.PRUNED_TREE_PATHS[name], "newick", format_branch_length="%.10f")
        pruned[name] = {"source_tree": str(cfg.TREE_PATHS[name]), "source_tree_sha256": sha256(cfg.TREE_PATHS[name]),
                        "pruned_tree": str(cfg.PRUNED_TREE_PATHS[name]), "pruned_tree_sha256": sha256(cfg.PRUNED_TREE_PATHS[name]),
                        "source_tips": len(tips), "pruned_tips": len(tips - wanted)}

    manifest["trees"] = pruned
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    return manifest

def accessions():

    return sorted(assignment().unique())

def pruned_from(tree):

    return Path(json.loads(MANIFEST_PATH.read_text())["trees"][tree]["source_tree"]).stem

def assignment():

    table = pd.read_csv(cfg.SAMPLES_PATH, sep="\t")

    return table.set_index("protein_accession", verify_integrity=True)["sample_accession"]
