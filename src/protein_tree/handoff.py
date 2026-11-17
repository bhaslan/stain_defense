import json
import shutil
from pathlib import Path

from Bio import Phylo

from src import config_main as main_cfg
from src import roster
from src.protein_tree import config_tree as cfg
from src.protein_tree import relatives, samples
from src.protein_tree.samples import sha256

PROTEINS = "proteins.txt"

BRIEFS = {"full": """# Full-protein tree

Infer a maximum-likelihood tree of the proteins in `proteins.txt` from their full sequences,
`full.faa`. Do it however the cluster needs.

The proteins are the study's samples, one whole protein per cluster (truncated proteins are left
out), and 142 relatives of three anchors from the pilot run, listed in `relatives.tsv`, all pruned
from the tree afterwards. Treat them like any other protein.

Why the relatives: the anchors LacN, CfeN and MfoN sit on long branches with nothing of their own
around them. The previous tree carried only each anchor's 12 nearest relatives (`role` nearest):
each anchor then sat among its own relatives, but the three groups still joined into one clade. In
the pilot tree, other proteins absent from the study lie between those groups; they are added now
(`role` between) to see whether they separate them.

Bring back the whole folder, including the scripts that actually ran, their logs, and a note on
anything done differently. It must hold the tree as Newick, every tip one of `proteins.txt`, with
SH-aLRT/UFBoot support on internal branches, and the alignment and IQ-TREE report it came from.

Last time, for reference only: MAFFT --large --localpair, trimAl -gappyout (441 columns), then
IQ-TREE LG+F+R6 with -alrt 1000 -B 1000 -wbt from its default start; 2 h 25 min on 1,704 proteins.
"""}

def write_fasta(path, records):

    with open(path, "w", encoding="ascii") as handle:

        for accession, sequence in sorted(records.items()): handle.write(f">{accession}\n{sequence}\n")

def write_handoff(tree, fastas, accessions, extra_files=()):

    folder = cfg.HANDOFF_DIR / cfg.TREE_NAMES[tree]
    folder.mkdir(parents=True, exist_ok=True)

    for name, records in fastas.items(): write_fasta(folder / name, records)
    for path in extra_files: shutil.copy2(path, folder / path.name)

    (folder / PROTEINS).write_text("".join(f"{accession}\n" for accession in accessions))
    (folder / "BRIEF.md").write_text(BRIEFS[tree])

    source_paths = {"samples": cfg.SAMPLES_PATH, "relatives": cfg.RELATIVES_PATH, "study_roster": main_cfg.STUDY_ROSTER_PATH,
                    "study_sequences": main_cfg.STUDY_SEQUENCE_FASTA}
    written = [*fastas, *(path.name for path in extra_files), PROTEINS, "BRIEF.md"]
    manifest = {"proteins": len(accessions),
                "sources": {name: {"path": str(path), "sha256": sha256(path)} for name, path in source_paths.items()},
                "files": {name: sha256(folder / name) for name in written}}
    (folder / "source_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    return folder

def export():

    sampled = samples.accessions()
    borrowed = relatives.read()
    sequences = roster.sequences()
    full = {**{accession: sequences[accession] for accession in sampled}, **borrowed}
    accessions = sorted(full)

    folder = write_handoff("full", {"full.faa": full}, accessions, [cfg.RELATIVES_PATH])

    print(f"Full tree: {len(accessions):,} proteins, {len(sampled):,} samples and {len(borrowed):,} relatives")
    print(folder)

def import_tree(tree, bundle_dir, treefile):

    name = cfg.TREE_NAMES[tree]
    returned, destination = cfg.RETURNED_DIR / name, cfg.TREE_PATHS[tree]

    if returned.exists() or destination.exists(): raise FileExistsError(f"{returned} or {destination} already exists; raise config_tree.FULL_TREE_VERSION for a new version.")

    proteins = set((cfg.HANDOFF_DIR / name / PROTEINS).read_text().split())
    tips = {tip.name for tip in Phylo.read(Path(bundle_dir) / treefile, "newick").get_terminals()}

    if tips != proteins: raise ValueError(f"{treefile} has {len(proteins - tips)} missing and {len(tips - proteins)} extra proteins")

    shutil.copytree(bundle_dir, returned)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(returned / treefile, destination)

    print(f"{tree} tree: {len(tips):,} tips -> {destination}")
    print(returned)
