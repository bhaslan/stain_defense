import argparse
import hashlib
import re
import subprocess
import tempfile
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd
from Bio import SeqIO

from src import config_main as main_cfg
from src import roster
from src.stage02_domains import config_domains as cfg
from src.stage02_domains import dpam

INTERPRO_COLUMNS = [
    "seqid",
    "md5",
    "seq_len",
    "analysis",
    "sig_acc",
    "sig_desc",
    "start",
    "end",
    "score",
    "status",
    "date",
    "ipr_acc",
    "ipr_desc",
    "go",
    "pathway"]

REPRESENTATIVE_CLUSTER_FIELDS = [
    "protein_accession",
    "cluster_id",
    "representative_accession",
    "cluster_size"]

REPRESENTATIVE_FIELDS = [
    "cluster_id",
    "representative_accession",
    "cluster_size"]

def read_interpro_tsv(path):

    df = pd.read_csv(path, sep="\t", names=INTERPRO_COLUMNS)
    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)
    df["seq_len"] = df["seq_len"].astype(int)
    df["ipr_acc"] = df["ipr_acc"].replace("-", pd.NA)
    df["ipr_desc"] = df["ipr_desc"].replace("-", pd.NA)
    df["sig_desc"] = df["sig_desc"].replace("-", pd.NA)
    df["sig_key"] = df["analysis"] + ":" + df["sig_acc"]

    return df

def interval_stats(a_start, a_end, b_start, b_end):

    a_len = a_end - a_start + 1
    b_len = b_end - b_start + 1
    ov_start = max(a_start, b_start)
    ov_end = min(a_end, b_end)
    overlap = max(0, ov_end - ov_start + 1)
    union = a_len + b_len - overlap

    return {
        "overlap": overlap,
        "a_len": a_len,
        "b_len": b_len,
        "a_cov": overlap / a_len,
        "b_cov": overlap / b_len,
        "jaccard": overlap / union,
        "smaller_cov": overlap / min(a_len, b_len),
        "larger_cov": overlap / max(a_len, b_len),
        "length_ratio": min(a_len, b_len) / max(a_len, b_len)}

def classify_overlap(stats, reciprocal=0.35, containment=0.95):

    if stats["a_cov"] >= reciprocal and stats["b_cov"] >= reciprocal:

        return "same_locus"

    if max(stats["a_cov"], stats["b_cov"]) >= containment:

        return "nested"

    if stats["overlap"] > 0:

        return "weak_overlap"

    return "no_overlap"

def overlapping_signature_pairs(df):

    rows = []

    for seqid, group in df.groupby("seqid"):

        group = group.reset_index(drop=True)

        for i, j in combinations(group.index, 2):

            a = group.loc[i]
            b = group.loc[j]
            stats = interval_stats(a.start, a.end, b.start, b.end)
            relation = classify_overlap(stats)

            if relation == "no_overlap":

                continue

            rows.append({
                "seqid": seqid,
                "a_key": a.sig_key,
                "b_key": b.sig_key,
                "a_analysis": a.analysis,
                "b_analysis": b.analysis,
                "a_sig_acc": a.sig_acc,
                "b_sig_acc": b.sig_acc,
                "a_desc": a.sig_desc,
                "b_desc": b.sig_desc,
                "a_ipr": a.ipr_acc,
                "b_ipr": b.ipr_acc,
                "a_start": a.start,
                "a_end": a.end,
                "b_start": b.start,
                "b_end": b.end,
                "relation": relation,
                **stats})

    pairs = pd.DataFrame(rows)

    if len(pairs):

        swap = pairs["a_key"] > pairs["b_key"]
        a_cols = [column for column in pairs.columns if column.startswith("a_")]
        b_cols = [column for column in pairs.columns if column.startswith("b_")]

        for a_col, b_col in zip(a_cols, b_cols):

            pairs.loc[swap, [a_col, b_col]] = pairs.loc[swap, [b_col, a_col]].values

    return pairs

def summarize_pairs(df, pairs):

    sig_counts = df.groupby("sig_key")["seqid"].nunique()
    summary = (pairs.groupby(["a_key", "b_key"])
               .agg(n_proteins=("seqid", "nunique"),
                    n_observations=("seqid", "size"),
                    same_locus_count=("relation", lambda x: (x == "same_locus").sum()),
                    nested_count=("relation", lambda x: (x == "nested").sum()),
                    median_a_cov=("a_cov", "median"),
                    median_b_cov=("b_cov", "median"),
                    median_jaccard=("jaccard", "median"),
                    median_length_ratio=("length_ratio", "median"),
                    a_desc=("a_desc", "first"),
                    b_desc=("b_desc", "first"),
                    a_ipr=("a_ipr", "first"),
                    b_ipr=("b_ipr", "first"),)
               .reset_index())
    summary["a_total_proteins"] = summary["a_key"].map(sig_counts)
    summary["b_total_proteins"] = summary["b_key"].map(sig_counts)
    summary["p_b_given_a"] = summary["n_proteins"] / summary["a_total_proteins"]
    summary["p_a_given_b"] = summary["n_proteins"] / summary["b_total_proteins"]
    summary["reciprocal_agreement"] = summary[["p_b_given_a", "p_a_given_b"]].min(axis=1)

    return summary.sort_values(["reciprocal_agreement", "n_proteins", "median_jaccard"],
                               ascending=False)

def candidate_equivalences(pair_summary):

    candidates = pair_summary.copy()
    candidates["same_ipr"] = (candidates["a_ipr"].notna()
                              & candidates["b_ipr"].notna()
                              & (candidates["a_ipr"] == candidates["b_ipr"]))
    candidates["strong_empirical_pair"] = ((candidates["n_proteins"] >= 3)
                                           & (candidates["reciprocal_agreement"] >= 0.80)
                                           & (candidates["median_a_cov"] >= 0.40)
                                           & (candidates["median_b_cov"] >= 0.40))
    candidates = candidates[candidates["same_ipr"] | candidates["strong_empirical_pair"]]

    return candidates.sort_values(["same_ipr", "reciprocal_agreement", "n_proteins"],
                                  ascending=False)

def find_parent(parent, x):

    while parent[x] != x:

        x = parent[x]

    return x

def union(parent, a, b):

    root_a = find_parent(parent, a)
    root_b = find_parent(parent, b)

    if root_a != root_b:

        parent[root_b] = root_a

def consensus_loci(df, reciprocal=0.35, containment=0.95):

    rows = []

    for seqid, group in df.groupby("seqid"):

        group = group.reset_index(drop=True)
        parent = list(range(len(group)))

        for i, j in combinations(group.index, 2):

            a = group.loc[i]
            b = group.loc[j]
            stats = interval_stats(a.start, a.end, b.start, b.end)
            relation = classify_overlap(stats, reciprocal=reciprocal, containment=containment)

            if relation in ("same_locus", "nested"):

                union(parent, i, j)

        groups = defaultdict(list)

        for i in group.index:

            groups[find_parent(parent, i)].append(i)

        for locus_number, indices in enumerate(groups.values(), 1):

            locus_hits = group.loc[indices]
            rows.append({
                "seqid": seqid,
                "locus_id": f"{seqid}__locus_{locus_number}",
                "start": locus_hits["start"].min(),
                "end": locus_hits["end"].max(),
                "seq_len": locus_hits["seq_len"].iloc[0],
                "sig_keys": tuple(sorted(locus_hits["sig_key"].unique())),
                "ipr_accs": tuple(sorted(locus_hits["ipr_acc"].dropna().unique())),
                "descriptions": tuple(sorted(locus_hits["sig_desc"].dropna().unique())),
                "n_hits": len(locus_hits)})

    loci = pd.DataFrame(rows)

    return loci.sort_values(["seqid", "start", "end"])

def label_locus(sig_keys, alias, label_priority):

    labels = sorted(set(alias.get(key, key) for key in sig_keys))

    if len(labels) == 1:

        return labels[0]

    for label in label_priority:

        if label in labels:

            return label

    return "+".join(labels)

def add_locus_labels(loci, alias, label_priority):

    loci = loci.copy()
    loci["label"] = [label_locus(row.sig_keys, alias, label_priority)
                     for row in loci.itertuples()]

    return loci

def u_token(length):

    if length < 20:

        return None

    if length < 75:

        return "U_short"

    if length < 200:

        return "U_medium"

    return "U_long"

def build_architectures(loci, include_u=False):

    rows = []

    for seqid, group in loci.groupby("seqid"):

        group = group.sort_values("start")
        seq_len = group["seq_len"].iloc[0]
        tokens = []
        position = 1

        for row in group.itertuples():

            gap = row.start - position
            token = u_token(gap)

            if token and include_u:

                tokens.append(token)

            tokens.append(row.label)
            position = row.end + 1

        tail = seq_len - position + 1
        token = u_token(tail)

        if token and include_u:

            tokens.append(token)

        rows.append({
            "seqid": seqid,
            "seq_len": seq_len,
            "architecture": " -- ".join(tokens),
            "n_loci": len(group),
            "tokens": tuple(tokens)})

    return pd.DataFrame(rows)

def get_signature_proteins(df, sig_key="PANTHER:PTHR30595"):

    return set(df.loc[df["sig_key"] == sig_key, "seqid"].unique())

def split_architectures_by_signature(architectures, signature_proteins, suffix=" > GLPR"):

    architectures = architectures.copy()
    architectures["base_architecture"] = architectures["architecture"]
    architectures["has_signature"] = architectures["seqid"].isin(signature_proteins)
    architectures.loc[architectures["has_signature"], "architecture"] = architectures.loc[architectures["has_signature"], "architecture"] + suffix

    return architectures

def get_architecture_counts(architectures):

    architecture_counts = (architectures.groupby("architecture")
                            .agg(n_proteins=("seqid", "nunique"),
                                 mean=("seq_len", "mean"),
                                 median=("seq_len", "median"),
                                 stdev=("seq_len", "std"),
                                 examples=("seqid", lambda x: ", ".join(list(x)[:5])),)
                            .reset_index()
                            .sort_values("n_proteins", ascending=False))

    return architecture_counts

def export_architecture_fastas(architectures, fasta_path, output_dir):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seq_to_arch = dict(zip(architectures["seqid"], architectures["architecture"]))
    arch_to_path = {arch: output_dir / f"{i:03d}_{_architecture_filename(arch)}.faa"
                    for i, arch in enumerate(architectures["architecture"].unique(), start=1)}
    handles = {arch: path.open("w") for arch, path in arch_to_path.items()}
    counts = Counter()

    with open(fasta_path) as fasta:

        output = None
        architecture = None

        for line in fasta:

            if line.startswith(">"):

                seqid = line[1:].split()[0]
                architecture = seq_to_arch.get(seqid)
                output = handles.get(architecture)

                if output:

                    counts[architecture] += 1

            if output:

                output.write(line)

    for handle in handles.values():

        handle.close()

    return pd.DataFrame({
        "architecture": list(arch_to_path),
        "n_sequences": [counts[arch] for arch in arch_to_path],
        "path": [str(path) for path in arch_to_path.values()]})

def _architecture_filename(architecture, max_len=120):

    name = re.sub(r"[^A-Za-z0-9._-]+", "_", architecture).strip("._-")

    return name[:max_len] or "architecture"

def _validate_representative_clustering_configuration():

    if not 0 <= float(cfg.REPRESENTATIVE_CLUSTER_MIN_IDENTITY) <= 1:

        raise ValueError("REPRESENTATIVE_CLUSTER_MIN_IDENTITY must be between 0 and 1")

    if not 0 <= float(cfg.REPRESENTATIVE_CLUSTER_COVERAGE) <= 1:

        raise ValueError("REPRESENTATIVE_CLUSTER_COVERAGE must be between 0 and 1")

MMSEQS_ALIGNMENT_COLUMNS = [
    "query",
    "target",
    "fident",
    "ppos",
    "nident",
    "alnlen",
    "mismatch",
    "gapopen",
    "qlen",
    "tlen",
    "qcov",
    "tcov"]

REPRESENTATIVE_ALIGNMENT_FIELDS = [
    "protein_accession",
    "representative_accession",
    "cluster_id",
    "cluster_size",
    "is_representative",
    "pct_identity",
    "pct_positives",
    "n_identical",
    "alignment_length",
    "mismatches",
    "gap_opens",
    "protein_length",
    "representative_length",
    "protein_coverage",
    "representative_coverage"]

class MmseqsClustering:

    def __init__(self, sequences):

        self.sequences = sequences

    def __enter__(self):

        self._workspace = tempfile.TemporaryDirectory(prefix="step4_representatives_")
        root = Path(self._workspace.name)
        self.temporary_path = root / "tmp"
        self.input_path = root / "proteins.faa"
        self.sequence_db = root / "sequenceDB"
        self.cluster_db = root / "clusterDB"
        self.alignment_db = root / "alignmentDB"
        self.cluster_tsv = root / "clusters.tsv"
        self.alignment_tsv = root / "alignments.tsv"
        with self.input_path.open("w", encoding="ascii") as handle:

            for accession in sorted(self.sequences): handle.write(f">{accession}\n{self.sequences[accession]}\n")

        self._mmseqs("createdb", self.input_path, self.sequence_db, "--dbtype", "1", "--shuffle", "0",)
        self._mmseqs("cluster", self.sequence_db, self.cluster_db, self.temporary_path,
                     "--min-seq-id", cfg.REPRESENTATIVE_CLUSTER_MIN_IDENTITY,
                     "-c", cfg.REPRESENTATIVE_CLUSTER_COVERAGE,
                     "--cov-mode", "0", "--cluster-mode", "0", "--single-step-clustering", "1",)

        return self

    def __exit__(self, *exception):

        self._workspace.cleanup()

    def _mmseqs(self, *arguments):

        command = [str(cfg.MMSEQS_BIN), *(str(argument) for argument in arguments),
                   "--threads", str(cfg.MMSEQS_THREADS)]
        result = subprocess.run(command, check=False, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)

        if result.returncode != 0:

            detail = result.stderr.strip() or result.stdout.strip()

            raise RuntimeError(f"MMseqs {arguments[0]} failed ({result.returncode}): {detail[-2000:]}")

    def membership(self):

        self._mmseqs("createtsv", self.sequence_db, self.sequence_db, self.cluster_db, self.cluster_tsv,)

        return pd.read_csv(self.cluster_tsv, sep="\t",
                           names=["representative_accession", "protein_accession"])

    def alignments(self):

        self._mmseqs("align", self.sequence_db, self.sequence_db, self.cluster_db, self.alignment_db,
                     "-a", "--alignment-mode", "3", "-e", "10000",
                     "--min-seq-id", "0.0", "-c", "0.0", "--cov-mode", "0",)
        self._mmseqs("convertalis", self.sequence_db, self.sequence_db, self.alignment_db,
                     self.alignment_tsv, "--format-output", ",".join(MMSEQS_ALIGNMENT_COLUMNS),)
        alignments = pd.read_csv(self.alignment_tsv, sep="\t", names=MMSEQS_ALIGNMENT_COLUMNS)

        if alignments[["fident", "ppos"]].to_numpy().max() > 1:

            raise ValueError("MMseqs reported fident or ppos above 1; this build does not return fractions")

        return alignments

def _require_roster_coverage(sequences, table, source_name):

    covered = set(table["protein_accession"])
    missing = sorted(set(sequences) - covered)
    if missing:

        raise ValueError(f"{source_name} omitted architecture proteins: " + ", ".join(missing[:10]))

    stale = sorted(covered - set(sequences))
    if stale:

        raise ValueError(f"{source_name} names proteins no longer on the roster: " + ", ".join(stale[:10]))

def _representative_cluster_tables(sequences, membership):

    unknown = (set(membership["protein_accession"]) | set(membership["representative_accession"])) - set(sequences)
    if unknown:

        raise ValueError("MMseqs representative clustering contains unknown IDs: " + ", ".join(sorted(unknown)[:10]))

    repeated = membership["protein_accession"][membership["protein_accession"].duplicated()]
    if not repeated.empty:

        raise ValueError(f"MMseqs assigned {repeated.iloc[0]} to multiple representative clusters")

    _require_roster_coverage(sequences, membership, "MMseqs representative clustering")

    cluster_rows = []
    representative_rows = []
    for representative, cluster in membership.groupby("representative_accession", sort=True):

        ordered = tuple(sorted(cluster["protein_accession"]))
        if representative not in ordered:

            raise ValueError(f"MMseqs representative {representative} is absent from its cluster")

        digest = hashlib.sha256("\n".join(ordered).encode("utf-8")).hexdigest()
        cluster_id = f"cluster_{digest[:20]}"
        representative_rows.append({
            "cluster_id": cluster_id,
            "representative_accession": representative,
            "cluster_size": len(ordered)})

        for accession in ordered:

            cluster_rows.append({
                "protein_accession": accession,
                "cluster_id": cluster_id,
                "representative_accession": representative,
                "cluster_size": len(ordered)})

    cluster_rows.sort(key=lambda row: str(row["protein_accession"]))

    return cluster_rows, representative_rows

def _representative_alignments(alignments, cluster_rows):

    clusters = pd.DataFrame(cluster_rows, columns=REPRESENTATIVE_CLUSTER_FIELDS)
    aligned = alignments.rename(columns={"query": "representative_accession", "target": "protein_accession"})
    pairs = clusters.merge(aligned, on=["representative_accession", "protein_accession"],
                           how="left", validate="one_to_one")

    if pairs["alnlen"].isna().any():

        raise ValueError(f"MMseqs did not align {int(pairs['alnlen'].isna().sum()):,} of its own clustered pairs")

    pairs = pairs.assign(is_representative=pairs["protein_accession"] == pairs["representative_accession"],
                         pct_identity=100.0*pairs["fident"],
                         pct_positives=100.0*pairs["ppos"],
                         n_identical=pairs["nident"],
                         alignment_length=pairs["alnlen"],
                         mismatches=pairs["mismatch"],
                         gap_opens=pairs["gapopen"],
                         protein_length=pairs["tlen"],
                         representative_length=pairs["qlen"],
                         protein_coverage=pairs["tcov"],
                         representative_coverage=pairs["qcov"])

    return pairs[REPRESENTATIVE_ALIGNMENT_FIELDS]

def _cluster_similarity(pairs, representative_rows):

    members = pairs[~pairs["is_representative"]]
    summary = (members.groupby("cluster_id", sort=False)
               .agg(min_pct_identity=("pct_identity", "min"),
                    median_pct_identity=("pct_identity", "median"),
                    mean_pct_identity=("pct_identity", "mean"),
                    max_pct_identity=("pct_identity", "max"),
                    min_pct_positives=("pct_positives", "min"),
                    median_pct_positives=("pct_positives", "median"),
                    min_protein_coverage=("protein_coverage", "min"))
               .reset_index())
    clusters = pd.DataFrame(representative_rows, columns=REPRESENTATIVE_FIELDS)

    return clusters.merge(summary, on="cluster_id", how="left")

def _require(path, producer):

    if not path.exists():

        raise FileNotFoundError(f"Missing input: {path}. Run `{producer}` first.")

def _filtered_interpro_tables():

    _require(cfg.INTERPRO_TSV_PATH,
             "python -m src.stage02_domains.pipeline step1_interproscan")
    interpro = read_interpro_tsv(cfg.INTERPRO_TSV_PATH)
    interpro = interpro[~interpro["analysis"].isin(cfg.EXCLUDE_ANALYSES)].copy()
    interpro = interpro[~interpro["sig_acc"].isin(cfg.EXCLUDE_SIGNATURES)].copy()
    architecture_hits = interpro[interpro["analysis"] != "PANTHER"].copy()

    return interpro, architecture_hits

def fasta_ids(path):

    with open(path) as handle:

        return {line[1:].split()[0] for line in handle if line.startswith(">")}

def _require_tsv_matches_input(tsv_path, fasta_path):

    if fasta_path.stat().st_mtime > tsv_path.stat().st_mtime:

        raise ValueError(
            f"{fasta_path} is newer than {tsv_path}. Rerun with --force.")

    orphans = sorted(set(pd.read_csv(tsv_path, sep="\t", usecols=[0], names=["seqid"])["seqid"])
                     - fasta_ids(fasta_path))

    if orphans:

        raise ValueError(
            f"{tsv_path} annotates {len(orphans):,} proteins absent from {fasta_path} "
            f"(for example {orphans[0]}). Rerun with --force.")

def step1_interproscan(force=False):

    if cfg.INTERPRO_TSV_PATH.exists():

        if not force:

            _require_tsv_matches_input(cfg.INTERPRO_TSV_PATH, cfg.INTERPROSCAN_INPUT_FASTA)
            print(f"Stage 02 InterProScan: using existing {cfg.INTERPRO_TSV_PATH}")

            return

        cfg.INTERPRO_TSV_PATH.unlink()

    cfg.INTERPROSCAN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg.INTERPROSCAN_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        "docker",
        "run",
        "--rm",
        f"--memory={cfg.INTERPROSCAN_MEMORY}",
        f"--cpus={cfg.INTERPROSCAN_CPUS}",
        "-v",
        f"{cfg.INTERPROSCAN_DB_PATH}:/opt/interproscan/data",
        "-v",
        f"{cfg.INTERPROSCAN_WORK_DIR}:/data",
        cfg.INTERPROSCAN_IMAGE,
        "--input",
        f"/data/input/{cfg.INTERPROSCAN_INPUT_NAME}",
        "--output-dir",
        "/data/output",
        "--tempdir",
        "/data/temp",
        "--cpu",
        str(cfg.INTERPROSCAN_CPUS)]

    print(f"Stage 02 InterProScan: input={cfg.INTERPROSCAN_INPUT_FASTA}; output={cfg.INTERPRO_TSV_PATH}")

    subprocess.run(command, check=True)

def step2_build_equivalences():

    _, architecture_hits = _filtered_interpro_tables()
    pairs = overlapping_signature_pairs(architecture_hits)
    pair_summary = summarize_pairs(architecture_hits, pairs)
    candidates = candidate_equivalences(pair_summary)
    cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(cfg.CANDIDATE_DOMAIN_EQUIVALENCES_PATH, sep="\t", index=False)

    print(f"Stage 02 equivalence candidates: {len(candidates):,} rows")
    print(cfg.CANDIDATE_DOMAIN_EQUIVALENCES_PATH)

def step3_build_domains():

    interpro, architecture_hits = _filtered_interpro_tables()
    loci = consensus_loci(architecture_hits)
    labeled_loci = add_locus_labels(loci, cfg.ALIAS, cfg.LABEL_PRIORITY)
    glpr_proteins = get_signature_proteins(interpro, cfg.GLPR_SIG_KEY)
    architectures = build_architectures(labeled_loci)
    architectures = split_architectures_by_signature(architectures, glpr_proteins,
                                                      suffix=" > GLPR")
    architecture_counts = get_architecture_counts(architectures)
    cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)

    for frame, path in [(interpro, cfg.DF_INTERPRO_PATH),
                        (architecture_hits, cfg.DF_ARCHITECTURE_PATH),
                        (loci, cfg.DF_LOCI_PATH),
                        (labeled_loci, cfg.DF_LOCI_LABELED_PATH),
                        (architectures, cfg.DF_ARCHITECTURES_PATH),
                        (architecture_counts, cfg.DF_ARCHITECTURE_COUNTS_PATH)]:

        frame.to_pickle(path)

    print(f"Stage 02 domains: {len(interpro):,} filtered hits; "
          f"{len(labeled_loci):,} loci; {len(architectures):,} proteins; "
          f"{len(architecture_counts):,} architectures")
    print(cfg.OUT_DIR)

def _write_clustering_bundle(destination, sequences, membership, cluster_rows, representative_rows, pairs):

    destination.mkdir(parents=True, exist_ok=True)
    membership.to_csv(destination / cfg.MMSEQS_CLUSTER_TSV_PATH.name, sep="\t", index=False)
    pd.DataFrame(cluster_rows, columns=REPRESENTATIVE_CLUSTER_FIELDS).to_csv(
        destination / cfg.REPRESENTATIVE_CLUSTERS_PATH.name, sep="\t", index=False)
    pd.DataFrame(representative_rows, columns=REPRESENTATIVE_FIELDS).to_csv(
        destination / cfg.REPRESENTATIVE_LIST_PATH.name, sep="\t", index=False)
    pairs.to_csv(destination / cfg.REPRESENTATIVE_ALIGNMENTS_PATH.name, index=False)
    _cluster_similarity(pairs, representative_rows).to_csv(
        destination / cfg.CLUSTER_SIMILARITY_PATH.name, index=False)

    with (destination / cfg.REPRESENTATIVE_FASTA_PATH.name).open("w", encoding="ascii") as handle:

        for row in representative_rows:

            accession = row["representative_accession"]
            handle.write(f">{accession} cluster_id={row['cluster_id']}\n")
            handle.write(f"{sequences[accession]}\n")

def _verify_clustering_bundle(destination):

    missing = [path.name for path in cfg.REPRESENTATIVE_CLUSTERING_BUNDLE
               if not (destination / path.name).exists()]

    if missing:

        raise RuntimeError(f"Clustering bundle in {destination} is incomplete: {', '.join(missing)}")

    listed = pd.read_csv(destination / cfg.REPRESENTATIVE_LIST_PATH.name, sep="\t")
    membership = pd.read_csv(destination / cfg.MMSEQS_CLUSTER_TSV_PATH.name, sep="\t")
    exported = {record.id for record in SeqIO.parse(destination / cfg.REPRESENTATIVE_FASTA_PATH.name, "fasta")}
    declared = set(listed["representative_accession"])
    clustered = set(membership["representative_accession"])

    if exported != declared or declared != clustered:

        raise RuntimeError(f"Clustering bundle in {destination} disagrees about its representatives: "
                           f"{len(exported):,} in the FASTA, {len(declared):,} in the representative "
                           f"list, {len(clustered):,} in the membership table.")

def _existing_clustering_bundle():

    missing = [path.name for path in cfg.REPRESENTATIVE_CLUSTERING_BUNDLE if not path.exists()]

    if missing and len(missing) < len(cfg.REPRESENTATIVE_CLUSTERING_BUNDLE):

        raise RuntimeError(f"Stage 02 representative clustering is incomplete: {', '.join(missing)} "
                           f"missing from {cfg.OUT_DIR}. One clustering run writes all six artifacts, "
                           f"so a partial bundle cannot be reused. Restore the missing files, or "
                           f"rebuild with: python -m src.stage02_domains.pipeline "
                           f"step4_cluster_representatives --force")

    return not missing

def _require_clustering_bundle_matches_roster(sequences):

    membership = pd.read_csv(cfg.MMSEQS_CLUSTER_TSV_PATH, sep="\t")
    _require_roster_coverage(sequences, membership, cfg.MMSEQS_CLUSTER_TSV_PATH.name)

    for record in SeqIO.parse(cfg.REPRESENTATIVE_FASTA_PATH, "fasta"):

        if str(record.seq) != sequences[record.id]:

            raise RuntimeError(f"Stage 02 representative clustering bundle in {cfg.OUT_DIR} disagrees "
                               f"with the roster: {record.id}'s sequence in "
                               f"{cfg.REPRESENTATIVE_FASTA_PATH.name} no longer matches roster.sequences(). "
                               f"Rerun with --force to recluster.")

def step4_cluster_representatives(force=False, into=None):

    destination = Path(into) if into else cfg.OUT_DIR
    publishing = destination == cfg.OUT_DIR
    sequences = roster.sequences()

    if publishing and _existing_clustering_bundle() and not force:

        _require_clustering_bundle_matches_roster(sequences)

        print(f"Stage 02 representative clustering: using existing bundle in {cfg.OUT_DIR}")
        print(f"{cfg.REPRESENTATIVE_FASTA_PATH.name} is what AlphaFold3 and DPAM were run on "
              f"off-machine; rerun with --force only to change the representative set.")

        return

    if publishing and cfg.REPRESENTATIVE_FASTA_PATH.exists():

        print(f"Stage 02 representative clustering: --force rewrites {cfg.REPRESENTATIVE_FASTA_PATH}. "
              f"If the representative set changes, the frozen AF3+DPAM annotations no longer cover it "
              f"and step 6 will refuse them in dpam.require_assignments_match_selection.")

    _validate_representative_clustering_configuration()

    with MmseqsClustering(sequences) as clustering:

        membership = clustering.membership()
        alignments = clustering.alignments()

    cluster_rows, representative_rows = _representative_cluster_tables(sequences, membership)
    pairs = _representative_alignments(alignments, cluster_rows)
    _write_clustering_bundle(destination, sequences, membership, cluster_rows, representative_rows, pairs)
    _verify_clustering_bundle(destination)
    members = pairs[~pairs["is_representative"]]

    print(f"Stage 02 representative clustering: {len(representative_rows):,} representatives "
          f"for {len(cluster_rows):,} proteins")
    print(f"Member-to-representative identity: median {members['pct_identity'].median():.1f}%, "
          f"minimum {members['pct_identity'].min():.1f}%; positives median {members['pct_positives'].median():.1f}%")
    for path in cfg.REPRESENTATIVE_CLUSTERING_BUNDLE: print(destination / path.name)

    if not publishing:

        print(f"This is a separate clustering generation. {cfg.OUT_DIR} is untouched, so the "
              f"previous generation and the frozen AF3+DPAM annotations keyed to it remain in "
              f"place. Compare the two before publishing anything.")

def _anchor_records(sequences):

    return [(entry["protein_acc"], entry["Name"], sequences[entry["protein_acc"]])
            for entry in main_cfg.the_nine.values()]

def _require_anchor_fasta_matches_dpam(records):

    observed = (dpam.read_dpam_domains(cfg.DPAM_ANCHOR_DOMAINS_PATH)[["protein_accession", "sequence_length"]]
                .drop_duplicates())
    ambiguous = sorted(observed.loc[observed["protein_accession"].duplicated(), "protein_accession"])

    if ambiguous:

        raise RuntimeError(f"{cfg.DPAM_ANCHOR_DOMAINS_PATH.name} reports more than one sequence_length "
                           f"for {len(ambiguous):,} proteins ({', '.join(ambiguous[:5])}), so no single "
                           f"sequence can be the one the batch was run on.")

    reported = dict(zip(observed["protein_accession"], observed["sequence_length"]))
    written = {accession: len(sequence) for accession, _, sequence in records}
    absent = sorted(set(reported) - set(written))
    wrong = sorted(accession for accession, length in reported.items()
                   if accession in written and written[accession] != length)

    if absent or wrong:

        raise RuntimeError(f"{cfg.ANCHOR_FASTA_PATH.name} does not describe the proteins "
                           f"{cfg.DPAM_ANCHOR_DOMAINS_PATH.name} reports domains for: "
                           f"{len(absent):,} observed anchors are not in the_nine "
                           f"({', '.join(absent[:5])}); {len(wrong):,} are written at a different "
                           f"length ({', '.join(wrong[:5])}).")

def step4b_export_anchor_fasta():

    records = _anchor_records(roster.sequences())
    _require_anchor_fasta_matches_dpam(records)

    with cfg.ANCHOR_FASTA_PATH.open("w", encoding="ascii") as handle:

        for accession, name, sequence in records:

            handle.write(f">{accession} anchor={name}\n")
            handle.write(f"{sequence}\n")

    print(f"Stage 02 anchor FASTA: {len(records):,} experimental anchors, "
          f"lengths agreeing with {cfg.DPAM_ANCHOR_DOMAINS_PATH.name}")
    print(cfg.ANCHOR_FASTA_PATH)

def step5_export_fastas():

    _require(cfg.DF_ARCHITECTURES_PATH,
             "python -m src.stage02_domains.pipeline step3_build_domains")
    architectures = pd.read_pickle(cfg.DF_ARCHITECTURES_PATH)
    manifest = export_architecture_fastas(architectures, cfg.COMBINED_FASTA_PATH,
                                          cfg.ARCHITECTURE_FASTA_DIR)
    manifest.to_csv(cfg.ARCHITECTURE_FASTA_MANIFEST_PATH, sep="\t", index=False)

    print(f"Stage 02 architecture FASTAs: {len(manifest):,} files")
    print(cfg.ARCHITECTURE_FASTA_MANIFEST_PATH)

def step6_build_dpam_architectures():

    _require(cfg.REPRESENTATIVE_CLUSTERS_PATH,
             "python -m src.stage02_domains.pipeline step4_cluster_representatives")
    batches = dpam.Batch.declared()
    by_batch = {batch.name: batch.read() for batch in batches}
    sequences = roster.sequences()
    manifest = dpam.batch_manifest(batches, by_batch, sequences)
    domains = pd.concat(by_batch.values(), ignore_index=True)
    observations = dpam.architecture_observations(domains)
    selected = dpam.select_observations(observations)
    clusters = pd.read_csv(cfg.REPRESENTATIVE_CLUSTERS_PATH, sep="\t")
    _require_roster_coverage(sequences, clusters, cfg.REPRESENTATIVE_CLUSTERS_PATH.name)
    proteins = dpam.assign_to_proteins(selected, clusters)
    representatives = selected[selected["protein_accession"].isin(set(clusters["representative_accession"]))]
    anchors = dpam.anchor_validation(observations, selected, clusters)
    cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(cfg.DPAM_BATCH_MANIFEST_PATH, index=False)
    domains.to_csv(cfg.DPAM_DOMAINS_TABLE_PATH, index=False)
    observations.to_csv(cfg.DPAM_ARCHITECTURE_OBSERVATIONS_PATH, index=False)
    representatives.drop(columns="batch").to_csv(cfg.DPAM_ARCHITECTURES_PATH, index=False)
    proteins.to_csv(cfg.DPAM_PROTEIN_ARCHITECTURES_PATH, index=False)
    anchors.to_csv(cfg.RESULT_ANCHOR_VALIDATION_PATH, index=False)
    classes = proteins["hth_class"].value_counts().reindex(cfg.DPAM_HTH_CLASSES)
    sources = proteins["architecture_source"].value_counts()
    disagreements = anchors[~anchors["agrees_architecture"] | ~anchors["agrees_n_hth"]]

    print(f"Stage 02 DPAM batches: " + "; ".join(
        f"{row.batch} {row.proteins:,} proteins, {row.domains:,} domains, sequences {row.sequence_identity}"
        for row in manifest.itertuples()))
    print(f"Assignments over {len(proteins):,} study proteins: " + ", ".join(
        f"{name} {count:,}" for name, count in sources.items()))
    print("HTH classes: " + ", ".join(f"{name} {count:,}" for name, count in classes.items()))
    print(f"Anchor checks: {len(anchors):,} anchors, "
          f"{int(anchors['agrees_architecture'].sum()):,} agree on architecture, "
          f"{int(anchors['agrees_hth_families'].sum()):,} agree on family signature")

    for path in (cfg.DPAM_BATCH_MANIFEST_PATH, cfg.DPAM_DOMAINS_TABLE_PATH,
                 cfg.DPAM_ARCHITECTURE_OBSERVATIONS_PATH, cfg.DPAM_ARCHITECTURES_PATH,
                 cfg.DPAM_PROTEIN_ARCHITECTURES_PATH, cfg.RESULT_ANCHOR_VALIDATION_PATH): print(path)

    if len(disagreements):

        print(f"{len(disagreements):,} anchor(s) disagree with their reference; see "
              f"{cfg.RESULT_ANCHOR_VALIDATION_PATH.name}. This is a recorded observation, not a failure.")

    unresolved = dpam.unresolved_assignments(proteins)

    if len(unresolved):

        preview = ", ".join(unresolved["protein_accession"].head(10))

        raise RuntimeError(f"{len(unresolved):,} study proteins have no structural assignment: no batch "
                           f"observed them and none observed their cluster representative ({preview}). "
                           f"They are written to {cfg.DPAM_PROTEIN_ARCHITECTURES_PATH.name} with "
                           f"architecture_source=unresolved and empty annotation columns, because an "
                           f"unmeasured protein is not a protein without HTH domains. Supply an AF3+DPAM "
                           f"batch covering them, or remove them from the study roster.")

def step7_build_results():

    for path, producer in [(cfg.DF_ARCHITECTURES_PATH,
                            "python -m src.stage02_domains.pipeline step3_build_domains"),
                           (cfg.DPAM_DOMAINS_TABLE_PATH,
                            "python -m src.stage02_domains.pipeline step6_build_dpam_architectures"),
                           (cfg.DPAM_ARCHITECTURE_OBSERVATIONS_PATH,
                            "python -m src.stage02_domains.pipeline step6_build_dpam_architectures"),
                           (cfg.DPAM_PROTEIN_ARCHITECTURES_PATH,
                            "python -m src.stage02_domains.pipeline step6_build_dpam_architectures"),
                           (cfg.REPRESENTATIVE_CLUSTERS_PATH,
                            "python -m src.stage02_domains.pipeline step4_cluster_representatives")]:

        _require(path, producer)

    interpro = pd.read_pickle(cfg.DF_ARCHITECTURES_PATH)
    domains = pd.read_csv(cfg.DPAM_DOMAINS_TABLE_PATH, keep_default_na=False)
    observations = pd.read_csv(cfg.DPAM_ARCHITECTURE_OBSERVATIONS_PATH, keep_default_na=False)
    proteins = pd.read_csv(cfg.DPAM_PROTEIN_ARCHITECTURES_PATH, keep_default_na=False)
    clusters = pd.read_csv(cfg.REPRESENTATIVE_CLUSTERS_PATH, sep="\t")
    selected = dpam.select_observations(observations)
    dpam.require_assignments_match_selection(proteins, selected)
    anchors = dpam.anchor_validation(observations, selected, clusters)
    interpro_comparison = dpam.interpro_vs_dpam_hth(interpro, selected, clusters)
    sensitivity = dpam.judge_sensitivity(domains, observations, selected, proteins, clusters)
    census = dpam.family_census(domains, selected)
    cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)
    anchors.to_csv(cfg.RESULT_ANCHOR_VALIDATION_PATH, index=False)
    interpro_comparison.to_csv(cfg.RESULT_INTERPRO_VS_DPAM_HTH_PATH, sep="\t", index=False)
    sensitivity.to_csv(cfg.RESULT_DPAM_JUDGE_SENSITIVITY_PATH, sep="\t", index=False)
    census.to_csv(cfg.RESULT_DPAM_FAMILY_CENSUS_PATH, sep="\t", index=False)

    print(f"Stage 02 results: {int(interpro_comparison['representative_count'].sum()):,} comparable "
          f"representatives; {int(sensitivity.loc[sensitivity['grain'] == 'protein', 'entities_changing_hth_class'].iloc[0]):,} "
          f"proteins change class under the rejected weak-HTH policy")
    for path in (cfg.RESULT_ANCHOR_VALIDATION_PATH, cfg.RESULT_INTERPRO_VS_DPAM_HTH_PATH,
                 cfg.RESULT_DPAM_JUDGE_SENSITIVITY_PATH, cfg.RESULT_DPAM_FAMILY_CENSUS_PATH): print(path)

def run(force=False):

    step1_interproscan(force=force)
    step3_build_domains()
    step4_cluster_representatives()
    step4b_export_anchor_fasta()
    step6_build_dpam_architectures()
    step7_build_results()

def main(argv=None):

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("step1_interproscan", help="Run Docker InterProScan.")
    p1.add_argument("--force", action="store_true")
    p1.set_defaults(func=step1_interproscan)

    p2 = sub.add_parser("step2_build_equivalences", help="Write optional equivalence candidates.")
    p2.set_defaults(func=step2_build_equivalences)

    p3 = sub.add_parser("step3_build_domains", help="Build the domain-analysis artifacts.")
    p3.set_defaults(func=step3_build_domains)

    p4 = sub.add_parser("step4_cluster_representatives",
                        help="Cluster architecture proteins, export representatives, measure clusters.")
    p4.add_argument("--force", action="store_true",
                    help="Recluster and overwrite the existing bundle, including the AF3+DPAM input FASTA.")
    p4.add_argument("--into", metavar="DIR",
                    help="Write a separate clustering generation into DIR, leaving the published one alone.")
    p4.set_defaults(func=step4_cluster_representatives)

    p4b = sub.add_parser("step4b_export_anchor_fasta",
                         help="Write the AF3+DPAM input FASTA for the ten experimental anchors.")
    p4b.set_defaults(func=step4b_export_anchor_fasta)

    p5 = sub.add_parser("step5_export_fastas", help="Export optional architecture FASTAs.")
    p5.set_defaults(func=step5_export_fastas)

    p6 = sub.add_parser("step6_build_dpam_architectures",
                        help="Build AF3+DPAM architectures and project them onto every protein.")
    p6.set_defaults(func=step6_build_dpam_architectures)

    p7 = sub.add_parser("step7_build_results",
                        help="Write Stage 02 AF3+DPAM comparison and sensitivity tables.")
    p7.set_defaults(func=step7_build_results)

    p_run = sub.add_parser("run", help="Run the required Stage 02 pipeline.")
    p_run.add_argument("--force", action="store_true",
                       help="Rerun InterProScan. Never reclusters; use step4_cluster_representatives --force.")
    p_run.set_defaults(func=run)

    args = parser.parse_args(argv)

    keywords = {name: value for name, value in vars(args).items() if name not in ("command", "func")}
    args.func(**keywords)

if __name__ == "__main__":

    main()
