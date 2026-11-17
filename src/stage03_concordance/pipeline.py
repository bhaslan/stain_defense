import argparse
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.stage02_domains import config_domains as domain_cfg
from src.stage03_concordance import archive
from src.stage03_concordance import config_concordance as cfg

@dataclass
class EmbeddingConfig:

    path_h5: str
    embedding_space: str = "raw"

@dataclass
class EvidenceConfig:

    hth_rich: float = 0.70
    hth_poor: float = 0.30

@dataclass
class EmbeddingData:

    ids: list
    embeddings: np.ndarray
    lengths: np.ndarray
    metadata: dict

def _decode_h5_strings(values):

    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in values]

def _h5_attrs_to_dict(attrs):

    out = {}

    for key, value in attrs.items():

        if hasattr(value, "item"):

            value = value.item()

        out[key] = value

    return out

def _l2_normalize(x):

    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

def load_esmc_embeddings(path_h5, embedding_space="raw"):

    with h5py.File(path_h5, "r") as f:

        ids = _decode_h5_strings(f["ids"][:])
        lengths = f["lengths"][:]
        embeddings = f["embeddings"][:].astype(np.float32)
        metadata = _h5_attrs_to_dict(f.attrs)

    if not metadata.get("completed", False):

        raise ValueError(
            f"{path_h5} holds {len(ids):,} proteins derived from an incomplete "
            "per-position archive and must not define the concordance universe. "
            "Finish `python -m src.stage03_concordance.pipeline "
            "step1_archive_embeddings`, which rederives this cache.")

    if embedding_space == "raw":

        embeddings = _l2_normalize(embeddings)
    elif embedding_space == "centered":

        embeddings = embeddings - embeddings.mean(axis=0, keepdims=True)
        embeddings = _l2_normalize(embeddings)
    else:

        raise ValueError("embedding_space must be 'raw' or 'centered'.")

    metadata["embedding_space"] = embedding_space
    metadata["path_h5"] = str(path_h5)

    return EmbeddingData(ids=ids, embeddings=embeddings, lengths=lengths, metadata=metadata)

def build_concordance_table(architectures, loci_labeled, df_interpro,
                            glpr_sig_key="PANTHER:PTHR30595"):

    rows = architectures.copy()
    rows["protein_id"] = rows["seqid"]
    rows["length"] = rows["seq_len"]

    if "base_architecture" not in rows:

        rows["base_architecture"] = rows["architecture"]

    glpr_proteins = set(df_interpro.loc[df_interpro["sig_key"] == glpr_sig_key, "seqid"])
    rows["has_GLPR"] = rows["seqid"].isin(glpr_proteins)

    hth_loci = loci_labeled[loci_labeled["label"] == "HTH"].copy()

    if len(hth_loci):

        hth_loci["coordinate"] = (hth_loci["start"].astype(str) + "-"
                                  + hth_loci["end"].astype(str))
        hth_summary = (hth_loci
                       .sort_values(["seqid", "start", "end"])
                       .groupby("seqid")
                       .agg(local_hth_n_loci=("locus_id", "nunique"),
                            HTH_coordinates=("coordinate", lambda x: ";".join(x)),
                            HTH_signature_source=("sig_keys", _join_signature_keys),
                            HTH_start=("start", "min"),
                            HTH_end=("end", "max"),)
                       .reset_index())
    else:

        hth_summary = pd.DataFrame(columns=[
            "seqid",
            "local_hth_n_loci",
            "HTH_coordinates",
            "HTH_signature_source",
            "HTH_start",
            "HTH_end"])

    rows = rows.merge(hth_summary, on="seqid", how="left")
    rows["local_hth_n_loci"] = rows["local_hth_n_loci"].fillna(0).astype(int)
    rows["has_local_hth"] = rows["local_hth_n_loci"] > 0
    rows["local_HTH_status"] = np.where(rows["has_local_hth"],
                                         "HTH_positive", "HTH_negative")
    rows["HTH_coordinates"] = rows["HTH_coordinates"].fillna("")
    rows["HTH_signature_source"] = rows["HTH_signature_source"].fillna("")

    columns = [
        "protein_id",
        "seqid",
        "length",
        "architecture",
        "base_architecture",
        "has_GLPR",
        "local_HTH_status",
        "has_local_hth",
        "local_hth_n_loci",
        "HTH_signature_source",
        "HTH_coordinates",
        "HTH_start",
        "HTH_end"]
    extra_columns = [c for c in rows.columns if c not in columns]

    return rows[columns + extra_columns]

def _join_signature_keys(values):

    keys = []

    for sig_keys in values:

        keys.extend(str(key) for key in sig_keys)

    return ";".join(sorted(set(keys)))

def align_embeddings_to_concordance(embedding_data, concordance, id_col="seqid"):

    emb_index = pd.DataFrame({
        id_col: embedding_data.ids,
        "_embedding_row": np.arange(len(embedding_data.ids)),
        "embedding_length": embedding_data.lengths})
    aligned = emb_index.merge(concordance, on=id_col, how="inner")
    aligned = aligned.sort_values("_embedding_row").reset_index(drop=True)
    embeddings = embedding_data.embeddings[aligned["_embedding_row"].to_numpy()]

    return aligned, embeddings

def compute_hth_neighbor_scores(embeddings, concordance, k=50):

    neighbor_idx, neighbor_dist, k_effective = _nearest_neighbor_arrays(embeddings, k)
    has_hth = concordance["has_local_hth"].to_numpy(dtype=bool)
    rows = []

    for idx_row, dist_row in zip(neighbor_idx, neighbor_dist):

        rows.append({
            "hth_neighbor_k": int(k),
            "neighbor_count": int(k_effective),
            "hth_neighbor_fraction": float(has_hth[idx_row].mean()),
            "mean_neighbor_cosine_distance": float(dist_row.mean())})

    return pd.concat([concordance.reset_index(drop=True), pd.DataFrame(rows)], axis=1)

def knn_neighbor_edges(embeddings, concordance, k=20, id_col="seqid"):

    neighbor_idx, neighbor_dist, k_effective = _nearest_neighbor_arrays(embeddings, k)
    seqids = concordance[id_col].to_numpy()
    n_query = len(seqids)

    return pd.DataFrame({
        "seqid": np.repeat(seqids, k_effective),
        "neighbor_rank": np.tile(np.arange(1, k_effective + 1, dtype=np.int32), n_query),
        "neighbor_seqid": seqids[neighbor_idx.reshape(-1)],
        "cosine_distance": neighbor_dist.reshape(-1).astype(np.float64)})

def write_knn_neighbor_edges(path_h5, concordance, path_out, k=20,
                             embedding_space="raw", id_col="seqid"):

    embedding_data = load_esmc_embeddings(path_h5, embedding_space)
    concordance_for_align = concordance.drop(columns=[
        c for c in ("_embedding_row", "embedding_length")
        if c in concordance.columns])
    aligned, embeddings = align_embeddings_to_concordance(embedding_data,
                                                           concordance_for_align,
                                                           id_col=id_col)
    edges = knn_neighbor_edges(embeddings, aligned, k=k, id_col=id_col)
    path_out = Path(path_out)
    path_out.parent.mkdir(parents=True, exist_ok=True)
    edges.to_csv(path_out, sep="\t", index=False)

    return edges

def length_aware_concordance_sensitivity(embeddings, concordance, k_values,
                                         n_bins=4, n_permutations=1000,
                                         random_state=0):

    rows = concordance.reset_index(drop=True).copy()
    labels = rows["has_local_hth"].to_numpy(dtype=bool)
    lengths = rows["length"].to_numpy(dtype=float)
    bin_ids = _length_bin_ids(lengths, n_bins)
    groups = _length_sensitivity_groups(bin_ids)
    rng = np.random.default_rng(random_state)

    if isinstance(k_values, (int, np.integer)):

        k_values = [int(k_values)]

    out = []

    for k in k_values:

        neighbor_idx, _, k_effective = _nearest_neighbor_arrays(embeddings, int(k))
        scores = labels[neighbor_idx].mean(axis=1)
        null_deltas = {name: [] for name, _ in groups}

        for _ in range(n_permutations):

            permuted = labels.copy()

            for bin_id in np.unique(bin_ids):

                mask = bin_ids == bin_id
                permuted[mask] = rng.permutation(permuted[mask])

            permuted_scores = permuted[neighbor_idx].mean(axis=1)

            for name, mask in groups:

                null_deltas[name].append(_score_delta(permuted_scores, permuted, mask))

        for name, mask in groups:

            out.append(_length_sensitivity_row(name=name,
                                               mask=mask,
                                               labels=labels,
                                               lengths=lengths,
                                               scores=scores,
                                               neighbor_k=k_effective,
                                               n_permutations=n_permutations,
                                               null_deltas=np.asarray(null_deltas[name], dtype=float)))

    return pd.DataFrame(out)

def assign_evidence_tiers(concordance, config=None):

    if config is None:

        config = EvidenceConfig()

    rows = concordance.copy()
    neighbor_signal = _signal_from_score(rows["hth_neighbor_fraction"],
                                         config.hth_rich,
                                         config.hth_poor)

    rows["neighbor_hth_signal"] = neighbor_signal
    rows["esmc_hth_signal"] = neighbor_signal
    rows["evidence_tier"] = [
        _evidence_tier(local_hth, has_glpr, signal)
        for local_hth, has_glpr, signal
        in zip(rows["has_local_hth"], rows["has_GLPR"], neighbor_signal)]

    return rows

def _signal_from_score(scores, hth_rich, hth_poor):

    scores = pd.Series(scores)

    return np.where(scores >= hth_rich, "hth_rich",
                    np.where(scores <= hth_poor, "hth_poor", "mixed"))

def _evidence_tier(has_local_hth, has_glpr, esmc_signal):

    if has_local_hth and esmc_signal == "hth_rich":

        return "high_confidence_hth"

    if (not has_local_hth) and esmc_signal == "hth_poor":

        return "likely_non_hth"

    if (not has_local_hth) and esmc_signal == "hth_rich":

        return "potential_missed_or_degenerate_hth"

    if has_glpr and not has_local_hth:

        return "ambiguous_glpr_only"

    return "ambiguous"

def _nearest_neighbor_arrays(embeddings, k):

    k_effective = min(k, len(embeddings) - 1)
    nn = NearestNeighbors(n_neighbors=k_effective + 1, metric="cosine")
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)

    out_indices = []
    out_distances = []

    for i, (dist_row, idx_row) in enumerate(zip(distances, indices)):

        keep = idx_row != i
        out_indices.append(idx_row[keep][:k_effective])
        out_distances.append(dist_row[keep][:k_effective])

    return np.asarray(out_indices), np.asarray(out_distances), k_effective

def _length_bin_ids(lengths, n_bins):

    return pd.qcut(pd.Series(lengths), q=n_bins,
                   labels=False, duplicates="drop").to_numpy(dtype=int)

def _length_sensitivity_groups(bin_ids):

    groups = [("all", np.ones(len(bin_ids), dtype=bool))]

    for bin_id in sorted(np.unique(bin_ids)):

        groups.append((f"q{int(bin_id) + 1}", bin_ids == bin_id))

    return groups

def _score_delta(scores, labels, mask):

    local_labels = labels[mask]
    local_scores = scores[mask]

    if local_labels.sum() == 0 or (~local_labels).sum() == 0:

        return np.nan

    return float(local_scores[local_labels].mean() - local_scores[~local_labels].mean())

def _length_sensitivity_row(name, mask, labels, lengths, scores, neighbor_k,
                            n_permutations, null_deltas):

    observed_delta = _score_delta(scores, labels, mask)
    finite_null = null_deltas[np.isfinite(null_deltas)]

    if np.isfinite(observed_delta) and len(finite_null):

        p_value = (np.sum(np.abs(finite_null) >= abs(observed_delta)) + 1) / (len(finite_null) + 1)
        null_mean = float(finite_null.mean())
        null_sd = float(finite_null.std(ddof=1)) if len(finite_null) > 1 else 0.0
    else:

        p_value = np.nan
        null_mean = np.nan
        null_sd = np.nan

    pos = mask & labels
    neg = mask & ~labels

    return {
        "neighbor_k": int(neighbor_k),
        "length_bin": name,
        "length_min": int(np.nanmin(lengths[mask])),
        "length_max": int(np.nanmax(lengths[mask])),
        "n_proteins": int(mask.sum()),
        "n_hth_positive": int(pos.sum()),
        "n_hth_negative": int(neg.sum()),
        "mean_score_hth_positive": float(scores[pos].mean()) if pos.any() else np.nan,
        "mean_score_hth_negative": float(scores[neg].mean()) if neg.any() else np.nan,
        "delta": observed_delta,
        "null_delta_mean": null_mean,
        "null_delta_sd": null_sd,
        "permutation_p": p_value,
        "n_permutations": int(n_permutations)}

def _require(path, producer):

    if not path.exists():

        raise FileNotFoundError(f"Missing input: {path}. Run `{producer}` first.")

def _report_join_asymmetry(embedding_ids, architecture_ids):

    embedding_ids = set(embedding_ids)
    architecture_ids = set(architecture_ids)

    for dropped, label in ((sorted(embedding_ids - architecture_ids), "embedded without an architecture row"),
                           (sorted(architecture_ids - embedding_ids), "architecture rows without an embedding")):

        if dropped:

            shown = ", ".join(dropped[:20])
            more = "" if len(dropped) <= 20 else f", +{len(dropped) - 20:,} more"
            print(f"Stage 03 concordance: {len(dropped):,} {label}: {shown}{more}")

def _embedding_config():

    return EmbeddingConfig(path_h5=cfg.EMBEDDING_ARCHIVE_PATH,
                           embedding_space=cfg.EMBEDDING_SPACE)

def _evidence_config():

    return EvidenceConfig(hth_rich=cfg.HTH_RICH, hth_poor=cfg.HTH_POOR)

def step1_archive_embeddings():

    archive.run_archive()

def step2_build_concordance():

    for path in (domain_cfg.DF_ARCHITECTURES_PATH,
                 domain_cfg.DF_LOCI_LABELED_PATH,
                 domain_cfg.DF_INTERPRO_PATH):

        _require(path, "python -m src.stage02_domains.pipeline step3_build_domains")

    _require(cfg.EMBEDDING_ARCHIVE_PATH,
             "python -m src.stage03_concordance.pipeline step1_archive_embeddings")

    embedding_config = _embedding_config()
    embedding_data = load_esmc_embeddings(embedding_config.path_h5,
                                          embedding_config.embedding_space)
    concordance = build_concordance_table(architectures=pd.read_pickle(domain_cfg.DF_ARCHITECTURES_PATH),
                                          loci_labeled=pd.read_pickle(domain_cfg.DF_LOCI_LABELED_PATH),
                                          df_interpro=pd.read_pickle(domain_cfg.DF_INTERPRO_PATH),
                                          glpr_sig_key=cfg.GLPR_SIG_KEY)
    architecture_rows = len(concordance)
    _report_join_asymmetry(embedding_data.ids, concordance["seqid"])
    concordance, embeddings = align_embeddings_to_concordance(embedding_data,
                                                               concordance)
    concordance = compute_hth_neighbor_scores(embeddings, concordance,
                                              k=cfg.NEIGHBOR_K)
    concordance = assign_evidence_tiers(concordance, config=_evidence_config())
    cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)
    concordance.to_csv(cfg.CONCORDANCE_PATH, sep="\t", index=False)

    print(f"Stage 03 concordance: {len(concordance):,}/{architecture_rows:,} "
          f"proteins aligned; k={cfg.NEIGHBOR_K}")
    print(cfg.CONCORDANCE_PATH)

def step3_build_neighbors():

    _require(cfg.EMBEDDING_ARCHIVE_PATH,
             "python -m src.stage03_concordance.pipeline step1_archive_embeddings")
    _require(cfg.CONCORDANCE_PATH,
             "python -m src.stage03_concordance.pipeline step2_build_concordance")
    concordance = pd.read_csv(cfg.CONCORDANCE_PATH, sep="\t")
    edges = write_knn_neighbor_edges(cfg.EMBEDDING_ARCHIVE_PATH,
                                     concordance,
                                     cfg.KNN_EDGES_PATH,
                                     k=cfg.NEIGHBOR_K,
                                     embedding_space=cfg.EMBEDDING_SPACE)

    print(f"Stage 03 neighbors: {len(edges):,} edges for "
          f"{edges['seqid'].nunique():,} proteins")
    print(cfg.KNN_EDGES_PATH)

def step4_build_sensitivity():

    _require(cfg.EMBEDDING_ARCHIVE_PATH,
             "python -m src.stage03_concordance.pipeline step1_archive_embeddings")
    _require(cfg.CONCORDANCE_PATH,
             "python -m src.stage03_concordance.pipeline step2_build_concordance")

    embedding_config = _embedding_config()
    embedding_data = load_esmc_embeddings(embedding_config.path_h5,
                                          embedding_config.embedding_space)
    concordance = (pd.read_csv(cfg.CONCORDANCE_PATH, sep="\t")
                   .drop(columns=["_embedding_row", "embedding_length"],
                         errors="ignore"))
    concordance, embeddings = align_embeddings_to_concordance(embedding_data,
                                                               concordance)
    sensitivity = length_aware_concordance_sensitivity(embeddings,
                                                       concordance,
                                                       k_values=cfg.LENGTH_SENSITIVITY_K_VALUES,
                                                       n_bins=cfg.LENGTH_SENSITIVITY_BINS,
                                                       n_permutations=cfg.LENGTH_SENSITIVITY_PERMUTATIONS,
                                                       random_state=cfg.LENGTH_SENSITIVITY_RANDOM_STATE)
    cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)
    sensitivity.to_csv(cfg.LENGTH_SENSITIVITY_PATH, sep="\t", index=False)

    print(f"Stage 03 sensitivity: {len(sensitivity):,} result rows")
    print(cfg.LENGTH_SENSITIVITY_PATH)

def run():

    step1_archive_embeddings()
    step2_build_concordance()
    step3_build_neighbors()
    step4_build_sensitivity()

def main(argv=None):

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("step1_archive_embeddings",
                        help="Download/resume raw embeddings and derive pooled caches.")
    p1.set_defaults(func=step1_archive_embeddings)

    p2 = sub.add_parser("step2_build_concordance",
                        help="Build the concordance table.")
    p2.set_defaults(func=step2_build_concordance)

    p3 = sub.add_parser("step3_build_neighbors",
                        help="Write the kNN edge table.")
    p3.set_defaults(func=step3_build_neighbors)

    p4 = sub.add_parser("step4_build_sensitivity",
                        help="Run length-aware sensitivity analysis.")
    p4.set_defaults(func=step4_build_sensitivity)

    p_run = sub.add_parser("run", help="Run the complete Stage 03 pipeline.")
    p_run.set_defaults(func=run)

    args = parser.parse_args(argv)
    args.func()

if __name__ == "__main__":

    main()
