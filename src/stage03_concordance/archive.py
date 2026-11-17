import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from Bio import SeqIO
from dotenv import load_dotenv
from tqdm import tqdm

from src.stage03_concordance import config_concordance as cfg

load_dotenv()

PER_POSITION_ARCHIVE_FORMAT = "esmc_per_position_archive_v1"
ARCHIVE_MODEL_ID = "esmc-6b-2024-12"
ARCHIVE_LAYER_INDEX = 79
ARCHIVE_FORGE_URL = "https://biohub.ai"
ARCHIVE_TOKEN_ENV_VAR = "ESM_API_KEY"
ESMC_API_NUM_LAYERS = {
    "esmc-300m-2024-12": 30,
    "esmc-600m-2024-12": 36,
    "esmc-6b-2024-12": 80,
}

def read_fasta_biopython(path):
    ids, seqs = [], []
    for rec in SeqIO.parse(str(path), "fasta"):
        seq = str(rec.seq).replace(" ", "").replace("\n", "").upper()
        if seq:
            ids.append(rec.id)
            seqs.append(seq)
    return ids, seqs

def resolve_esmc_api_layer(model_id, layer, num_layers=None):
    if layer is None:
        raise ValueError("layer=None would request all hidden states; use 'safe', 'last', or an integer.")

    n_layers = num_layers or ESMC_API_NUM_LAYERS.get(model_id)

    if layer == "safe":
        if n_layers is None:
            raise ValueError("Pass num_layers when using layer='safe' with an unknown model_id.")
        return n_layers - 1

    if layer == "last":
        if n_layers is None:
            raise ValueError("Pass num_layers when using layer='last' with an unknown model_id.")
        return n_layers

    if isinstance(layer, int):
        if layer < 0:
            if n_layers is None:
                raise ValueError("Pass num_layers when using a negative layer with an unknown model_id.")
            return n_layers + 1 + layer
        return layer

    raise ValueError("layer must be an integer, 'safe', or 'last'.")

def write_per_position_archive_h5(
    fasta_path,
    path_h5,
    model_id="esmc-6b-2024-12",
    layer="safe",
    token_env_var="ESM_API_KEY",
    forge_url="https://biohub.ai",
    sleep_s=0.1,
    parallel_requests=5,
    max_records=None,
):
    fasta_path = Path(fasta_path)
    path_h5 = Path(path_h5)
    path_h5.parent.mkdir(parents=True, exist_ok=True)

    ids, seqs = read_fasta_biopython(fasta_path)
    if not ids:
        raise ValueError(f"No FASTA records found in {fasta_path}")

    lengths = [len(seq) for seq in seqs]
    layer_index = resolve_esmc_api_layer(model_id, layer)
    parallel_requests = int(parallel_requests)
    if parallel_requests < 1:
        raise ValueError("parallel_requests must be >= 1.")

    if token_env_var not in os.environ:
        raise ValueError(f"{token_env_var} is not set.")

    expected_attrs = {
        "format": PER_POSITION_ARCHIVE_FORMAT,
        "model_id": model_id,
        "layer": str(layer),
        "layer_index": int(layer_index),
        "forge_url": forge_url,
    }

    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path_h5, "a") as h5:
        if "ids" not in h5:
            h5.create_dataset("ids", data=np.asarray(ids, dtype=object), dtype=string_dtype)
            h5.create_dataset("sequences", data=np.asarray(seqs, dtype=object), dtype=string_dtype)
            h5.create_dataset("lengths", data=np.asarray(lengths, dtype=np.int32))
            h5.create_dataset("done", data=np.zeros(len(ids), dtype=bool))
            h5.create_group("proteins")
            for key, value in expected_attrs.items():
                h5.attrs[key] = value
            h5.attrs["fasta_path"] = str(fasta_path)
            h5.attrs["n_total"] = len(ids)
            h5.attrs["n_done"] = 0
            h5.attrs["completed"] = False
            h5.flush()
        else:
            _validate_existing_per_position_archive(h5, ids, seqs, lengths, expected_attrs)

        done = h5["done"][:].astype(bool)
        if bool(h5.attrs["completed"]) and done.all():
            print(f"Already complete: {path_h5}")
            return path_h5

        n_done = int(done.sum())
        run_target = len(ids) if max_records is None else min(int(max_records), len(ids))
        if max_records is not None and n_done >= run_target:
            print(f"max_records={max_records} target already reached: {n_done}/{len(ids)} archive rows complete.")
            return path_h5

        token = os.environ[token_env_var]
        proteins_group = h5["proteins"]
        pending_indices = [i for i in range(len(ids)) if not done[i]][:run_target - n_done]

        print(f"Writing ESMC per-position archive: {n_done}/{len(ids)} archive rows complete -> {path_h5}")
        print(f"Keeping up to {parallel_requests} API requests in flight; HDF5 writes stay sequential.")
        if max_records is not None:
            print(f"This run is limited to max_records={max_records}; progress bar target is {run_target}.")

        with tqdm(total=run_target, initial=min(n_done, run_target), unit="seq") as pbar:
            _fetch_and_write_per_position_entries(
                pending_indices=pending_indices,
                ids=ids,
                seqs=seqs,
                h5=h5,
                proteins_group=proteins_group,
                model_id=model_id,
                forge_url=forge_url,
                token=token,
                layer_index=layer_index,
                sleep_s=sleep_s,
                parallel_requests=parallel_requests,
                pbar=pbar,
            )

        if max_records is not None and int(h5.attrs["n_done"]) >= run_target and int(h5.attrs["n_done"]) < len(ids):
            print(f"Stopping after max_records={max_records}; rerun with max_records=None to continue.")

    return path_h5

def inspect_per_position_archive(path_h5):
    path_h5 = Path(path_h5)
    rows = []
    with h5py.File(path_h5, "r") as h5:
        attrs = _h5_attrs_to_dict(h5.attrs)
        done = h5["done"][:].astype(bool) if "done" in h5 else np.array([], dtype=bool)
        storage = _storage_summary(h5)
        rows.append({
            "path": str(path_h5),
            "format": attrs.get("format", ""),
            "model_id": attrs.get("model_id", ""),
            "layer": attrs.get("layer", ""),
            "layer_index": attrs.get("layer_index", ""),
            "n_total": int(attrs.get("n_total", len(done))),
            "n_done": int(done.sum()),
            "completed": bool(attrs.get("completed", False)),
            "file_size_mib": path_h5.stat().st_size / 1024 / 1024,
            "hidden_states_mib": storage["hidden_bytes"] / 1024 / 1024,
        })
    return pd.DataFrame(rows)

def derive_meanpooled_embeddings(
    path_positions_h5,
    path_out_h5,
    hidden_layer,
    l2_normalize=True,
    token_scope="all_tokens",
):
    path_positions_h5 = Path(path_positions_h5)
    path_out_h5 = Path(path_out_h5)
    path_out_h5.parent.mkdir(parents=True, exist_ok=True)

    if token_scope not in {"all_tokens", "residue_tokens"}:
        raise ValueError("token_scope must be 'all_tokens' or 'residue_tokens'.")

    with h5py.File(path_positions_h5, "r") as raw:
        done = raw["done"][:].astype(bool)
        ids = _decode_h5_strings(raw["ids"][:])
        seqs = _decode_h5_strings(raw["sequences"][:])
        lengths = raw["lengths"][:].astype(np.int32)
        keep_idx = np.where(done)[0]

        embeddings = []
        kept_ids = []
        kept_lengths = []
        for i in keep_idx:
            group = raw["proteins"][f"{i:06d}"]
            dataset_name = f"hidden_layer_{int(hidden_layer)}"
            if dataset_name not in group:
                raise ValueError(f"{dataset_name} is missing for {ids[i]}")
            hidden = group[dataset_name][:].astype(np.float32)
            if token_scope == "residue_tokens":
                hidden = slice_positions_to_residues(hidden, len(seqs[i]))
            vector = hidden.mean(axis=0)
            if l2_normalize:
                vector = vector / (np.linalg.norm(vector) + 1e-12)
            embeddings.append(vector.astype(np.float32))
            kept_ids.append(ids[i])
            kept_lengths.append(lengths[i])

        embeddings = np.vstack(embeddings).astype(np.float32)
        string_dtype = h5py.string_dtype(encoding="utf-8")
        with h5py.File(path_out_h5, "w") as out:
            out.create_dataset("ids", data=np.asarray(kept_ids, dtype=object), dtype=string_dtype)
            out.create_dataset("lengths", data=np.asarray(kept_lengths, dtype=np.int32))
            out.create_dataset("embeddings", data=embeddings)
            out.attrs["model"] = raw.attrs.get("model_id", "")
            out.attrs["layer"] = str(hidden_layer)
            out.attrs["layer_index"] = int(hidden_layer)
            out.attrs["pooling_method"] = "mean"
            out.attrs["token_scope"] = token_scope
            out.attrs["l2_normalize"] = bool(l2_normalize)
            out.attrs["source_per_position_h5"] = str(path_positions_h5)
            out.attrs["n_total"] = len(kept_ids)
            out.attrs["n_done"] = len(kept_ids)
            out.attrs["completed"] = bool(done.all())

    return path_out_h5

def slice_positions_to_residues(hidden, seq_len):
    if hidden.shape[0] == seq_len:
        return hidden
    if hidden.shape[0] >= seq_len + 2:
        return hidden[1 : 1 + seq_len]
    raise ValueError(f"Cannot align hidden-state rows {hidden.shape[0]} to sequence length {seq_len}.")

def _validate_existing_per_position_archive(h5, ids, seqs, lengths, expected_attrs):
    if _decode_h5_strings(h5["ids"][:]) != ids:
        raise ValueError("Existing HDF5 ids do not match the FASTA.")
    if _decode_h5_strings(h5["sequences"][:]) != seqs:
        raise ValueError("Existing HDF5 sequences do not match the FASTA.")
    if h5["lengths"][:].tolist() != lengths:
        raise ValueError("Existing HDF5 lengths do not match the FASTA.")
    if int(h5.attrs["n_total"]) != len(ids):
        raise ValueError("Existing HDF5 n_total does not match the FASTA.")

    for key, value in expected_attrs.items():
        if not _same_attr_value(h5.attrs.get(key), value):
            raise ValueError(f"Existing HDF5 {key} does not match this run.")

def _fetch_and_write_per_position_entries(
    pending_indices,
    ids,
    seqs,
    h5,
    proteins_group,
    model_id,
    forge_url,
    token,
    layer_index,
    sleep_s,
    parallel_requests,
    pbar,
):
    if not pending_indices:
        return

    next_submit = 0
    next_write = 0
    futures = {}
    finished = {}
    n_done = int(h5["done"][:].sum())

    def submit_one(executor):
        nonlocal next_submit
        if next_submit >= len(pending_indices):
            return
        i = pending_indices[next_submit]
        future = executor.submit(
            _fetch_sequence_hidden_states_with_new_client,
            model_id,
            forge_url,
            token,
            seqs[i],
            layer_index,
        )
        futures[future] = i
        next_submit += 1
        if sleep_s and next_submit < len(pending_indices):
            time.sleep(sleep_s)

    with ThreadPoolExecutor(max_workers=parallel_requests) as executor:
        for _ in range(min(parallel_requests, len(pending_indices))):
            submit_one(executor)

        while futures:
            completed, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in completed:
                i = futures.pop(future)
                try:
                    finished[i] = future.result()
                except BaseException as exc:
                    finished[i] = exc
                submit_one(executor)

            while next_write < len(pending_indices):
                i = pending_indices[next_write]
                if i not in finished:
                    break

                output = finished.pop(i)
                if isinstance(output, BaseException):
                    raise output

                _write_per_position_entry(
                    proteins_group,
                    i,
                    ids[i],
                    seqs[i],
                    output,
                    layer_index,
                )
                n_done += 1
                h5["done"][i] = True
                h5.attrs["n_done"] = n_done
                h5.attrs["completed"] = bool(n_done == len(ids))
                h5.flush()
                pbar.update(1)
                next_write += 1

def _write_per_position_entry(proteins_group, i, seqid, sequence, output, layer_index):
    group_name = f"{i:06d}"
    if group_name in proteins_group:
        del proteins_group[group_name]
    group = proteins_group.create_group(group_name)
    group.attrs["seqid"] = seqid
    group.attrs["length"] = len(sequence)
    write_hidden_state(group, output.hidden_states, layer_index, len(sequence))

def _fetch_sequence_hidden_states_with_new_client(model_id, forge_url, token, sequence, layer_index):
    from esm.sdk.forge import ESMCForgeInferenceClient

    client = ESMCForgeInferenceClient(model=model_id, url=forge_url, token=token)
    return _fetch_sequence_hidden_states(client, sequence, layer_index)

def _fetch_sequence_hidden_states(client, sequence, layer_index):
    from esm.sdk.api import ESMProtein, ESMProteinError, LogitsConfig

    protein_tensor = client.encode(ESMProtein(sequence=sequence))
    if isinstance(protein_tensor, ESMProteinError):
        raise RuntimeError(protein_tensor)

    output = client.logits(
        protein_tensor,
        LogitsConfig(
            return_hidden_states=True,
            ith_hidden_layer=int(layer_index),
        ),
    )
    if isinstance(output, ESMProteinError):
        raise RuntimeError(output)
    return output

def write_hidden_state(group, hidden_states, layer_index, seq_len):
    h = hidden_state_to_positions(hidden_states)
    if h.shape[0] not in {seq_len, seq_len + 2}:
        group.attrs[f"hidden_layer_{layer_index}_unexpected_rows"] = int(h.shape[0])
    group.create_dataset(
        f"hidden_layer_{int(layer_index)}",
        data=h.astype(np.float32),
        compression="gzip",
    )

def hidden_state_to_positions(hidden_states):
    if hidden_states is None:
        raise ValueError("API response did not include hidden_states.")
    h = hidden_states.detach().cpu().float().numpy()
    h = np.squeeze(h)
    if h.ndim != 2:
        raise ValueError(f"Expected squeezed hidden states [positions, dim], got {h.shape}")
    return h

def _decode_h5_strings(values):
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in values]

def _h5_attrs_to_dict(attrs):
    out = {}
    for key, value in attrs.items():
        if hasattr(value, "item"):
            value = value.item()
        out[key] = value
    return out

def _same_attr_value(left, right):
    if hasattr(left, "item"):
        left = left.item()
    return left == right

def _storage_summary(h5):
    hidden_bytes = 0

    def visit(name, obj):
        nonlocal hidden_bytes
        if not isinstance(obj, h5py.Dataset):
            return
        size = int(obj.id.get_storage_size())
        if "/hidden_layer_" in name:
            hidden_bytes += size

    h5.visititems(visit)
    return {"hidden_bytes": hidden_bytes}

def run_archive():
    if (
        cfg.MODEL_ID != ARCHIVE_MODEL_ID
        or cfg.EMBEDDING_LAYER_INDEX != ARCHIVE_LAYER_INDEX
        or cfg.FORGE_URL != ARCHIVE_FORGE_URL
        or cfg.TOKEN_ENV_VAR != ARCHIVE_TOKEN_ENV_VAR
    ):
        raise ValueError(
            "The Stage 03 archive command requires the fixed archive contract: "
            f"{ARCHIVE_MODEL_ID}, layer {ARCHIVE_LAYER_INDEX}, "
            f"{ARCHIVE_FORGE_URL}, and {ARCHIVE_TOKEN_ENV_VAR}."
        )
    cfg.EMBEDDING_OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("ESMC per-position archive extraction for concordance")
    print(f"  FASTA: {cfg.FASTA_PATH}")
    print(f"  per-position output: {cfg.PER_POSITION_ARCHIVE_PATH}")
    print(f"  model: {cfg.MODEL_ID}")
    print(f"  layer: {cfg.EMBEDDING_LAYER} -> {cfg.EMBEDDING_LAYER_INDEX}")
    print(f"  max_records: {cfg.MAX_RECORDS}")
    print(f"  parallel API requests: {cfg.API_PARALLEL_REQUESTS}")
    print()

    write_per_position_archive_h5(
        cfg.FASTA_PATH,
        path_h5=cfg.PER_POSITION_ARCHIVE_PATH,
        model_id=cfg.MODEL_ID,
        layer=cfg.EMBEDDING_LAYER,
        token_env_var=cfg.TOKEN_ENV_VAR,
        forge_url=cfg.FORGE_URL,
        parallel_requests=cfg.API_PARALLEL_REQUESTS,
        max_records=cfg.MAX_RECORDS,
    )

    status = inspect_per_position_archive(cfg.PER_POSITION_ARCHIVE_PATH).iloc[0]
    print(
        f"Archive status: {int(status['n_done']):,}/{int(status['n_total']):,} "
        f"proteins; completed={bool(status['completed'])}"
    )
    print("Deriving mean-pooled concordance embedding caches")
    print(f"  token scope: {cfg.MEAN_POOL_TOKEN_SCOPE}")
    print(f"  raw output: {cfg.MEANPOOLED_RAW_PATH}")
    print(f"  L2 output: {cfg.MEANPOOLED_L2_PATH}")

    derive_meanpooled_embeddings(
        cfg.PER_POSITION_ARCHIVE_PATH,
        cfg.MEANPOOLED_RAW_PATH,
        hidden_layer=cfg.EMBEDDING_LAYER_INDEX,
        l2_normalize=False,
        token_scope=cfg.MEAN_POOL_TOKEN_SCOPE,
    )
    derive_meanpooled_embeddings(
        cfg.PER_POSITION_ARCHIVE_PATH,
        cfg.MEANPOOLED_L2_PATH,
        hidden_layer=cfg.EMBEDDING_LAYER_INDEX,
        l2_normalize=True,
        token_scope=cfg.MEAN_POOL_TOKEN_SCOPE,
    )
