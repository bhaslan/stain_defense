import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from itertools import product

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from src.stage02_domains import config_domains as domains_cfg
from src.stage05_C_vmatch import background as panel
from src.stage05_C_vmatch import config_vmatch as vmatch_cfg
from src.protein_tree import config_tree as tree_cfg
from src.protein_tree import samples as tree_samples
from src.stage06_repeat_enrichment import config_enrichment as cfg
from src.stage06_repeat_enrichment import kinship

def focal_flanks(panel_name=cfg.BACKGROUND_PANEL):

    pooled = json.loads(vmatch_cfg.STEP6_SAMPLE_SUMMARY_MANIFEST_PATH.read_text())["samples_sha256"]
    if pooled != tree_samples.sha256(tree_cfg.SAMPLES_PATH): raise RuntimeError(f"Step 6C pooled another sample table than {tree_cfg.SAMPLES_PATH}; rerun its summarize.")
    units = pd.read_csv(cfg.UNIT_MEASURES_PATH, sep="\t")
    proteins = pd.read_csv(tree_cfg.PROTEIN_STATUS_PATH, sep="\t").set_index("protein_accession")
    units = units.join(proteins[["hth_class", "status"]], on="protein_accession")
    genomes = panel.focal_units()[["protein_accession", "source_context_id", "genome"]].drop_duplicates()
    units = units.merge(genomes, on=["protein_accession", "source_context_id"], how="left", validate="many_to_one")
    panel_genomes = pd.read_csv(panel.panel_paths(panel_name)["ledger_path"], sep="\t")["genome"]
    manifest = panel.contributing_genomes("GC").set_index("accession")
    units["genus"] = units["genome"].map(manifest["genus"])
    flags = pd.read_csv(panel.panel_paths(panel_name)["focal_check_path"], sep="\t", usecols=["protein_accession", "source_context_id", "cds_found", "cds_pseudo", "cds_partial"])
    flags = flags.drop_duplicates().set_index(["protein_accession", "source_context_id"], verify_integrity=True)
    units = units.join(flags, on=["protein_accession", "source_context_id"])
    flawed = units["cds_pseudo"].eq(True) | units["cds_partial"].eq(True)
    bases = [_bases(path) for path in _fasta_paths(units)]
    units["ambiguous"] = [not set(sequence) <= set("ACGT") for sequence in bases]
    units["gc"] = [round(sum(base in "GC" for base in sequence) / len(sequence), 4) for sequence in bases]
    units["exclusion"] = np.select([units["status"] != "intact", ~units["genome"].isin(panel_genomes), flawed, units["ambiguous"]],
                                   ["protein_not_intact", "genome_not_in_panel", "cds_partial_or_pseudo", "ambiguous_bases"], default="")
    unchecked = units["genome"].isin(panel_genomes) & units["cds_found"].isna()
    if unchecked.any(): raise ValueError(f"{unchecked.sum():,} flanks in the panel's genomes have no focal check; rerun Stage 05C's focal-check for {panel_name}.")
    cfg.COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    units.to_csv(panel_path(cfg.BACKGROUND_FLANKS_PATH, panel_name), sep="\t", index=False)

    return units

def panel_path(path, panel_name):

    return path if panel_name == cfg.BACKGROUND_PANEL else path.with_name(f"{path.stem}_{panel_name}{path.suffix}")

def _fasta_paths(units):

    contexts = pd.read_csv(vmatch_cfg.STEP3_CONTEXTS_PATH, sep="\t", dtype=str, usecols=["protein_accession", "scope", "search_unit_id", "database_fasta_path"])
    paths = contexts.set_index(["protein_accession", "scope", "search_unit_id"])["database_fasta_path"]

    return [paths[(protein, f"within_{side}", unit)] for protein, side, unit in zip(units["protein_accession"], units["side"], units["search_unit_id"])]

def _bases(path):

    with open(path) as handle: return "".join(line.strip() for line in handle if not line.startswith(">")).upper()

CLASSES = {"HTH+": ("HTH+",), "HTH++": ("HTH++",), "any HTH": ("HTH+", "HTH++"), "no_HTH": ("no_HTH",)}

def matched(flanks, background, repeat_set, window):

    occupied, searched = f"occupied_{window}", f"searched_bases_{window}"
    reference = background.loc[(background["repeat_set"] == repeat_set) & (background[searched] == window)]
    rate = reference.groupby(["genome", "side"])[occupied].mean().rename("background")
    focal = flanks.loc[(flanks["exclusion"] == "") & (flanks["repeat_set"] == repeat_set) & (flanks[searched] == window)]
    focal = focal.join(rate, on=["genome", "side"])

    return focal.loc[focal["background"].notna()], reference

def gc_matched(flanks, background, repeat_set, window, panel_name=cfg.BACKGROUND_PANEL):

    occupied, searched = f"occupied_{window}", f"searched_bases_{window}"
    genes = pd.read_csv(panel.panel_paths(panel_name)["genes_path"], sep="\t")
    gc = pd.concat([genes[["sequence_id", f"{side}_gc"]].rename(columns={f"{side}_gc": "gc"}).assign(side=side) for side in ("left", "right")])
    reference = background.loc[(background["repeat_set"] == repeat_set) & (background[searched] == window)]
    reference = reference.merge(gc, left_on=["protein_accession", "side"], right_on=["sequence_id", "side"], validate="one_to_one")
    edges = np.quantile(reference["gc"], np.linspace(0, 1, cfg.BACKGROUND_GC_BINS + 1))[1:-1]
    reference["gc_bin"] = np.searchsorted(edges, reference["gc"], side="right")
    rate = reference.groupby(["gc_bin", "side"])[occupied].mean().rename("background")
    focal = flanks.loc[(flanks["exclusion"] == "") & (flanks["repeat_set"] == repeat_set) & (flanks[searched] == window)].copy()
    focal["gc_bin"] = np.searchsorted(edges, focal["gc"], side="right")

    return focal.join(rate, on=["gc_bin", "side"]), reference

def compare(flanks, panel_name=cfg.BACKGROUND_PANEL):

    background = pd.read_csv(panel.panel_paths(panel_name)["units_path"], sep="\t")
    scopes = {"all": set()}

    if panel_name != cfg.BACKGROUND_PANEL:

        main = read_flanks(cfg.BACKGROUND_PANEL)
        scopes["absent_from_main"] = set(main.loc[main["exclusion"] == "", "group_accession"])

    rows = []

    for repeat_set, window in product((repeat_set.name for repeat_set in vmatch_cfg.STEP5_REPEAT_SETS), cfg.BACKGROUND_WINDOWS):

        occupied = f"occupied_{window}"

        for matching, (focal, reference) in (("genome", matched(flanks, background, repeat_set, window)),
                                             ("gc", gc_matched(flanks, background, repeat_set, window, panel_name))):

            for (name, members), (scope, left_out) in product(CLASSES.items(), scopes.items()):

                chosen = focal.loc[focal["hth_class"].isin(members) & ~focal["group_accession"].isin(left_out)]
                if chosen.empty: continue
                rows.append({"matching": matching, "samples_scope": scope, "repeat_set": repeat_set, "window": window, "hth_class": name,
                             "samples": chosen["group_accession"].nunique(), "flanks": len(chosen), "genomes": chosen["genome"].nunique(),
                             "genera": chosen["genus"].nunique(), "background_flanks": len(reference), **_estimates(chosen, occupied)})

    table = pd.DataFrame(rows)
    table.to_csv(panel_path(cfg.BACKGROUND_COMPARISON_PATH, panel_name), sep="\t", index=False)

    return table

def _estimates(flanks, outcome):

    cells = flanks.groupby(["group_accession", "genus"]).agg(n=(outcome, "size"), focal=(outcome, "sum"), background=("background", "sum"))
    if cells["n"].sum() != len(flanks): raise ValueError("A used flank has no genus.")
    wide = cells.unstack(fill_value=0)
    samples, genera = wide["n"].shape
    rng = np.random.default_rng(cfg.BACKGROUND_SEED)
    drawn_genera = rng.multinomial(genera, np.full(genera, 1 / genera), size=cfg.BACKGROUND_BOOTSTRAP)
    drawn_samples = rng.multinomial(samples, np.full(samples, 1 / samples), size=cfg.BACKGROUND_BOOTSTRAP)
    counts = drawn_genera @ wide["n"].to_numpy().T
    present = drawn_samples * (counts > 0)

    with np.errstate(invalid="ignore", divide="ignore"):

        focal, background = (np.nansum(present * (drawn_genera @ wide[column].to_numpy().T) / counts, axis=1) / present.sum(axis=1)
                             for column in ("focal", "background"))

    totals = wide.T.groupby(level=0).sum().T
    point_focal, point_background = (totals["focal"] / totals["n"]).mean(), (totals["background"] / totals["n"]).mean()
    interval = lambda values: np.percentile(values, [2.5, 97.5]).round(4).tolist()
    differences = focal - background
    p_value = min(max(2 * min((differences <= 0).mean(), (differences >= 0).mean()), 1 / cfg.BACKGROUND_BOOTSTRAP), 1.0)
    estimates = {"focal_rate": round(point_focal, 4), "focal_rate_ci": interval(focal), "background_rate": round(point_background, 4),
                 "difference": round(point_focal - point_background, 4), "difference_ci": interval(differences),
                 "ratio": round(point_focal / point_background, 2), "ratio_ci": np.percentile(focal / background, [2.5, 97.5]).round(2).tolist(),
                 "p_value": p_value}

    if not flanks[outcome].any(): estimates.update(focal_rate_ci=np.nan, difference_ci=np.nan, ratio_ci=np.nan, p_value=np.nan)

    return estimates

def entered(flanks):

    rows = flanks.loc[flanks["repeat_set"] == vmatch_cfg.STEP5_PRIMARY_REPEAT_SET]
    lines = []

    for hth_class, group in rows.groupby("hth_class"):

        used = group.loc[group["exclusion"] == ""]
        lines.append(f"{hth_class}: {len(group):,} flanks in {group['group_accession'].nunique():,} samples; used {len(used):,} flanks "
                     f"in {used['group_accession'].nunique():,} samples, {used['genome'].nunique():,} genomes, {used['genus'].nunique():,} genera; "
                     f"excluded {group['exclusion'].replace('', np.nan).dropna().value_counts().to_dict()}; "
                     f"complete at 200 {int((used['searched_bases_200'] == 200).sum()):,}, at 500 {int((used['searched_bases_500'] == 500).sum()):,}; "
                     f"gene not found in today's annotation, kept {int(used['cds_found'].eq(False).sum()):,}")

    return lines

def model_cells(flanks, background, repeat_set, window, reported=CLASSES):

    occupied = f"occupied_{window}"
    focal, reference = matched(flanks, background, repeat_set, window)
    focal = focal.loc[focal["hth_class"].isin({member for members in reported.values() for member in members})].copy()
    reference = reference.loc[reference["genome"].isin(focal["genome"])]
    classes = [name for name in cfg.HTH_CLASSES if name in set(focal["hth_class"])]
    genomes = {genome: index for index, genome in enumerate(sorted(focal["genome"].unique()), 1)}
    samples = {sample: index for index, sample in enumerate(sorted(focal["group_accession"].unique()), 1)}

    for name, members in reported.items():

        chosen = focal["hth_class"].isin(members)
        per_sample = focal.loc[chosen].groupby("group_accession").size()
        focal[f"w_{name}"] = np.where(chosen, 1 / focal["group_accession"].map(per_sample) / len(per_sample), 0.0)

    weights = [f"w_{name}" for name in reported]
    focal_cells = (focal.groupby(["hth_class", "group_accession", "genome", "side"])
                   .agg(n=(occupied, "size"), k=(occupied, "sum"), **{column: (column, "sum") for column in weights}).reset_index())
    focal_cells["class_index"] = focal_cells["hth_class"].map({name: index for index, name in enumerate(classes, 1)})
    focal_cells["sample"] = focal_cells["group_accession"].map(samples)
    background_cells = reference.groupby(["genome", "side"]).agg(n=(occupied, "size"), k=(occupied, "sum")).reset_index()
    background_cells[["class_index", "sample", *weights]] = 0
    cells = pd.concat([focal_cells, background_cells], ignore_index=True)
    cells["genome"] = cells["genome"].map(genomes)
    cells["side"] = np.where(cells["side"] == "right", 0.5, -0.5)

    return cells[["n", "k", "class_index", "genome", "sample", "side", *weights]], classes

def read_flanks(panel_name=cfg.BACKGROUND_PANEL):

    if panel_name == cfg.POOLED_PANEL:

        flanks = pd.concat([flanks.loc[flanks["exclusion"] == ""] for flanks in map(read_flanks, cfg.POOLED_PANELS)], ignore_index=True)
        if flanks.duplicated(["repeat_set", "representative_unit_id"]).any(): raise ValueError("A flank is used in more than one pooled panel.")

        return flanks

    flanks = pd.read_csv(panel_path(cfg.BACKGROUND_FLANKS_PATH, panel_name), sep="\t", keep_default_na=False, na_values=[""])
    flanks["exclusion"] = flanks["exclusion"].fillna("")

    return flanks

def model_dir(panel_name, repeat_set, window): return cfg.MODELS_DIR / panel_name / f"{repeat_set}_w{window}"

def fit_models(panel_name, repeat_set, window):

    background = pd.read_csv(panel.panel_paths(panel_name)["units_path"], sep="\t")

    return _fit(*model_cells(read_flanks(panel_name), background, repeat_set, window), model_dir(panel_name, repeat_set, window))

def _fit(cells, classes, out, settings=cfg.BAYES_SETTINGS):

    out.mkdir(parents=True, exist_ok=True)
    cells.to_csv(out / "cells.tsv", sep="\t", index=False)
    (out / "classes.json").write_text(json.dumps(classes))
    (out / "settings.json").write_text(json.dumps(dict(settings, build_dir=str(cfg.BAYES_BUILD_DIR)), indent=2))
    cfg.BAYES_BUILD_DIR.mkdir(parents=True, exist_ok=True)

    for name, environment, script in (("bayes", cfg.BAYES_ENV, cfg.BACKGROUND_BAYES_SCRIPT), ("glmm", cfg.GLMM_ENV, cfg.BACKGROUND_GLMM_SCRIPT)):

        command = ["conda", "run", "--no-capture-output", "-n", environment, "Rscript", str(script), str(out)]
        with open(out / f"{name}_fit.log", "w") as log: subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)

    return out

def _pair(text):

    if not isinstance(text, str): return np.nan, np.nan
    low, high = (float(value) for value in text.strip("[]").split(","))

    return low, high

def model_table():

    rows = []

    for panel_name in cfg.BACKGROUND_MODEL_PANELS:

        counting = pd.read_csv(panel_path(cfg.BACKGROUND_COMPARISON_PATH, panel_name), sep="\t")
        counting = counting.loc[(counting["matching"] == "genome") & (counting["samples_scope"] == "all")]

        for repeat_set, window in cfg.BACKGROUND_MODEL_FITS:

            key = {"panel": panel_name, "repeat_set": repeat_set, "window": window}
            chosen = counting.loc[(counting["repeat_set"] == repeat_set) & (counting["window"] == window)]
            rows.extend(_model_rows(key, chosen, model_dir(panel_name, repeat_set, window)))

    table = pd.DataFrame(rows)
    table.to_csv(cfg.BACKGROUND_MODEL_TABLE_PATH, sep="\t", index=False)

    return table

def _model_rows(key, counting, out):

    rows = []

    for row in counting.itertuples():

        low, high = _pair(row.difference_ci)
        ratio_low, ratio_high = _pair(row.ratio_ci)
        rows.append({**key, "hth_class": row.hth_class, "method": "counting", "samples": row.samples, "focal_rate": row.focal_rate,
                     "background_rate": row.background_rate, "difference": row.difference, "low": low, "high": high,
                     "ratio": row.ratio, "ratio_low": ratio_low, "ratio_high": ratio_high, "p_value": row.p_value})

    bayes = pd.read_csv(out / "bayes_summary.tsv", sep="\t")
    glmm = pd.read_csv(out / "glmm_summary.tsv", sep="\t")
    rows.extend({**key, "hth_class": row.group, "method": "bayes", "focal_rate": row.focal_rate, "background_rate": row.background_rate,
                 "difference": row.difference_median, "low": row.difference_low, "high": row.difference_high,
                 "ratio": row.ratio_median, "ratio_low": row.ratio_low, "ratio_high": row.ratio_high,
                 "posterior_above_zero": row.difference_above_zero} for row in bayes.itertuples())
    rows.extend({**key, "hth_class": row.group, "method": "glmmTMB", "focal_rate": row.focal_rate, "background_rate": row.background_rate,
                 "difference": row.difference, "ratio": row.ratio} for row in glmm.itertuples())

    return rows

LAYOUTS = ("convergent", "same strand, far", "same strand, near", "none within the flank", "uncertain")

def downstream_layout(strand, status, gap, neighbour):

    gap = pd.to_numeric(pd.Series(gap), errors="coerce").to_numpy()
    same = np.asarray(neighbour) == np.asarray(strand)
    status = np.asarray(status)

    return np.select([status == "no_cds_within_flank", status == "uncertain_boundary", same & (gap <= cfg.OPERON_GAP), same, ~same],
                     ["none within the flank", "uncertain", "same strand, near", "same strand, far", "convergent"], "uncertain")

def terminator_check(panel_name=cfg.BACKGROUND_PANEL):

    occupied, searched = f"occupied_{cfg.TERMINATOR_WINDOW}", f"searched_bases_{cfg.TERMINATOR_WINDOW}"
    focal = read_flanks(panel_name)
    focal = focal.loc[(focal["exclusion"] == "") & (focal["repeat_set"] == cfg.TERMINATOR_REPEAT_SET) & (focal["side"] == "right")
                      & (focal[searched] == cfg.TERMINATOR_WINDOW)].copy()
    contexts = pd.read_csv(vmatch_cfg.STEP2_CONTEXTS_PATH, sep="\t", dtype=str, keep_default_na=False).set_index(["protein_accession", "context_id"])
    context = contexts.loc[list(zip(focal["protein_accession"], focal["source_context_id"]))]
    focal["layout"] = downstream_layout(context["strand"], context["right_boundary_status"], context["right_intergenic_length"], context["right_neighbor_cds_strand"])

    paths = panel.panel_paths(panel_name)
    background = pd.read_csv(paths["units_path"], sep="\t")
    background = background.loc[(background["repeat_set"] == cfg.TERMINATOR_REPEAT_SET) & (background["side"] == "right")
                                & (background[searched] == cfg.TERMINATOR_WINDOW) & background["genome"].isin(focal["genome"])]
    units = pd.read_csv(paths["contexts_path"], sep="\t", dtype=str, keep_default_na=False)
    units = units.loc[units["database_side"] == "right"].set_index("protein_accession").loc[background["protein_accession"]]
    background["layout"] = downstream_layout(units["central_cds_strand"], units["right_boundary_status"], units["right_intergenic_length"],
                                             units["right_neighbor_cds_strand"])
    rate = background.groupby("layout")[occupied].mean()
    focal["background"] = focal["layout"].map(rate)
    rows = []

    for name, members in CLASSES.items():

        chosen_class = focal.loc[focal["hth_class"].isin(members)]

        for layout in LAYOUTS:

            chosen = chosen_class.loc[chosen_class["layout"] == layout]
            if chosen["group_accession"].nunique() < 2 or not rate.get(layout, 0): continue
            rows.append({"panel": panel_name, "hth_class": name, "layout": layout, "flanks": len(chosen), "positive_flanks": int(chosen[occupied].sum()),
                         "samples": chosen["group_accession"].nunique(), "genera": chosen["genus"].nunique(), "share": round(len(chosen) / len(chosen_class), 3),
                         "background_flanks": int((background["layout"] == layout).sum()),
                         "background_share": round((background["layout"] == layout).mean(), 3), **_estimates(chosen, occupied)})

    table = pd.DataFrame(rows)
    table.to_csv(panel_path(cfg.TERMINATOR_CHECK_PATH, panel_name), sep="\t", index=False)

    return table

def with_rings(units):

    for inner, outer in cfg.SIDE_RINGS:

        covered = units[f"repeat_bases_{outer}"] - (units[f"repeat_bases_{inner}"] if inner else 0)
        units[f"occupied_{inner}_{outer}"] = (covered > 0).astype(float).where(units[f"searched_bases_{outer}"] == outer)

    return units

def by_side(flanks, panel_name=cfg.BACKGROUND_PANEL):

    names = cfg.POOLED_PANELS if panel_name == cfg.POOLED_PANEL else (panel_name,)
    background = with_rings(pd.concat([pd.read_csv(panel.panel_paths(name)["units_path"], sep="\t") for name in names], ignore_index=True))
    flanks = with_rings(flanks.loc[flanks["exclusion"] == ""].copy())
    rows = []

    for repeat_set, (inner, outer) in product((repeat_set.name for repeat_set in vmatch_cfg.STEP5_REPEAT_SETS), cfg.SIDE_RINGS):

        outcome = f"occupied_{inner}_{outer}"
        reference = background.loc[(background["repeat_set"] == repeat_set) & background[outcome].notna()]
        rate = reference.groupby(["genome", "side"])[outcome].mean().rename("background")
        focal = flanks.loc[(flanks["repeat_set"] == repeat_set) & flanks[outcome].notna()].join(rate, on=["genome", "side"])
        focal = focal.loc[focal["background"].notna()]

        for (name, members), side in product(CLASSES.items(), ("left", "right")):

            chosen = focal.loc[focal["hth_class"].isin(members) & (focal["side"] == side)]
            if chosen.empty: continue
            rows.append({"repeat_set": repeat_set, "inner": inner, "outer": outer, "hth_class": name, "side": side, "samples": chosen["group_accession"].nunique(),
                         "flanks": len(chosen), "genera": chosen["genus"].nunique(), **_estimates(chosen, outcome)})

    table = pd.DataFrame(rows)
    table.to_csv(panel_path(cfg.BACKGROUND_SIDES_PATH, panel_name), sep="\t", index=False)

    return table

ARRAY_KEYS = ["protein_accession", "search_unit_id", "side"]

def with_unit_sizes(units, arrays, ranges):

    window = cfg.UNIT_SIZE_WINDOW
    near = arrays.loc[arrays["start_0based"] < window]
    units = units.loc[(units["repeat_set"] == "tandem") & (units[f"searched_bases_{window}"] == window)].copy()
    flank = pd.MultiIndex.from_frame(units[ARRAY_KEYS])
    held = lambda chosen: flank.isin(pd.MultiIndex.from_frame(chosen[ARRAY_KEYS])).astype(int)
    if (held(near) != units[f"occupied_{window}"]).any(): raise ValueError("Stage 05C's TRF arrays no longer give the tandem set's window outcome.")

    marks = {f"units_{low}_{high}": held(near.loc[near["period"].between(low, high)]) for low, high in ranges}

    return pd.concat([units, pd.DataFrame(marks, index=units.index)], axis=1)

def by_unit_size(flanks):

    paths = panel.panel_paths(cfg.BACKGROUND_PANEL)
    sizes = range(vmatch_cfg.TRF_MIN_PERIOD, cfg.UNIT_SPECTRUM_LARGEST + 1)
    ranges = [*cfg.UNIT_SIZE_CLASSES, *((size, size) for size in sizes)]
    background = with_unit_sizes(pd.read_csv(paths["units_path"], sep="\t"), pd.read_csv(paths["trf_arrays_path"], sep="\t"), ranges)
    focal = with_unit_sizes(flanks.loc[flanks["exclusion"] == ""], pd.read_csv(vmatch_cfg.STEP3_TRF_ARRAYS_PATH, sep="\t"), ranges)
    rates = background.groupby(["genome", "side"])[[f"units_{low}_{high}" for low, high in ranges]].mean()
    focal = focal.loc[focal.set_index(["genome", "side"]).index.isin(rates.index)]
    matched_rates = rates.loc[list(zip(focal["genome"], focal["side"]))].set_axis(focal.index)
    class_rows, spectrum_rows = [], []

    for (name, members), side in product(CLASSES.items(), ("left", "right")):

        chosen = focal.loc[focal["hth_class"].isin(members) & (focal["side"] == side)]
        if chosen.empty: continue
        key = {"hth_class": name, "side": side, "samples": chosen["group_accession"].nunique(), "flanks": len(chosen)}

        for low, high in cfg.UNIT_SIZE_CLASSES:

            outcome = f"units_{low}_{high}"

            with np.errstate(divide="ignore", invalid="ignore"): estimates = _estimates(chosen.assign(background=matched_rates[outcome]), outcome)

            class_rows.append({"low": low, "high": high, **key, "genera": chosen["genus"].nunique(),
                               "samples_with": chosen.loc[chosen[outcome] == 1, "group_accession"].nunique(), **estimates})

        weight = (1 / chosen.groupby("group_accession")["group_accession"].transform("size") / key["samples"]).to_numpy()
        columns = [f"units_{size}_{size}" for size in sizes]
        held = chosen[columns]
        spectrum_rows.extend({"unit_size": size, **key, "samples_with": samples_with, "focal_rate": round(focal_rate, 5), "background_rate": round(background_rate, 5)}
                             for size, samples_with, focal_rate, background_rate in zip(sizes, held.groupby(chosen["group_accession"]).max().sum(),
                                                                                        weight @ held.to_numpy(), weight @ matched_rates.loc[chosen.index, columns].to_numpy()))

    tables = pd.DataFrame(class_rows), pd.DataFrame(spectrum_rows)
    tables[0].to_csv(cfg.UNIT_SIZE_PATH, sep="\t", index=False)
    tables[1].to_csv(cfg.UNIT_SPECTRUM_PATH, sep="\t", index=False)

    return tables

GC_SIDES = {"left": ("left",), "right": ("right",), "both": ("left", "right")}

GAP_LABELS = ("overlapping", f"0-{cfg.GC_GAP_EDGES[0]}", *(f"{low + 1}-{high}" for low, high in zip(cfg.GC_GAP_EDGES, cfg.GC_GAP_EDGES[1:])),
              f"over {cfg.GC_GAP_EDGES[-1]}", "none within the flank")

def gap_bins(status, gap):

    gap = pd.to_numeric(pd.Series(gap), errors="coerce").to_numpy()
    status = np.asarray(status)
    if np.isnan(gap[status == "annotated"]).any(): raise ValueError("An annotated neighbour has no intergenic length.")
    binned = np.array(GAP_LABELS[1:-1])[np.searchsorted(cfg.GC_GAP_EDGES, gap)]

    return np.select([status == "overlapping", status == "no_cds_within_flank", status == "annotated"], ["overlapping", "none within the flank", binned], "uncertain")

def _offset(flanks, expected):

    names = {"focal_rate": "focal_gc", "focal_rate_ci": "focal_gc_ci", "background_rate": "expected_gc", "difference": "offset", "difference_ci": "offset_ci", "p_value": "p_value"}

    return {names[key]: value for key, value in _estimates(flanks.assign(background=expected), "gc").items() if key in names}

def flank_gc(flanks):

    whole = vmatch_cfg.FLANK_LENGTH
    paths = panel.panel_paths(cfg.BACKGROUND_PANEL)
    searched = pd.read_csv(paths["units_path"], sep="\t", usecols=["repeat_set", "protein_accession", "side", f"searched_bases_{whole}"])
    searched = searched.loc[(searched["repeat_set"] == vmatch_cfg.STEP5_PRIMARY_REPEAT_SET) & (searched[f"searched_bases_{whole}"] == whole), ["protein_accession", "side"]]
    genes = pd.read_csv(paths["genes_path"], sep="\t")
    contexts = pd.read_csv(paths["contexts_path"], sep="\t", dtype=str, keep_default_na=False,
                           usecols=["protein_accession", "database_side", *(f"{side}_{field}" for side in ("left", "right") for field in ("boundary_status", "intergenic_length"))])
    ordinary = []

    for side in ("left", "right"):

        rows = genes[["genome", "sequence_id", f"{side}_gc"]].set_axis(["genome", "protein_accession", "gc"], axis=1).assign(side=side)
        rows = rows.merge(searched, on=["protein_accession", "side"], validate="one_to_one")
        context = contexts.loc[contexts["database_side"] == side].set_index("protein_accession").loc[rows["protein_accession"]]
        ordinary.append(rows.assign(gap=gap_bins(context[f"{side}_boundary_status"], context[f"{side}_intergenic_length"])))

    ordinary = pd.concat(ordinary, ignore_index=True)
    if ordinary["gc"].isna().any(): raise ValueError("A searched ordinary flank has no GC in the panel's gene table.")
    genome_gc = ordinary.groupby(["genome", "side"])["gc"].mean().rename("genome_gc")
    ordinary["offset"] = ordinary["gc"] - ordinary.join(genome_gc, on=["genome", "side"])["genome_gc"]
    gap_offset = ordinary.loc[ordinary["gap"] != "uncertain"].groupby(["side", "gap"])["offset"].mean()

    focal = flanks.loc[(flanks["exclusion"] == "") & (flanks["repeat_set"] == vmatch_cfg.STEP5_PRIMARY_REPEAT_SET) & (flanks[f"searched_bases_{whole}"] == whole)]
    context = pd.read_csv(vmatch_cfg.STEP2_CONTEXTS_PATH, sep="\t", dtype=str, keep_default_na=False).set_index(["protein_accession", "context_id"])
    context = context.loc[list(zip(focal["protein_accession"], focal["source_context_id"]))]
    left = focal["side"].to_numpy() == "left"
    gap = gap_bins(np.where(left, context["left_boundary_status"], context["right_boundary_status"]), np.where(left, context["left_intergenic_length"], context["right_intergenic_length"]))
    focal = focal.assign(gap=gap).join(genome_gc, on=["genome", "side"])
    if focal["genome_gc"].isna().any(): raise ValueError("A used flank's genome has no ordinary flank searched to its whole length on that side.")
    scopes = {"all flanks": focal, "flanks without a repeat": focal.loc[focal[f"occupied_{whole}"] == 0]}
    rows = []

    for (name, members), (side, held_sides), (scope, chosen_scope) in product(CLASSES.items(), GC_SIDES.items(), scopes.items()):

        chosen = chosen_scope.loc[chosen_scope["hth_class"].isin(members) & chosen_scope["side"].isin(held_sides)]
        if chosen["group_accession"].nunique() < 2: continue
        known = chosen.loc[chosen["gap"] != "uncertain"]
        expected = known["genome_gc"] + gap_offset.reindex(list(zip(known["side"], known["gap"]))).to_numpy()
        if expected.isna().any(): raise ValueError("A focal flank's gap has no ordinary flank to set its expected GC.")

        for adjustment, held, against in (("genome", chosen, chosen["genome_gc"]), ("genome and gap", known, expected)):

            rows.append({"hth_class": name, "side": side, "scope": scope, "adjustment": adjustment, "samples": held["group_accession"].nunique(),
                         "flanks": len(held), "genera": held["genus"].nunique(), **_offset(held, against)})

    overall = pd.DataFrame(rows)
    overall.to_csv(cfg.FLANK_GC_PATH, sep="\t", index=False)
    rows = []

    for (name, members), (side, held_sides), gap in product(CLASSES.items(), GC_SIDES.items(), GAP_LABELS):

        chosen = focal.loc[focal["hth_class"].isin(members) & focal["side"].isin(held_sides) & (focal["gap"] == gap)]
        reference = ordinary.loc[ordinary["side"].isin(held_sides) & (ordinary["gap"] == gap), "offset"]
        if chosen["group_accession"].nunique() < 2: continue
        rows.append({"hth_class": name, "side": side, "gap": gap, "samples": chosen["group_accession"].nunique(), "flanks": len(chosen), "genera": chosen["genus"].nunique(),
                     **_offset(chosen, chosen["genome_gc"]), "ordinary_flanks": len(reference), "ordinary_offset": round(reference.mean(), 4)})

    by_gap = pd.DataFrame(rows)
    by_gap.to_csv(cfg.FLANK_GC_GAPS_PATH, sep="\t", index=False)

    return overall, by_gap

def branches(count):

    matrix = pd.read_csv(kinship.frozen(cfg.BRANCH_KINSHIP_PATH), sep="\t", index_col=0)
    diagonal = np.diag(matrix.to_numpy())
    distance = diagonal[:, None] + diagonal[None, :] - 2 * matrix.to_numpy()
    np.fill_diagonal(distance, 0)
    groups = fcluster(linkage(squareform(distance.clip(min=0), checks=False), "average"), count, "maxclust")

    return pd.Series(groups, index=tree_samples.clusters().loc[matrix.index].to_numpy(), name="branch")

def by_branch(flanks, repeat_set="all", window=200):

    background = pd.read_csv(panel.panel_paths(cfg.BACKGROUND_PANEL)["units_path"], sep="\t")
    focal, _ = matched(flanks, background, repeat_set, window)
    focal["branch"] = focal["group_accession"].map(tree_samples.clusters()).map(branches(cfg.BACKGROUND_BRANCHES))
    if focal["branch"].isna().any(): raise ValueError(f"{focal.loc[focal['branch'].isna(), 'group_accession'].nunique():,} samples have no cluster tip on the frozen core tree.")
    occupied = f"occupied_{window}"
    rows = []

    for (branch, name), chosen in focal.groupby(["branch", "hth_class"]):

        if chosen["group_accession"].nunique() < 2: continue
        rows.append({"repeat_set": repeat_set, "window": window, "branch": int(branch), "hth_class": name, "samples": chosen["group_accession"].nunique(),
                     "genera": chosen["genus"].nunique(), **_estimates(chosen, occupied)})

    table = pd.DataFrame(rows)
    table.to_csv(cfg.BACKGROUND_BRANCHES_PATH, sep="\t", index=False)

    return table

def with_families(flanks):

    families = pd.read_csv(domains_cfg.DPAM_PROTEIN_ARCHITECTURES_PATH, usecols=["protein_accession", "hth_families"], index_col="protein_accession")
    labels = flanks["protein_accession"].map(families["hth_families"])
    missing = labels.isna() & (flanks["hth_class"] != "no_HTH")
    if missing.any(): raise ValueError(f"{flanks.loc[missing, 'protein_accession'].nunique():,} proteins of an HTH class have no HTH family in Stage 02.")
    labels = labels.str.split(" + ", regex=False).map(lambda names: sorted(set(names)), na_action="ignore")

    return flanks.assign(family=labels).explode("family").dropna(subset=["family"])

def by_family(flanks):

    background = pd.read_csv(panel.panel_paths(cfg.BACKGROUND_PANEL)["units_path"], sep="\t")
    flanks = with_families(flanks)
    rows = []

    for repeat_set, window in product((repeat_set.name for repeat_set in vmatch_cfg.STEP5_REPEAT_SETS), cfg.BACKGROUND_WINDOWS):

        focal, _ = matched(flanks, background, repeat_set, window)

        for (name, family), chosen in focal.groupby(["hth_class", "family"]):

            samples, genera = chosen["group_accession"].nunique(), chosen["genus"].nunique()
            if samples < cfg.FAMILY_MIN_SAMPLES or genera < cfg.FAMILY_MIN_GENERA: continue
            rows.append({"repeat_set": repeat_set, "window": window, "hth_class": name, "family": family, "samples": samples, "flanks": len(chosen),
                         "genomes": chosen["genome"].nunique(), "genera": genera, **_estimates(chosen, f"occupied_{window}")})

    table = pd.DataFrame(rows)
    table.to_csv(cfg.FAMILY_COMPARISON_PATH, sep="\t", index=False)

    return table

def family_dir(family, repeat_set, window): return cfg.FAMILY_MODELS_DIR / re.sub(r"\W+", "_", family).strip("_") / f"{repeat_set}_w{window}"

def _family_fits(counting):

    fitted = [(row.repeat_set, row.window) in cfg.FAMILY_MODEL_FITS for row in counting.itertuples()]

    return counting.loc[fitted].groupby(["family", "repeat_set", "window"])

def fit_family_models():

    flanks = with_families(read_flanks())
    background = pd.read_csv(panel.panel_paths(cfg.BACKGROUND_PANEL)["units_path"], sep="\t")

    def fit(group):

        (family, repeat_set, window), rows = group
        reported = {name: (name,) for name in rows["hth_class"]}
        cells, classes = model_cells(flanks.loc[flanks["family"] == family], background, repeat_set, window, reported)

        return _fit(cells, classes, family_dir(family, repeat_set, window), cfg.FAMILY_BAYES_SETTINGS)

    with ThreadPoolExecutor(cfg.FAMILY_PARALLEL_FITS) as pool:

        return list(pool.map(fit, _family_fits(pd.read_csv(cfg.FAMILY_COMPARISON_PATH, sep="\t"))))

def family_model_table():

    rows = []

    for (family, repeat_set, window), chosen in _family_fits(pd.read_csv(cfg.FAMILY_COMPARISON_PATH, sep="\t")):

        rows.extend(_model_rows({"family": family, "repeat_set": repeat_set, "window": window}, chosen, family_dir(family, repeat_set, window)))

    table = pd.DataFrame(rows)
    table.to_csv(cfg.FAMILY_MODEL_TABLE_PATH, sep="\t", index=False)

    return table
