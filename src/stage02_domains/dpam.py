import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from Bio import SeqIO

from src.stage02_domains import config_domains as cfg

DPAM_ARCHITECTURE_FIELDS = [
    "protein_accession",
    "architecture",
    "n_domains",
    "n_hth",
    "hth_class",
    "hth_families",
    "min_judge",
    "has_discontinuous_domain",
    "sequence_length",
    "assigned_fraction_of_protein"]

DPAM_OBSERVATION_FIELDS = ["batch"] + DPAM_ARCHITECTURE_FIELDS

DPAM_PROTEIN_FIELDS = [
    "protein_accession",
    "cluster_id",
    "representative_accession",
    "cluster_size",
    "architecture_source",
    "annotation_donor",
    "annotation_batch",
    "architecture",
    "n_hth",
    "hth_class",
    "hth_families",
    "min_judge",
    "has_discontinuous_domain"]

@dataclass(frozen=True)
class Batch:

    name: str
    domains_path: Path
    input_fasta: Path
    description: str

    @classmethod
    def declared(cls):

        return tuple(cls(entry["name"], entry["domains_path"], entry["input_fasta"], entry["description"])
                     for entry in cfg.DPAM_BATCHES)

    def read(self):

        if not self.domains_path.exists():

            raise FileNotFoundError(f"Missing {self.domains_path} for AF3+DPAM batch '{self.name}'. "
                                    f"AlphaFold3 and DPAM produce it off-machine and no step here "
                                    f"creates it.")

        return read_dpam_domains(self.domains_path).assign(batch=self.name)

    def sequence_identity(self, roster_sequences, domains):

        observed = domains[["protein_accession", "sequence_length"]].drop_duplicates()
        observed_accessions = set(observed["protein_accession"])
        outside_roster = sorted(observed_accessions - set(roster_sequences))

        if outside_roster:

            raise ValueError(f"AF3+DPAM batch '{self.name}' observes {len(outside_roster):,} proteins outside "
                             f"the study roster ({', '.join(outside_roster[:5])}). Add them through the "
                             f"explicit roster route before selecting their structural evidence.")

        if self.input_fasta is None:

            lengths = {row.protein_accession: row.sequence_length for row in observed.itertuples()}
            wrong = sorted(accession for accession, length in lengths.items()
                           if len(roster_sequences[accession]) != length)

            if wrong:

                raise ValueError(f"AF3+DPAM batch '{self.name}' reports lengths that contradict the study "
                                 f"roster for {len(wrong):,} proteins ({', '.join(wrong[:5])}). A batch "
                                 f"describing a different sequence under the same accession is an error.")

            return "reported", f"no input FASTA; {len(lengths):,} reported lengths agree with the roster"

        present = {record.id: str(record.seq).upper() for record in SeqIO.parse(self.input_fasta, "fasta")}
        covered = sorted(observed_accessions & present.keys())
        absent = sorted(observed_accessions - present.keys())
        wrong = sorted(accession for accession in covered if roster_sequences[accession] != present[accession])

        if wrong:

            raise ValueError(f"AF3+DPAM batch '{self.name}' ran on sequences that differ from the study "
                             f"roster for {len(wrong):,} proteins ({', '.join(wrong[:5])}). The structural "
                             f"evidence does not describe the proteins being analysed.")

        if absent:

            return "partial", (f"{self.input_fasta.name} covers {len(covered):,}/{len(observed_accessions):,} "
                               f"observed proteins; {len(absent):,} have no entry in the batch FASTA "
                               f"({', '.join(absent[:5])})")

        return "verified", (f"{self.input_fasta.name} covers all {len(observed_accessions):,} observed "
                            f"proteins, all matching the roster")

def _file_digest(path):

    digest = hashlib.md5()

    with path.open("rb") as handle:

        for block in iter(lambda: handle.read(1 << 20), b""): digest.update(block)

    return digest.hexdigest()

def batch_manifest(batches, observations_by_batch, roster_sequences):

    rows = []

    for batch in batches:

        domains = observations_by_batch[batch.name]
        identity, basis = batch.sequence_identity(roster_sequences, domains)
        rows.append({"batch": batch.name,
                     "domains_path": str(batch.domains_path),
                     "domains_md5": _file_digest(batch.domains_path),
                     "input_fasta": str(batch.input_fasta) if batch.input_fasta else "",
                     "input_fasta_md5": _file_digest(batch.input_fasta) if batch.input_fasta else "",
                     "sequence_identity": identity,
                     "identity_basis": basis,
                     "proteins": int(domains["protein_accession"].nunique()),
                     "domains": int(len(domains)),
                     "description": batch.description})

    return pd.DataFrame(rows)

def read_dpam_domains(path):

    domains = pd.read_csv(path).rename(columns={"Protein": "protein_accession"})
    unconfigured = sorted(set(domains["Judge"]) - set(cfg.DPAM_JUDGE_ORDER))

    if unconfigured:

        raise ValueError("DPAM reported Judge values missing from DPAM_JUDGE_ORDER: " + ", ".join(unconfigured))

    spans = domains["Range"].astype(str).str.split(",")
    token = domains["T-group"].map(cfg.DPAM_TGROUP_ALIAS).fillna(cfg.DPAM_OTHER_TOKEN)
    domains = domains.assign(token=token.where(domains["x_name"] != cfg.DPAM_HTH_X_NAME, cfg.DPAM_HTH_X_NAME),
                             is_discontinuous=spans.str.len() > 1,
                             start=spans.str[0].str.split("-").str[0].astype(int),
                             end=spans.str[-1].str.split("-").str[-1].astype(int))
    domains = domains.assign(hth_family=_hth_family_labels(domains))
    domains = domains.sort_values(["protein_accession", "start", "end"], kind="stable").reset_index(drop=True)
    domains["domain_index"] = domains.groupby("protein_accession").cumcount() + 1

    return domains

def _hth_family_labels(domains):

    is_hth = domains["token"] == cfg.DPAM_HTH_X_NAME
    unlabelled = is_hth & domains["f_name"].isna() & domains["t_name"].isna()

    if unlabelled.any():

        offenders = ", ".join(domains.loc[unlabelled, "protein_accession"].head(5))

        raise ValueError(f"{int(unlabelled.sum()):,} HTH domains have neither f_name nor t_name "
                         f"({offenders}); the family signature has nothing to fall back on.")

    labels = domains["f_name"].where(domains["f_name"].notna(), "T:" + domains["t_name"].astype(str))

    return labels.where(is_hth, "")

def dpam_architectures(domains):

    grouped = domains.groupby("protein_accession", sort=True)
    architectures = grouped.agg(architecture=("token", " -- ".join),
                                n_domains=("token", "size"),
                                n_hth=("token", lambda tokens: (tokens == cfg.DPAM_HTH_X_NAME).sum()),
                                hth_families=("hth_family", lambda labels: " + ".join(filter(None, labels))),
                                min_judge=("Judge", lambda judges: max(judges, key=cfg.DPAM_JUDGE_ORDER.index)),
                                has_discontinuous_domain=("is_discontinuous", "any"),
                                sequence_length=("sequence_length", "first"),
                                assigned_fraction_of_protein=("assigned_fraction_of_protein", "first"),)
    saturated = len(cfg.DPAM_HTH_CLASSES) - 1
    architectures["hth_class"] = [cfg.DPAM_HTH_CLASSES[min(count, saturated)] for count in architectures["n_hth"]]

    return architectures.reset_index()[DPAM_ARCHITECTURE_FIELDS]

def architecture_observations(domains):

    observed = [dpam_architectures(frame).assign(batch=name)
                for name, frame in domains.groupby("batch", sort=False)]

    return pd.concat(observed, ignore_index=True)[DPAM_OBSERVATION_FIELDS]

def select_observations(observations):

    precedence = [entry["name"] for entry in cfg.DPAM_BATCHES]
    undeclared = sorted(set(observations["batch"]) - set(precedence))

    if undeclared:

        raise ValueError(f"AF3+DPAM observations carry undeclared batches: {', '.join(undeclared)}. "
                         f"Selection precedence is a configuration decision, so a batch must be added "
                         f"to config_domains.DPAM_BATCHES before its observations can be used.")

    ranked = observations.assign(precedence=observations["batch"].map(precedence.index))
    chosen = ranked.sort_values(["protein_accession", "precedence"], kind="stable")

    return chosen.drop_duplicates("protein_accession").drop(columns="precedence").reset_index(drop=True)

def _hth_class(n_hth):

    return cfg.DPAM_HTH_CLASSES[min(int(n_hth), len(cfg.DPAM_HTH_CLASSES) - 1)]

def _selected_domains(domains, selected):

    selection = selected[["batch", "protein_accession"]]

    return domains.merge(selection, on=["batch", "protein_accession"], how="inner",
                         validate="many_to_one")

def selected_domains():

    domains = pd.read_csv(cfg.DPAM_DOMAINS_TABLE_PATH)
    selected = select_observations(domains[["batch", "protein_accession"]].drop_duplicates())

    return _selected_domains(domains, selected)

def interpro_vs_dpam_hth(interpro_architectures, selected, clusters):

    representatives = set(clusters["representative_accession"])
    selected_representatives = selected[selected["protein_accession"].isin(representatives)]
    interpro = interpro_architectures[["seqid", "tokens"]].copy()
    interpro["interpro_n_hth"] = interpro["tokens"].map(lambda tokens: tokens.count("HTH"))
    comparable = selected_representatives.merge(interpro[["seqid", "interpro_n_hth"]],
                                                left_on="protein_accession", right_on="seqid", how="inner",
                                                validate="one_to_one")
    if comparable.empty:

        counts = pd.DataFrame({"interpro_n_hth": [pd.NA], "dpam_n_hth": [pd.NA],
                               "representative_count": [0],
                               "comparison_status": ["no_comparable_representatives"]})

    else:

        counts = (comparable.groupby(["interpro_n_hth", "n_hth"], sort=True).size()
                  .rename("representative_count").reset_index()
                  .rename(columns={"n_hth": "dpam_n_hth"}))
        counts.insert(0, "comparison_status", "comparable")

    counts.insert(0, "population", "current representatives with a Stage 02 InterProScan architecture row")
    counts.insert(1, "observation_selection", "one DPAM observation per protein by DPAM_BATCHES precedence")
    counts.insert(2, "representatives_selected", len(selected_representatives))
    counts.insert(3, "representatives_with_interpro_evidence", len(comparable))
    counts.insert(4, "representatives_missing_interpro_evidence", len(selected_representatives) - len(comparable))

    return counts[["population", "observation_selection", "representatives_selected",
                   "representatives_with_interpro_evidence", "representatives_missing_interpro_evidence",
                   "comparison_status", "interpro_n_hth", "dpam_n_hth", "representative_count"]]

def require_assignments_match_selection(proteins, selected):

    unresolved = proteins[proteins["architecture_source"] == "unresolved"]

    if len(unresolved):

        raise ValueError("Cannot summarize saved DPAM assignments with unresolved proteins. Rerun "
                         "step6_build_dpam_architectures after supplying their structural evidence.")

    resolved = proteins.reset_index(drop=True)
    expected = selected.set_index("protein_accession").reindex(resolved["annotation_donor"]).reset_index(drop=True)

    if expected["batch"].isna().any():

        raise ValueError("Saved DPAM assignments name donors absent from the currently selected observations. "
                         "Rerun step6_build_dpam_architectures before step7_build_results.")

    fields = {"batch": "annotation_batch", "architecture": "architecture", "n_hth": "n_hth",
              "hth_class": "hth_class", "hth_families": "hth_families", "min_judge": "min_judge",
              "has_discontinuous_domain": "has_discontinuous_domain"}
    changed = [saved for observed, saved in fields.items() if not expected[observed].equals(resolved[saved])]

    if changed:

        raise ValueError("Saved DPAM assignments differ from the currently selected observations in "
                         + ", ".join(changed) + ". Rerun step6_build_dpam_architectures before "
                         "step7_build_results.")

def _alternative_hth_observations(domains, observations):

    kept = domains[(domains["token"] == cfg.DPAM_HTH_X_NAME)
                   & (domains["Judge"] == cfg.DPAM_GOOD_JUDGE)]
    retained = kept.groupby(["batch", "protein_accession"], sort=False).size().rename("n_hth_after").reset_index()
    alternative = observations.merge(retained, on=["batch", "protein_accession"], how="left",
                                     validate="one_to_one")
    alternative["n_hth_after"] = alternative["n_hth_after"].fillna(0).astype(int)
    alternative["hth_class_after"] = alternative["n_hth_after"].map(_hth_class)

    return alternative

def _judge_sensitivity_row(grain, population, before, after):

    excluded = before["n_hth"] - after["n_hth_after"]

    return {"grain": grain,
            "population": population,
            "weak_hth_policy": f"drop HTH domains where Judge != {cfg.DPAM_GOOD_JUDGE}",
            "entities": len(before),
            "hth_domains_before": int(before["n_hth"].sum()),
            "non_good_hth_domains_excluded": int(excluded.sum()),
            "entities_with_excluded_hth": int((excluded > 0).sum()),
            "entities_changing_hth_class": int((before["hth_class"] != after["hth_class_after"]).sum())}

def judge_sensitivity(domains, observations, selected, proteins, clusters):

    alternative = _alternative_hth_observations(domains, observations)
    selected_alternative = alternative.merge(selected[["batch", "protein_accession"]],
                                             on=["batch", "protein_accession"], how="inner",
                                             validate="one_to_one")
    representatives = set(clusters["representative_accession"])
    representative_alternative = selected_alternative[
        selected_alternative["protein_accession"].isin(representatives)]
    donor_counts = selected_alternative.set_index("protein_accession")["n_hth_after"]
    protein_alternative = proteins[["protein_accession", "architecture_source", "annotation_donor",
                                    "n_hth", "hth_class"]].copy()
    protein_alternative["n_hth_after"] = protein_alternative["annotation_donor"].map(donor_counts)

    if protein_alternative["n_hth_after"].isna().any():

        missing = protein_alternative.loc[protein_alternative["n_hth_after"].isna(), "protein_accession"]
        raise ValueError("Cannot calculate Judge sensitivity for assignments without selected evidence: "
                         + ", ".join(missing.head(5)))

    protein_alternative["n_hth_after"] = protein_alternative["n_hth_after"].astype(int)
    protein_alternative["hth_class_after"] = protein_alternative["n_hth_after"].map(_hth_class)
    rows = [_judge_sensitivity_row("observation", "all batch observations; repeated batches retained",
                                   observations, alternative),
            _judge_sensitivity_row("representative",
                                   "selected observations for current cluster representatives",
                                   representative_alternative, representative_alternative),
            _judge_sensitivity_row("protein",
                                   "study proteins after selected direct/representative transfer",
                                   protein_alternative, protein_alternative)]

    return pd.DataFrame(rows)

def family_census(domains, selected):

    chosen = _selected_domains(domains, selected)
    metadata = {"population": "all proteins with a selected DPAM observation",
                "observation_selection": "one observation per protein by DPAM_BATCHES precedence",
                "selected_proteins": int(chosen["protein_accession"].nunique()),
                "selected_domains": int(len(chosen))}
    tokens = (chosen.groupby("token", sort=True)
              .agg(domain_count=("token", "size"), protein_count=("protein_accession", "nunique"))
              .reset_index())
    token_rows = tokens.assign(census="domain_token", hth_family="", family_resolution="")
    hth = chosen[chosen["token"] == cfg.DPAM_HTH_X_NAME].copy()
    hth["family_resolution"] = hth["hth_family"].map(
        lambda label: "topology_fallback" if label.startswith("T:") else "family")
    families = (hth.groupby(["hth_family", "family_resolution"], sort=True)
                .agg(domain_count=("hth_family", "size"), protein_count=("protein_accession", "nunique"))
                .reset_index())
    family_rows = families.assign(census="hth_family", token=cfg.DPAM_HTH_X_NAME)
    result = pd.concat([token_rows, family_rows], ignore_index=True)

    for column, value in reversed(tuple(metadata.items())):

        result.insert(0, column, value)

    return result[["population", "observation_selection", "selected_proteins", "selected_domains",
                   "census", "token", "hth_family", "family_resolution", "domain_count", "protein_count"]]

def assign_to_proteins(selected, clusters):

    available = selected.set_index("protein_accession")
    direct = clusters["protein_accession"].isin(available.index)
    donor = clusters["protein_accession"].where(direct, clusters["representative_accession"])
    resolved = donor.isin(available.index)
    annotation = available.reindex(donor).reset_index(drop=True)
    source = pd.Series("unresolved", index=clusters.index).mask(resolved & direct, "direct")
    proteins = clusters.assign(architecture_source=source.mask(resolved & ~direct, "transferred"),
                               annotation_donor=donor.where(resolved, ""),
                               annotation_batch=annotation["batch"].fillna(""),
                               architecture=annotation["architecture"],
                               n_hth=annotation["n_hth"],
                               hth_class=annotation["hth_class"],
                               hth_families=annotation["hth_families"],
                               min_judge=annotation["min_judge"],
                               has_discontinuous_domain=annotation["has_discontinuous_domain"])

    return proteins[DPAM_PROTEIN_FIELDS]

def unresolved_assignments(proteins):

    missing = proteins[proteins["architecture_source"] == "unresolved"]

    return missing[["protein_accession", "representative_accession"]].reset_index(drop=True)

ANCHOR_VALIDATION_FIELDS = [
    "anchor", "protein_accession", "dna_binding_evidence", "tRNAse_activity",
    "comparison", "cluster_size", "is_representative",
    "observed_batch", "observed_architecture", "observed_n_hth", "observed_hth_class",
    "observed_hth_families", "observed_min_judge",
    "reference_kind", "reference_batch", "reference_protein",
    "reference_architecture", "reference_n_hth", "reference_hth_class",
    "reference_hth_families", "reference_min_judge",
    "agrees_architecture", "agrees_n_hth", "agrees_hth_families"]

def anchor_validation(observations, selected, clusters, anchor_batch="anchors"):

    from src import config_main as main_cfg

    members = clusters.set_index("protein_accession")
    available = selected.set_index("protein_accession")
    grouped = {accession: frame for accession, frame in observations.groupby("protein_accession", sort=False)}
    rows = []

    for record in main_cfg.the_nine.values():

        accession = record["protein_acc"]
        seen = grouped.get(accession)

        if seen is None or anchor_batch not in set(seen["batch"]): continue

        observed = seen[seen["batch"] == anchor_batch].iloc[0]
        elsewhere = seen[seen["batch"] != anchor_batch]
        representative = members.loc[accession, "representative_accession"]

        if len(elsewhere):

            comparison, reference = "repeat_run", elsewhere.iloc[0]
            reference_kind, reference_batch, reference_protein = "batch", reference["batch"], ""

        else:

            comparison, reference = "transfer", available.loc[representative]
            reference_kind, reference_batch, reference_protein = "protein", "", representative

        rows.append({"anchor": record["Name"],
                     "protein_accession": accession,
                     "dna_binding_evidence": record["DNA_binding_evidence"],
                     "tRNAse_activity": record["tRNAse_activity"],
                     "comparison": comparison,
                     "cluster_size": int(members.loc[accession, "cluster_size"]),
                     "is_representative": accession == representative,
                     "observed_batch": observed["batch"],
                     "observed_architecture": observed["architecture"],
                     "observed_n_hth": int(observed["n_hth"]),
                     "observed_hth_class": observed["hth_class"],
                     "observed_hth_families": observed["hth_families"],
                     "observed_min_judge": observed["min_judge"],
                     "reference_kind": reference_kind,
                     "reference_batch": reference_batch,
                     "reference_protein": reference_protein,
                     "reference_architecture": reference["architecture"],
                     "reference_n_hth": int(reference["n_hth"]),
                     "reference_hth_class": reference["hth_class"],
                     "reference_hth_families": reference["hth_families"],
                     "reference_min_judge": reference["min_judge"],
                     "agrees_architecture": observed["architecture"] == reference["architecture"],
                     "agrees_n_hth": int(observed["n_hth"]) == int(reference["n_hth"]),
                     "agrees_hth_families": observed["hth_families"] == reference["hth_families"]})

    return pd.DataFrame(rows, columns=ANCHOR_VALIDATION_FIELDS)
