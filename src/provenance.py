import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from src import config_main as main_cfg
from src.protein_tree import config_tree as tree_cfg
from src.stage02_domains import config_domains as domains_cfg
from src.stage03_concordance import config_concordance as concordance_cfg
from src.stage05_C_vmatch import config_vmatch as vmatch_cfg
from src.stage06_repeat_enrichment import config_enrichment as enrichment_cfg

BASELINES_DIR = main_cfg.P_TRIAL3 / "baselines"
SNAPSHOT_FILENAME = "SNAPSHOT.json"

def _artifacts():

    return {
        "roster": {
            "study_roster": main_cfg.STUDY_ROSTER_PATH,
            "study_sequences": main_cfg.STUDY_SEQUENCE_FASTA,
            "df_architectures": domains_cfg.DF_ARCHITECTURES_PATH,
            "combined_fasta": domains_cfg.COMBINED_FASTA_PATH,
            "concordance": concordance_cfg.CONCORDANCE_PATH,
            "blast_alignment": vmatch_cfg.BLAST_ALIGNMENT_PATH,
        },
        "structural_input": {
            "dpam_domains": domains_cfg.DPAM_DOMAINS_PATH,
            "dpam_input_fasta": domains_cfg.DPAM_INPUT_FASTA_PATH,
        },
        "batch_selection": {
            "dpam_anchor_domains": domains_cfg.DPAM_ANCHOR_DOMAINS_PATH,
            "dpam_anchor_input_fasta": domains_cfg.DPAM_ANCHOR_INPUT_FASTA_PATH,
            "dpam_batch_manifest": domains_cfg.DPAM_BATCH_MANIFEST_PATH,
            "dpam_architecture_observations": domains_cfg.DPAM_ARCHITECTURE_OBSERVATIONS_PATH,
        },
        "clustering": {
            "representative_fasta": domains_cfg.REPRESENTATIVE_FASTA_PATH,
            "representative_clusters": domains_cfg.REPRESENTATIVE_CLUSTERS_PATH,
            "representative_list": domains_cfg.REPRESENTATIVE_LIST_PATH,
            "mmseqs_cluster": domains_cfg.MMSEQS_CLUSTER_TSV_PATH,
            "representative_alignments": domains_cfg.REPRESENTATIVE_ALIGNMENTS_PATH,
            "cluster_similarity": domains_cfg.CLUSTER_SIMILARITY_PATH,
        },
        "annotation": {
            "dpam_domains_table": domains_cfg.DPAM_DOMAINS_TABLE_PATH,
            "dpam_architectures": domains_cfg.DPAM_ARCHITECTURES_PATH,
            "dpam_protein_architectures": domains_cfg.DPAM_PROTEIN_ARCHITECTURES_PATH,
        },
        "genomic_context": {
            "context_availability": vmatch_cfg.CONTEXT_AVAILABILITY_PATH,
            "context_records": vmatch_cfg.CONTEXT_RECORDS_PATH,
            "candidate_summary": vmatch_cfg.CANDIDATE_SUMMARY_PATH,
            "manifest_full": vmatch_cfg.FULL_MANIFEST_PATH,
            "manifest_active": vmatch_cfg.ACTIVE_MANIFEST_PATH,
        },
        "repeat_evidence": {
            "step2_contexts": vmatch_cfg.STEP2_CONTEXTS_PATH,
            "step2_proteins": vmatch_cfg.STEP2_PROTEINS_PATH,
            "step2_left_clusters": vmatch_cfg.STEP2_LEFT_CLUSTERS_PATH,
            "step2_right_clusters": vmatch_cfg.STEP2_RIGHT_CLUSTERS_PATH,
            "step3_vmatch_matches": vmatch_cfg.STEP3_MATCHES_PATH,
            "step3_vmatch_runs": vmatch_cfg.STEP3_RUNS_PATH,
            "step3_trf_arrays": vmatch_cfg.STEP3_TRF_ARRAYS_PATH,
            "step3_trf_runs": vmatch_cfg.STEP3_TRF_RUNS_PATH,
            "step3_trf_normalization_fingerprint": vmatch_cfg.STEP3_TRF_NORMALIZATION_FINGERPRINT_PATH,
            "step4c_architecture_audit": vmatch_cfg.STEP4_ARCHITECTURE_AUDIT_PATH,
            "step4c_vmatch_suppression_ledger": vmatch_cfg.STEP4_VMATCH_SUPPRESSION_LEDGER_PATH,
            "step5c_copy_matches": vmatch_cfg.STEP5_COPY_MATCHES_PATH,
            "step5c_architecture_recurrence": vmatch_cfg.STEP5_ARCHITECTURE_RECURRENCE_PATH,
            "step5c_evidence_recurrence": vmatch_cfg.STEP5_EVIDENCE_RECURRENCE_PATH,
            "step6c_evidence_recurrence": vmatch_cfg.STEP6_EVIDENCE_RECURRENCE_PATH,
        },
        "samples": {
            "protein_tree_samples": tree_cfg.SAMPLES_PATH,
            "pruned_tree": tree_cfg.PRUNED_TREE_PATHS["core"],
            "pruned_full_tree": tree_cfg.PRUNED_TREE_PATHS["full"],
        },
        "repeat_product": {
            "step4c_architectures": vmatch_cfg.STEP4_ARCHITECTURES_PATH,
            "step4c_evidence": vmatch_cfg.STEP4_EVIDENCE_PATH,
            "step4c_unit_composition": vmatch_cfg.STEP4_UNIT_COMPOSITION_PATH,
            "step5c_copies": vmatch_cfg.STEP5_COPIES_PATH,
            "step5c_architectures": vmatch_cfg.STEP5_ARCHITECTURES_PATH,
            "step5c_protein_summary": vmatch_cfg.STEP5_PROTEIN_SUMMARY_PATH,
            "measured_architectures": vmatch_cfg.STEP5_MEASURED_ARCHITECTURES_PATH,
            "measured_evidence": vmatch_cfg.STEP5_MEASURED_EVIDENCE_PATH,
            "step6c_sample_summary": vmatch_cfg.STEP6_SAMPLE_SUMMARY_PATH,
            "step6c_near_copies": vmatch_cfg.STEP6_NEAR_COPIES_PATH,
            "step6c_unit_measures": vmatch_cfg.STEP6_UNIT_MEASURES_PATH,
        },
        "results": {
            "anchor_validation": domains_cfg.RESULT_ANCHOR_VALIDATION_PATH,
            "interpro_vs_dpam_hth": domains_cfg.RESULT_INTERPRO_VS_DPAM_HTH_PATH,
            "dpam_judge_sensitivity": domains_cfg.RESULT_DPAM_JUDGE_SENSITIVITY_PATH,
            "dpam_family_census": domains_cfg.RESULT_DPAM_FAMILY_CENSUS_PATH,
            **{f"measures_by_{view}": path
               for view, path in vmatch_cfg.RESULT_MEASURE_PATHS.items()},
        },
    }

def _policies():

    return {
        "clustering": {"min_seq_id": domains_cfg.REPRESENTATIVE_CLUSTER_MIN_IDENTITY,
                       "coverage": domains_cfg.REPRESENTATIVE_CLUSTER_COVERAGE,
                       "mmseqs_bin": str(domains_cfg.MMSEQS_BIN),
                       "mmseqs_threads": domains_cfg.MMSEQS_THREADS},
        "domain_vocabulary": {"hth_x_name": domains_cfg.DPAM_HTH_X_NAME,
                              "tgroup_alias": dict(domains_cfg.DPAM_TGROUP_ALIAS),
                              "other_token": domains_cfg.DPAM_OTHER_TOKEN,
                              "judge_order": list(domains_cfg.DPAM_JUDGE_ORDER),
                              "hth_classes": list(domains_cfg.DPAM_HTH_CLASSES)},
        "dpam_annotation": {"batch_order": [batch["name"] for batch in domains_cfg.DPAM_BATCHES]},
        "vmatch_caller": {"min_length": vmatch_cfg.VMATCH_MIN_LENGTH,
                          "max_edit_distance": vmatch_cfg.VMATCH_MAX_EDIT_DISTANCE,
                          "min_identity": vmatch_cfg.VMATCH_MIN_IDENTITY,
                          "max_evalue": vmatch_cfg.VMATCH_MAX_EVALUE,
                          "max_spacer": vmatch_cfg.VMATCH_MAX_SPACER,
                          "within_seed_length": vmatch_cfg.VMATCH_WITHIN_SEED_LENGTH,
                          "within_hxdrop": vmatch_cfg.VMATCH_WITHIN_HXDROP,
                          "within_min_identity": vmatch_cfg.VMATCH_WITHIN_MIN_IDENTITY,
                          "within_max_evalue": vmatch_cfg.VMATCH_WITHIN_MAX_EVALUE,
                          "within_select_max_evalue": vmatch_cfg.VMATCHSELECT_WITHIN_MAX_EVALUE},
        "repeat_builder": {"policy": vmatch_cfg.V5_POLICY, "settings": dict(vmatch_cfg.V5_SETTINGS)},
        "evidence_scoring": {"markov_pseudocount": vmatch_cfg.STEP4_MARKOV_PSEUDOCOUNT,
                             "pair_evalue": "repeat_evalue_markov, two copies, whole-unit first-order Markov background, one formula for direct and inverted pairs",
                             "arm_complexity": "single-strand DUST triplet score of each whole arm"},
        "recurrence": {"similarity_threshold": vmatch_cfg.STEP5_SIMILARITY_THRESHOLD,
                       "min_independent_matches": vmatch_cfg.STEP5_MIN_INDEPENDENT_MATCHES,
                       "min_recurrence_units": vmatch_cfg.STEP5_MIN_RECURRENCE_UNITS,
                       "evidence_piece_match": "a set's counted evidence per unit, overlapping stretches merged into pieces and touching ones kept apart; same-side units of one group; edlib infix identity over the shorter piece, either orientation; two units recur on min_independent_matches one-to-one matches, or on one match whose matched region (edlib's first location, the direct orientation on ties) lies inside array spans on both pieces"},
        "sample_near_copies": {"rule": "Step 2's flank-redundancy rule across a sample's proteins, per side; candidates by length, the sample protein's own flank, Step 2 cluster size, protein and unit ID; never by repeats",
                               "min_prefix_similarity": vmatch_cfg.FLANK_CLUSTER_MIN_PREFIX_SIMILARITY},
        "measures": {"repeat_sets": {repeat_set.name: {"kinds": list(repeat_set.kinds),
                                                       "max_pair_evalue": repeat_set.max_pair_evalue,
                                                       "sources": list(repeat_set.sources)}
                                     for repeat_set in vmatch_cfg.STEP5_REPEAT_SETS},
                     "primary_repeat_set": vmatch_cfg.STEP5_PRIMARY_REPEAT_SET,
                     "min_copy_length": vmatch_cfg.STEP5_MIN_COPY_LENGTH,
                     "proximal_cutoff": vmatch_cfg.STEP5_PROXIMAL_CUTOFF,
                     "near_cds_cutoff": vmatch_cfg.STEP5_NEAR_CDS_CUTOFF,
                     "unit_windows": list(vmatch_cfg.STEP5_UNIT_WINDOWS),
                     "min_units": vmatch_cfg.STEP5_MIN_UNITS},
        "repeat_enrichment": {"cohort": {"rule": "intact: starts with M, both core domains at least complete_share of the shortest anchor call, not an exact piece of a longer roster protein carrying a domain it lacks",
                                         "core_tokens": list(tree_cfg.CORE_TOKENS), "complete_share": tree_cfg.CORE_COMPLETE_SHARE,
                                         "truncated_share": tree_cfg.CORE_TRUNCATED_SHARE},
                              "background": {"panel": enrichment_cfg.BACKGROUND_PANEL,
                                             "genes_per_genome": vmatch_cfg.BACKGROUND_PANELS[enrichment_cfg.BACKGROUND_PANEL]["genes_per_genome"],
                                             "genomes": "RefSeq (GCF) assemblies holding a kept Step 6C unit", "seed": vmatch_cfg.BACKGROUND_SEED},
                              "comparison": {"windows": list(enrichment_cfg.BACKGROUND_WINDOWS), "estimand": "focal outcome minus its genome-and-side background rate, within sample, samples equally",
                                             "interval": "host-genus resampling", "bootstrap": enrichment_cfg.BACKGROUND_BOOTSTRAP,
                                             "gc_bins": enrichment_cfg.BACKGROUND_GC_BINS, "branches": enrichment_cfg.BACKGROUND_BRANCHES},
                              "bayes": {"model": "binomial cells, cloglog; genome effect shared, sample effect on focal flanks; alpha Student-t(3,-3,2.5), class shift N(0,1), side N(0,1), SDs exponential with P(sd>5)=0.05",
                                        "settings": enrichment_cfg.BAYES_SETTINGS}},
    }

@dataclass(frozen=True)
class Digest:

    role: str
    label: str
    path: str
    present: bool
    md5: str
    size_bytes: int
    modified: str

    @classmethod
    def of(cls, role, label, path):

        if not path.exists(): return cls(role, label, str(path), False, "", 0, "")

        digest = hashlib.md5()

        with path.open("rb") as handle:

            for block in iter(lambda: handle.read(1 << 20), b""): digest.update(block)

        stat = path.stat()
        modified = datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")

        return cls(role, label, str(path), True, digest.hexdigest(), stat.st_size, modified)

@dataclass(frozen=True)
class Snapshot:

    name: str
    description: str
    policies: dict
    digests: tuple

    @classmethod
    def capture(cls, name, description):

        digests = tuple(Digest.of(role, label, path)
                        for role, group in _artifacts().items()
                        for label, path in group.items())

        return cls(name, description, _policies(), digests)

    @classmethod
    def read(cls, path):

        payload = json.loads(path.read_text(encoding="utf-8"))
        digests = tuple(Digest(**item) for item in payload["digests"])

        return cls(payload["name"], payload["description"], payload["policies"], digests)

    def write(self, path):

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    def differences(self, other):

        mine = {digest.label: digest for digest in self.digests}
        theirs = {digest.label: digest for digest in other.digests}
        report = []

        for label in sorted(mine.keys() - theirs.keys()): report.append(f"only in {self.name}: {label}")

        for label in sorted(theirs.keys() - mine.keys()): report.append(f"only in {other.name}: {label}")

        for label in sorted(mine.keys() & theirs.keys()):

            before, after = mine[label], theirs[label]

            if before.present != after.present:

                report.append(f"{label}: present={before.present} -> present={after.present}")

            elif before.md5 != after.md5:

                report.append(f"{label}: {before.md5} -> {after.md5} "
                              f"({before.size_bytes:,} -> {after.size_bytes:,} bytes)")

        for topic in sorted(set(self.policies) | set(other.policies)):

            if self.policies.get(topic) != other.policies.get(topic):

                report.append(f"policy {topic}: {self.policies.get(topic)} -> {other.policies.get(topic)}")

        return report

def snapshot_path(name):

    return BASELINES_DIR / name / SNAPSHOT_FILENAME

def capture(name, description, allow_absent=False):

    path = snapshot_path(name)

    if path.exists():

        raise RuntimeError(f"{path} already exists. A snapshot is a fixed record of one moment; "
                           f"capture under a new name instead of overwriting it.")

    recorded = Snapshot.capture(name, description)
    recorded.write(path)
    absent = [digest.label for digest in recorded.digests if not digest.present]
    print(f"Recorded {len(recorded.digests)} artifacts as '{name}' in {path}")

    if absent: print(f"Absent at capture time: {', '.join(absent)}")

    if absent and not allow_absent:

        print("Refusing to treat this as a clean capture; pass --allow-absent to record "
              "a deliberately partial snapshot.")

        return 1

    return 0

def compare(name):

    recorded = Snapshot.read(snapshot_path(name))
    current = Snapshot.capture("current", "the tree as it stands now")
    report = recorded.differences(current)

    if not report:

        print(f"Current tree matches snapshot '{name}' on all {len(recorded.digests)} artifacts and policies.")

        return 0

    print(f"Current tree differs from snapshot '{name}':")

    for line in report: print(f"  {line}")

    return 1

def main(argv=None):

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_capture = sub.add_parser("capture", help="Record the current artifacts and policies under a name.")
    p_capture.add_argument("name")
    p_capture.add_argument("--description", default="")
    p_capture.add_argument("--allow-absent", action="store_true",
                           help="Record the snapshot even though some artifacts are absent.")

    p_compare = sub.add_parser("compare", help="Compare a recorded snapshot against the current tree.")
    p_compare.add_argument("name")

    args = parser.parse_args(argv)

    if args.command == "capture":

        return capture(args.name, args.description, args.allow_absent)

    return compare(args.name)

if __name__ == "__main__":

    raise SystemExit(main())
