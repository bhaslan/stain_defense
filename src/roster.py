import argparse
import hashlib

import pandas as pd
from Bio import SeqIO

from src import config_main as cfg

ROSTER_FIELDS = ("protein_accession", "sequence_md5", "sequence_length", "source", "provenance")

def sequence_digest(sequence):

    return hashlib.md5(sequence.encode("ascii")).hexdigest()

def _fasta_sequences(path):

    sequences = {}

    for record in SeqIO.parse(path, "fasta"):

        sequence = str(record.seq).upper()
        previous = sequences.setdefault(record.id, sequence)

        if previous != sequence:

            raise ValueError(f"Conflicting protein FASTA records for {record.id} in {path}")

    return sequences

def read():

    if not cfg.STUDY_ROSTER_PATH.exists():

        raise FileNotFoundError(f"Missing {cfg.STUDY_ROSTER_PATH}. Seed it once with: "
                                f"python -m src.roster seed-from-concordance")

    frame = pd.read_csv(cfg.STUDY_ROSTER_PATH, sep="\t", dtype=str)
    duplicated = frame["protein_accession"].duplicated()

    if duplicated.any():

        offenders = ", ".join(frame.loc[duplicated, "protein_accession"].head(10))

        raise ValueError(f"Study roster lists {int(duplicated.sum()):,} accessions twice: {offenders}")

    return frame

def accessions():

    return set(read()["protein_accession"])

def sequences():

    frame = read()
    available = _fasta_sequences(cfg.STUDY_SEQUENCE_FASTA)
    missing = sorted(set(frame["protein_accession"]) - set(available))

    if missing:

        preview = ", ".join(missing[:10])
        suffix = "" if len(missing) <= 10 else f" and {len(missing) - 10:,} more"

        raise ValueError(f"{cfg.STUDY_SEQUENCE_FASTA} lacks {len(missing):,} roster proteins: "
                         f"{preview}{suffix}")

    selected = {row.protein_accession: available[row.protein_accession]
                for row in frame.itertuples()}
    altered = [row.protein_accession for row in frame.itertuples()
               if sequence_digest(selected[row.protein_accession]) != row.sequence_md5]

    if altered:

        preview = ", ".join(altered[:10])
        suffix = "" if len(altered) <= 10 else f" and {len(altered) - 10:,} more"

        raise ValueError(f"{len(altered):,} roster sequences no longer match their recorded digest: "
                         f"{preview}{suffix}. The roster pins the study's sequences; a changed "
                         f"sequence is a new input and needs an explicit roster decision.")

    return selected

def seed_from_concordance():

    from src.stage03_concordance import config_concordance as concordance_cfg

    if cfg.STUDY_ROSTER_PATH.exists():

        raise RuntimeError(f"{cfg.STUDY_ROSTER_PATH} already exists. Seeding is a one-time migration; "
                           f"edit or extend the roster deliberately rather than reseeding it.")

    source_path = concordance_cfg.CONCORDANCE_PATH
    historical = pd.read_csv(source_path, sep="\t", usecols=["seqid"], dtype=str)
    available = _fasta_sequences(cfg.STUDY_SEQUENCE_FASTA)
    provenance = f"{source_path.name} md5={_file_digest(source_path)}"
    rows = []

    for accession in sorted(set(historical["seqid"])):

        if accession not in available:

            raise ValueError(f"{cfg.STUDY_SEQUENCE_FASTA} lacks historical member {accession}")

        sequence = available[accession]
        rows.append({"protein_accession": accession,
                     "sequence_md5": sequence_digest(sequence),
                     "sequence_length": len(sequence),
                     "source": "legacy_migration",
                     "provenance": provenance})

    frame = pd.DataFrame(rows, columns=list(ROSTER_FIELDS))
    cfg.STUDY_ROSTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cfg.STUDY_ROSTER_PATH, sep="\t", index=False)

    print(f"Study roster seeded with {len(frame):,} proteins from {source_path.name}")
    print(cfg.STUDY_ROSTER_PATH)

def _file_digest(path):

    digest = hashlib.md5()

    with path.open("rb") as handle:

        for block in iter(lambda: handle.read(1 << 20), b""): digest.update(block)

    return digest.hexdigest()

def verify():

    frame = read()
    selected = sequences()
    extra = len(_fasta_sequences(cfg.STUDY_SEQUENCE_FASTA)) - len(selected)

    print(f"Study roster: {len(frame):,} proteins, every sequence present and matching its digest")
    print(f"{cfg.STUDY_SEQUENCE_FASTA.name} holds {extra:,} further records not in the study")
    print(f"Sources: {', '.join(f'{name} {count:,}' for name, count in frame['source'].value_counts().items())}")

def main(argv=None):

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_seed = sub.add_parser("seed-from-concordance",
                            help="One-time migration: recover historical membership into the roster.")
    p_seed.set_defaults(func=seed_from_concordance)

    p_verify = sub.add_parser("verify", help="Check every roster protein against the preprocessing FASTA.")
    p_verify.set_defaults(func=verify)

    args = parser.parse_args(argv)
    args.func()

if __name__ == "__main__":

    main()
