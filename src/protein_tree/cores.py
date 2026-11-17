from Bio import SeqIO

from src import roster
from src.protein_tree import config_tree as cfg
from src.stage02_domains import dpam

SOURCES = {"representatives": cfg.REPRESENTATIVE_FASTA_PATH, "anchors": cfg.ANCHOR_FASTA_PATH}

def members():

    domains = dpam.selected_domains()
    tokens = [partition.token for partition in cfg.CORE_PARTITIONS]
    cored = set(domains.loc[domains["token"].isin(tokens), "protein_accession"])
    listed = set().union(*({record.id for record in SeqIO.parse(path, "fasta")} for path in SOURCES.values()))

    return sorted(listed & cored)

def source_membership(sequences):

    membership = {}

    for source, path in SOURCES.items():

        seen = set()

        for record in SeqIO.parse(path, "fasta"):

            if record.id in seen: raise ValueError(f"Duplicate {record.id} in {path}")

            if record.id in sequences and str(record.seq).upper() != sequences[record.id]:

                raise ValueError(f"Source FASTA sequence for {record.id} differs from the study roster")

            seen.add(record.id)
            membership.setdefault(record.id, []).append(source)

    return membership

def _segments(range_text):

    return [tuple(int(bound) for bound in segment.split("-")) for segment in str(range_text).split(",")]

def subsequence(sequence, range_text):

    return "".join(sequence[start - 1:end] for start, end in _segments(range_text))

def partition_sequences(domains, partition, accessions):

    sequences = roster.sequences()
    rows = domains.loc[(domains["token"] == partition.token)
                       & (domains["protein_accession"].isin(accessions))]
    duplicated = sorted(rows.loc[rows["protein_accession"].duplicated(), "protein_accession"])

    if duplicated:

        raise RuntimeError(f"{partition.key} is called more than once in {len(duplicated):,} proteins "
                           f"({', '.join(duplicated[:5])}), so no single region is the core.")

    cut = {}

    for row in rows.itertuples():

        residues = subsequence(sequences[row.protein_accession], row.Range)

        if len(residues) != row.domain_length:

            raise RuntimeError(f"{partition.key} cut from {row.protein_accession} is {len(residues):,} "
                               f"residues but the domain table reports {row.domain_length:,}. The Range "
                               f"convention or the roster sequence has changed.")

        cut[row.protein_accession] = residues

    return cut
