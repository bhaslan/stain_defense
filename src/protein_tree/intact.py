import pandas as pd
from Bio import SeqIO

from src import roster
from src.protein_tree import config_tree as cfg
from src.stage02_domains import dpam

def classify():

    sequences = roster.sequences()
    architectures = pd.read_csv(cfg.PROTEIN_ARCHITECTURES_PATH).set_index("protein_accession")
    domains = core_domains()

    table = pd.DataFrame({"protein_accession": list(sequences)})
    table["hth_class"] = table["protein_accession"].map(architectures["hth_class"])
    table["architecture"] = table["protein_accession"].map(architectures["architecture"])
    table["length"] = table["protein_accession"].map(lambda accession: len(sequences[accession]))
    table["donor"] = table["protein_accession"].map(architectures["annotation_donor"])
    table = table.join(domains, on="donor")
    table["container"] = table["protein_accession"].map(_containers(sequences, table["protein_accession"]))
    table["container_extra"] = [_extra(own, architectures["architecture"].get(container))
                                for own, container in zip(table["architecture"], table["container"])]

    cuts = core_cuts(domains)
    classed = [_status(row, sequences[row.protein_accession], cuts, row.donor in domains.index) for row in table.itertuples()]
    table["status"] = [status for status, _ in classed]
    table["reasons"] = [";".join(reasons) for _, reasons in classed]

    cfg.PROTEIN_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(cfg.PROTEIN_STATUS_PATH, sep="\t", index=False)

    return table

def core_domains():

    domains = dpam.selected_domains()
    called = pd.Index(domains["protein_accession"].unique())
    core = domains.loc[domains["token"].isin(cfg.CORE_TOKENS)]
    lengths = core.groupby(["protein_accession", "token"])["domain_length"].sum().unstack()

    return lengths.reindex(index=called, columns=list(cfg.CORE_TOKENS))

def core_cuts(domains):

    anchors = [record.id for record in SeqIO.parse(cfg.ANCHOR_FASTA_PATH, "fasta")]
    shortest = domains.loc[anchors].min()

    if shortest.isna().any(): raise ValueError(f"An anchor lacks a core domain call: {shortest[shortest.isna()].index.tolist()}")

    return {"complete": shortest * cfg.CORE_COMPLETE_SHARE, "truncated": shortest * cfg.CORE_TRUNCATED_SHARE}

def _containers(sequences, accessions):

    by_length = sorted(sequences, key=lambda accession: len(sequences[accession]), reverse=True)
    found = {}

    for accession in accessions:

        piece = sequences[accession]
        found[accession] = next((other for other in by_length if len(sequences[other]) > len(piece) and piece in sequences[other]), None)

    return found

def _extra(own, container):

    if container is None: return ""

    return ";".join(sorted(set(container.split(" -- ")) - set(own.split(" -- "))))

def _status(row, sequence, cuts, called):

    reasons = [] if sequence.startswith("M") else ["no_start_m"]

    if called:

        for token in cfg.CORE_TOKENS:

            length = getattr(row, token)
            if pd.isna(length): reasons.append(f"{token}_absent")
            elif length < cuts["truncated"][token]: reasons.append(f"{token}_truncated")

    if row.container_extra: reasons.append(f"piece_lacking_{row.container_extra.replace(';', '_')}")
    if reasons: return "suspect", reasons
    if not called: return "unresolved", ["no_domain_call"]

    short = [f"{token}_short" for token in cfg.CORE_TOKENS if getattr(row, token) < cuts["complete"][token]]

    return ("unresolved", short) if short else ("intact", [])

def summary(table):

    lines = []

    for hth_class, rows in table.groupby("hth_class"):

        lines.append(f"{hth_class}: {len(rows):,} proteins; {rows['status'].value_counts().to_dict()}")

        for status, group in rows.groupby("status"):

            quartiles = group["length"].quantile([0.25, 0.5, 0.75]).astype(int).tolist()
            lines.append(f"  {status}: {len(group):,} proteins, length quartiles {quartiles}")

        reasons = rows["reasons"].str.split(";").explode()
        lines.append(f"  reasons: {reasons[reasons != ''].value_counts().to_dict()}")

    return lines
