import json
from functools import cached_property
from urllib.parse import unquote

import pandas as pd
from Bio import SeqIO

from src.genomes import config_genomes as cfg

class Assembly:

    def __init__(self, accession):

        self.accession = accession
        self.directory = cfg.GENOMES_DIR / accession

    def path(self, label): return self.directory / cfg.FILES[label][0]

    @cached_property
    def sequences(self):

        return {record.id: str(record.seq).upper() for record in SeqIO.parse(self.path("fna"), "fasta")}

    @cached_property
    def names(self):

        names = {name: name for name in self.sequences}

        with open(self.path("sequence_report")) as handle:

            for line in handle:

                report = json.loads(line)
                fasta_name = next((report[key] for key in ("refseqAccession", "genbankAccession") if report.get(key) in self.sequences), None)
                if fasta_name is None: continue

                for key in ("refseqAccession", "genbankAccession"):
                    if report.get(key): names[report[key]] = fasta_name

        return names

    def sequence(self, name): return self.sequences[self.names[name]]

    @cached_property
    def cds(self):

        rows = []

        with open(self.path("gff")) as handle:

            for line in handle:

                if line.startswith("#"): continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 9 or fields[2] != "CDS": continue
                attributes = dict(item.split("=", 1) for item in fields[8].split(";") if "=" in item)
                rows.append({"cds_id": attributes["ID"], "seqid": fields[0], "start": int(fields[3]), "end": int(fields[4]),
                             "strand": fields[6], "protein_id": unquote(attributes.get("protein_id", "")),
                             "pseudo": attributes.get("pseudo") == "true", "partial": attributes.get("partial") == "true",
                             "start_fuzzy": "start_range" in attributes, "end_fuzzy": "end_range" in attributes})

        parts = pd.DataFrame(rows)
        joined = parts.groupby("cds_id", sort=False).agg(seqid=("seqid", "first"), start=("start", "min"), end=("end", "max"),
                                                          strand=("strand", "first"), protein_id=("protein_id", "first"),
                                                          pseudo=("pseudo", "any"), partial=("partial", "any"),
                                                          start_fuzzy=("start_fuzzy", "any"), end_fuzzy=("end_fuzzy", "any"),
                                                          parts=("start", "size"))

        return joined.reset_index()

    @cached_property
    def proteins(self):

        return {record.id: str(record.seq) for record in SeqIO.parse(self.path("faa"), "fasta")}
