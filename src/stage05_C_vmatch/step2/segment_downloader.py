import os
import re
import tempfile
import time
from http.client import IncompleteRead
from io import StringIO
from pathlib import Path
from urllib.error import HTTPError, URLError

from Bio import Entrez, SeqIO

TRANSIENT_FETCH_ERRORS = (IncompleteRead, URLError, ConnectionError, TimeoutError,)
FETCH_ATTEMPTS = 4
FETCH_BACKOFF_SECONDS = 2

def context_key(context):
    strand = "plus" if context["strand"] == "+" else "minus"
    return (
        f"{safe_name(context['nucleotide_accession'])}__"
        f"{int(context['start'])}-{int(context['stop'])}__{strand}"
    )

def genbank_segment_path(download_root, protein_accession, context, flank_length):
    filename = f"{context_key(context)}__flank_{flank_length}.gb"
    return Path(download_root) / protein_accession / filename

def fetch_bounds(context, flank_length):
    return (
        max(1, int(context["start"]) - flank_length),
        int(context["stop"]) + flank_length,
    )

def fetch_genbank_segment(
    protein_accession,
    context,
    download_root,
    flank_length,
    email,
    api_key=None,
):
    path = genbank_segment_path(
        download_root,
        protein_accession,
        context,
        flank_length,
    )
    if path.exists():
        try:
            read_genbank_segment(path)
            return path, False
        except (OSError, ValueError):
            pass

    if not email:
        raise ValueError("Set NCBI_EMAIL before downloading genomic segments.")

    Entrez.email = email
    Entrez.tool = "repo_druid_stage05C_segment_download"
    Entrez.api_key = api_key or None

    fetch_start, fetch_stop = fetch_bounds(context, flank_length)
    payload = _efetch_interval(
        context["nucleotide_accession"],
        fetch_start,
        fetch_stop,
    )

    record = SeqIO.read(StringIO(payload), "genbank")
    if not record.seq:
        raise ValueError(
            f"NCBI returned an empty sequence for "
            f"{context['nucleotide_accession']}:{fetch_start}-{fetch_stop}"
        )

    _write_text_atomic(path, payload)
    return path, True

def _efetch_interval(accession, fetch_start, fetch_stop):

    for attempt in range(1, FETCH_ATTEMPTS + 1):

        try:

            with Entrez.efetch(db="nuccore", id=accession, rettype="gb", retmode="text",
                               strand=1, seq_start=fetch_start, seq_stop=fetch_stop,) as handle:

                payload = handle.read()

            return payload.decode("utf-8") if isinstance(payload, bytes) else payload

        except TRANSIENT_FETCH_ERRORS as error:

            if isinstance(error, HTTPError) and error.code < 500: raise
            if attempt == FETCH_ATTEMPTS: raise

            time.sleep(FETCH_BACKOFF_SECONDS * attempt)

def read_genbank_segment(path):
    record = SeqIO.read(path, "genbank")
    if not record.seq:
        raise ValueError(f"Empty GenBank sequence: {path}")
    return record

def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))

def _write_text_atomic(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
