from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping

from src.stage05_C_vmatch.step2.download_extract import redundant_flank_distance
from src.stage05_C_vmatch.step5c.models import Opportunity

NEAR_COPY_FIELDS = ("sample_accession", "side", "representative_unit_id", "protein_accession",
                    "kept", "kept_unit_id", "prefix_edit_distance")

def select(units: Iterable[Opportunity],
           sample_of: Mapping[str, str],
           sequences: Mapping[tuple[str, str, str], str]) -> tuple[list[Opportunity], list[dict[str, object]]]:

    pooled = defaultdict(list)

    for unit in units:
        if unit.protein_accession in sample_of: pooled[(sample_of[unit.protein_accession], unit.database_side)].append(unit)

    kept, rows = [], []

    for (sample, side), candidates in sorted(pooled.items()):

        order = sorted(candidates, key=lambda unit: (-unit.side_length(side), unit.protein_accession != sample,
                                                     -unit.represented_context_count, unit.protein_accession,
                                                     unit.representative_unit_id))
        copy_of: dict[str, tuple[str, int]] = {}

        for representative in order:

            if representative.representative_unit_id in copy_of: continue

            copy_of[representative.representative_unit_id] = (representative.representative_unit_id, 0)
            kept.append(representative)
            flank = sequences[representative.upstream_key]

            for member in order:

                if member.representative_unit_id in copy_of or member.protein_accession == representative.protein_accession:

                    continue

                distance = redundant_flank_distance(sequences[member.upstream_key], flank)
                if distance is not None: copy_of[member.representative_unit_id] = (representative.representative_unit_id, distance)

        for unit in order:

            kept_unit_id, distance = copy_of[unit.representative_unit_id]
            rows.append({"sample_accession": sample, "side": side, "representative_unit_id": unit.representative_unit_id,
                         "protein_accession": unit.protein_accession, "kept": kept_unit_id == unit.representative_unit_id,
                         "kept_unit_id": kept_unit_id, "prefix_edit_distance": distance})

    return kept, rows
