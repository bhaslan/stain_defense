import argparse

from src.stage07_figures import concordance, enrichment, esmc_distance, evidence_sweep, flank_table, homolog_table, hth_count, repeat_character, repeat_recurrence, repeat_size, sae, side_profile
from src.stage07_figures.tree import recipe

FIGURES = {"tree": recipe.draw, "esmc_distance": esmc_distance.draw, "enrichment": enrichment.draw,
           "family_enrichment": enrichment.draw_families, "side_profile": side_profile.draw, "repeat_character": repeat_character.draw, "repeat_size": repeat_size.draw,
           "evidence_sweep": evidence_sweep.draw, "repeat_recurrence": repeat_recurrence.draw,
           "concordance": concordance.draw, "sae": sae.draw,
           "hth_count": hth_count.draw, "homolog_table": homolog_table.draw,
           "flank_table": flank_table.draw}

def main(argv=None):

    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("figure", choices=[*FIGURES, "all"])
    args = parser.parse_args(argv)
    chosen = FIGURES if args.figure == "all" else {args.figure: FIGURES[args.figure]}

    for draw in chosen.values():

        for path in draw(): print(path)

if __name__ == "__main__":

    main()
