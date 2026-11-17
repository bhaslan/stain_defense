import argparse
from concurrent.futures import ThreadPoolExecutor

from src.stage06_repeat_enrichment import config_enrichment as cfg
from src.stage06_repeat_enrichment import background, kinship, lineages

def step2_build_kinship():

    for tree in cfg.KINSHIP_TREES: kinship.build_kinship(tree)

def _report(name, lines):

    print(f"Stage 06, {name}:")
    for line in lines: print(f"  {line}")

def background_compare(panel):

    flanks = background.focal_flanks(panel)
    _report("focal flanks", background.entered(flanks))
    table = background.compare(flanks, panel)
    print(table.to_string(index=False))
    print(background.panel_path(cfg.BACKGROUND_COMPARISON_PATH, panel))

def background_branches():

    print(background.by_branch(background.read_flanks()).to_string(index=False))
    print(cfg.BACKGROUND_BRANCHES_PATH)

def background_families():

    print(background.by_family(background.read_flanks()).to_string(index=False))
    print(cfg.FAMILY_COMPARISON_PATH)

def background_family_models():

    for out in background.fit_family_models(): print(out, flush=True)
    print(background.family_model_table().to_string(index=False))
    print(cfg.FAMILY_MODEL_TABLE_PATH)

def background_sides(panel):

    print(background.by_side(background.read_flanks(panel), panel).to_string(index=False))
    print(background.panel_path(cfg.BACKGROUND_SIDES_PATH, panel))

def background_unit_sizes():

    classes, _ = background.by_unit_size(background.read_flanks())

    print(classes.to_string(index=False))
    print(cfg.UNIT_SIZE_PATH)
    print(cfg.UNIT_SPECTRUM_PATH)

def background_gc():

    overall, _ = background.flank_gc(background.read_flanks())

    print(overall.to_string(index=False))
    print(cfg.FLANK_GC_PATH)
    print(cfg.FLANK_GC_GAPS_PATH)

def repeat_lineages():

    all_or_none, follows = lineages.lineages(background.read_flanks())

    print(all_or_none.to_string(index=False))
    print(follows.to_string(index=False))
    print(cfg.ALL_OR_NONE_PATH)
    print(cfg.FOLLOWS_PATH)

def background_terminators(panel):

    print(background.terminator_check(panel).to_string(index=False))
    print(background.panel_path(cfg.TERMINATOR_CHECK_PATH, panel))

def background_models(panel):

    with ThreadPoolExecutor(len(cfg.BACKGROUND_MODEL_FITS)) as pool:

        for out in pool.map(lambda fit: background.fit_models(panel, *fit), cfg.BACKGROUND_MODEL_FITS): print(out, flush=True)

def background_table():

    table = background.model_table()
    print(table.to_string(index=False))
    print(cfg.BACKGROUND_MODEL_TABLE_PATH)

def main(argv=None):

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("step2_build_kinship").set_defaults(func=step2_build_kinship)

    parser_compare = sub.add_parser("background_compare")
    parser_compare.add_argument("--panel", default=cfg.BACKGROUND_PANEL, help="Stage 05C's background panel: full, or genbank for the check")
    parser_compare.set_defaults(func=background_compare)
    sub.add_parser("background_branches").set_defaults(func=background_branches)
    sub.add_parser("background_families").set_defaults(func=background_families)
    sub.add_parser("background_family_models").set_defaults(func=background_family_models)
    parser_models = sub.add_parser("background_models")
    parser_models.add_argument("--panel", default=cfg.BACKGROUND_PANEL, help="Stage 05C's background panel: full, or genbank for the check")
    parser_models.set_defaults(func=background_models)
    parser_sides = sub.add_parser("background_sides")
    parser_sides.add_argument("--panel", default=cfg.BACKGROUND_PANEL, help="Stage 05C's background panel: full, genbank for the check, or pooled for both")
    parser_sides.set_defaults(func=background_sides)
    parser_terminators = sub.add_parser("background_terminators")
    parser_terminators.add_argument("--panel", default=cfg.BACKGROUND_PANEL, help="Stage 05C's background panel: full, or genbank for the check")
    parser_terminators.set_defaults(func=background_terminators)
    sub.add_parser("background_table").set_defaults(func=background_table)
    sub.add_parser("background_unit_sizes").set_defaults(func=background_unit_sizes)
    sub.add_parser("background_gc").set_defaults(func=background_gc)
    sub.add_parser("repeat_lineages").set_defaults(func=repeat_lineages)

    args = parser.parse_args(argv)

    keywords = {name: value for name, value in vars(args).items() if name not in ("command", "func")}
    args.func(**keywords)

if __name__ == "__main__":

    main()
