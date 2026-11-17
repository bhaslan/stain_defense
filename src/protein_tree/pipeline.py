import argparse
import json

from src.protein_tree import config_tree as cfg
from src.protein_tree import handoff, intact, relatives, samples

def classify():

    lines = intact.summary(intact.classify())

    for line in lines: print(line)
    print(cfg.PROTEIN_STATUS_PATH)

def select_relatives():

    table = relatives.select()

    print(table.drop(columns="sequence").to_string(index=False))
    print(cfg.RELATIVES_PATH)

def export():

    handoff.export()

def import_tree(tree, bundle_dir, treefile):

    handoff.import_tree(tree, bundle_dir, treefile)

def write_samples():

    manifest = samples.write_samples()

    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(cfg.SAMPLES_PATH)

def prune():

    manifest = samples.prune()

    print(json.dumps(manifest["trees"], indent=2, sort_keys=True))

def main(argv=None):

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_classify = sub.add_parser("classify")
    p_classify.set_defaults(func=classify)

    p_relatives = sub.add_parser("relatives")
    p_relatives.set_defaults(func=select_relatives)

    p_export = sub.add_parser("export")
    p_export.set_defaults(func=export)

    p_import = sub.add_parser("import_tree")
    p_import.add_argument("--tree", required=True, choices=cfg.TREES)
    p_import.add_argument("--bundle-dir", required=True, dest="bundle_dir")
    p_import.add_argument("--treefile", required=True, help="the tree's path inside the returned folder")
    p_import.set_defaults(func=import_tree)

    p_samples = sub.add_parser("samples")
    p_samples.set_defaults(func=write_samples)

    p_prune = sub.add_parser("prune")
    p_prune.set_defaults(func=prune)

    args = parser.parse_args(argv)

    keywords = {name: value for name, value in vars(args).items() if name not in ("command", "func")}
    args.func(**keywords)

if __name__ == "__main__":

    main()
