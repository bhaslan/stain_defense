from __future__ import annotations

import argparse
import logging
import sys

from src.stage05_C_vmatch import config_vmatch as cfg

def _logger() -> logging.Logger:
    logger = logging.getLogger("stage05c.step5c")
    if logger.handlers:
        return logger
    cfg.STEP5_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    concise = logging.StreamHandler(sys.stdout)
    concise.setLevel(logging.INFO)
    detailed = logging.FileHandler(cfg.STEP5_LOG_PATH, encoding="utf-8")
    detailed.setLevel(logging.DEBUG)
    concise.setFormatter(logging.Formatter("%(message)s"))
    detailed.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    logger.addHandler(concise)
    logger.addHandler(detailed)
    return logger

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "summarize",
        help="write the observed Step 5C tables, then the protein measures",
    )
    subparsers.add_parser(
        "results",
        help="write the terminal result_ views of the measures, every repeat set and unit",
    )
    return parser.parse_args()

def main() -> None:
    args = _parse_args()
    logger = _logger()
    try:
        if args.command == "summarize":
            from src.stage05_C_vmatch.step5c.measures import write_measures
            from src.stage05_C_vmatch.step5c.observed import (
                run_observed_analysis,
            )

            write_measures(run_observed_analysis(logger), logger)
        else:
            from src.stage05_C_vmatch.step5c.results import run_results

            run_results(logger)
    except Exception:
        logger.exception("Step 5C %s failed", args.command)
        raise

if __name__ == "__main__":
    main()
