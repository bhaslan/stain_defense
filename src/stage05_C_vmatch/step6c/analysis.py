from __future__ import annotations

import argparse
import logging
import sys

from src.stage05_C_vmatch import config_vmatch as cfg

def _logger() -> logging.Logger:
    logger = logging.getLogger("stage05c.step6c")
    if logger.handlers:
        return logger
    cfg.STEP6_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    concise = logging.StreamHandler(sys.stdout)
    concise.setLevel(logging.INFO)
    detailed = logging.FileHandler(cfg.STEP6_LOG_PATH, encoding="utf-8")
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
        help="write the sample summary: Step 5C's measures over each sample's pooled units",
    )
    return parser.parse_args()

def main() -> None:
    args = _parse_args()
    logger = _logger()
    try:
        from src.stage05_C_vmatch.step6c.pooling import write_sample_summary

        write_sample_summary(logger)
    except Exception:
        logger.exception("Step 6C %s failed", args.command)
        raise

if __name__ == "__main__":
    main()
