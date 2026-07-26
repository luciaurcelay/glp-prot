#!/usr/bin/env python
"""run_steering — thin CLI stub (PLAN §11).

Not implemented yet (PLAN §9). See docs/INTERPLM_MAP.md for the InterPLM
entrypoints this will wrap. Kept as a stub so the repo skeleton matches PLAN §5/§11.
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.parse_args()
    raise NotImplementedError(
        "run_steering is scaffolded but not implemented yet (PLAN §9)."
    )


if __name__ == "__main__":
    main()
