#!/usr/bin/env python
"""auto_interp_metaneurons — thin CLI stub (PLAN §11).

Not implemented this session: the data/eval pipeline is gated behind RECON.md
review (PLAN §9). See RECON.md §9-10 for the verified InterPLM entrypoints this
will wrap. Kept as a stub so the repo skeleton matches PLAN §5/§11 exactly.
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.parse_args()
    raise NotImplementedError(
        "auto_interp_metaneurons is scaffolded but not implemented yet (post-RECON-review; PLAN §9)."
    )


if __name__ == "__main__":
    main()
