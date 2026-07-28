#!/usr/bin/env python
"""Download InterPLM's released SAE weights from HuggingFace (PLAN §11, W1).

Reuses `interplm.sae.inference.load_sae_from_hf` (docs/INTERPLM_MAP.md) — we do NOT retrain SAEs
(D6). Default: layer 4 (InterPLM's primary analysis layer). Pass --all-layers to cache
1-6. Prints dict_size / activation_dim as a sanity check (expect 10,240 / 320).
"""
import argparse

import _bootstrap  # noqa: F401  (puts repo root + InterPLM on sys.path)
from glp_plm.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--all-layers", action="store_true",
                    help="cache layers 1-6 (default: just the config's layer)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    from interplm.sae.inference import INTERPLM_HF_LAYERS, load_sae_from_hf

    plm_model = "esm2-8m" if "8M" in cfg.data.model_name else "esm2-650m"
    layers = INTERPLM_HF_LAYERS[plm_model] if args.all_layers else [cfg.data.layer]

    for layer in layers:
        sae = load_sae_from_hf(plm_model, layer)
        print(f"[download_saes] {plm_model} layer {layer}: "
              f"dict_size={sae.dict_size} activation_dim={sae.activation_dim}")
    print(f"[download_saes] cached {len(layers)} SAE(s) for {plm_model}.")


if __name__ == "__main__":
    main()
