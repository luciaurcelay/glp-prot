"""Single config schema for GLP-PLM.

One dataclass tree, loaded from YAML. `poc_mac.yaml` and `esm8m_h100.yaml` differ
only in device / precision / scale / buffer fields (PLAN §5 interface contract, D8).

Width rule (PLAN §6.2, verified in docs/INTERPLM_MAP.md against
`generative_latent_prior/configs/train_llama8b_static.yaml`):
    d_model = 2 * activation_dim
    d_mlp   = 4 * activation_dim
If `d_model` / `d_mlp` are left null in YAML they are derived from `activation_dim`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, List, Optional


@dataclass
class ModelConfig:
    # activation_dim = ESM hidden size at the hooked layer (320 for esm2-8m).
    activation_dim: int = 320
    # derived from activation_dim when None (2x / 4x, docs/INTERPLM_MAP.md)
    d_model: Optional[int] = None
    d_mlp: Optional[int] = None
    n_blocks: int = 2  # PoC 2; spine 4->6 (PLAN §6.2 table)

    def resolve(self) -> "ModelConfig":
        if self.d_model is None:
            self.d_model = 2 * self.activation_dim
        if self.d_mlp is None:
            self.d_mlp = 4 * self.activation_dim
        return self


@dataclass
class DiffusionConfig:
    # t in [0,1]; t=1 is pure noise, t=0 is clean (PLAN §2.1/§6.3).
    num_train_timesteps: int = 1000


@dataclass
class SamplerConfig:
    num_steps: int = 20          # steering cleanup & PCA sanity (PLAN §6.3)
    # Final FD needs far more steps: measured on the 6-block/10M local run,
    # FD ratio to the lower bound is 2.81x @20 steps, 1.74x @100, 1.55x @1000 —
    # i.e. 20 steps reports sampler error, not model quality (PLAN §6.4).
    fd_num_steps: int = 1000
    t_start: float = 1.0         # 1.0 = generate-from-noise; 0.5 = SDEdit/loss-recovered


@dataclass
class DataConfig:
    # Which PLM + hook point. layer=4 = hidden_states[4] = residual after block 4
    # (docs/INTERPLM_MAP.md). Byte-identical to what the released SAE saw.
    model_name: str = "facebook/esm2_t6_8M_UR50D"
    layer: int = 4
    # Activation budget (PLAN §6.1). PoC ~2M; spine 50->300M streamed.
    n_train_activations: int = 2_000_000
    # "precache" (Mac) writes memmap shards; "stream" (cluster) uses producer-consumer.
    store_mode: str = "precache"
    activation_dtype: str = "float16"   # store format (PLAN §6.1)
    # Training source: null -> Swiss-Prot REST slice; "uniref50" -> shared local pull
    # (scripts/prefetch_data.py --with-uniref50); any other string -> that path.
    train_fasta: Optional[str] = None
    # Reuse an EXISTING activation store instead of the per-run <output_dir>/train_store,
    # so many runs (e.g. a depth/scaling grid over identical activations) share one 320GB
    # store with no re-extraction. Absolute path used as-is; a relative path resolves under
    # $GLP_DATA_ROOT (e.g. "uniref50_500m/train_store"). null -> per-run default.
    # Only train_glp/eval_generation READ it; extract_activations always writes the per-run
    # store (it never overwrites a shared one). CLI --store-dir overrides this.
    store_dir: Optional[str] = None
    # Same idea for the labeled Swiss-Prot eval store (embedded shards + labels +
    # frozen_concepts.json): it depends only on model+layer+n_prot, not the GLP run, so
    # many runs share one prebuilt `.../swissprot/processed`. Absolute path as-is; relative
    # resolves under $GLP_DATA_ROOT (e.g. "uniref50_100m/swissprot/processed"). Only
    # freeze_concepts/run_probing READ it; build_swissprot_eval always writes the per-run
    # store. CLI --eval-dir overrides this. null -> per-run default.
    eval_dir: Optional[str] = None
    eval_uniprot_tsv: Optional[str] = None
    # Swiss-Prot eval split by shard ranges (docs/INTERPLM_MAP.md, prepare_eval_set).
    n_shards: int = 8
    valid_shard_range: List[int] = field(default_factory=lambda: [0, 3])
    test_shard_range: List[int] = field(default_factory=lambda: [4, 7])
    min_aa_per_concept: int = 1500       # docs/INTERPLM_MAP.md: real default is 1500/25, not 10
    min_domains_per_concept: int = 25


@dataclass
class TrainConfig:
    lr: float = 5e-5              # d=4096 prior; re-tune at d=320 (PLAN §6.2, R10)
    batch_size: int = 4096
    num_epochs: int = 1
    warmup_ratio: float = 0.01
    lr_initial_factor: float = 0.01
    lr_final_factor: float = 0.1
    grad_clip: float = 1.0
    log_every_n_steps: int = 10


@dataclass
class ProbingConfig:
    # Probing time lives in `u` (schedule fraction), released default 0.9 (docs/INTERPLM_MAP.md).
    # Select on val, report on test (D9): search this grid.
    u_grid: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    topk: int = 512
    seed: int = 42


@dataclass
class Config:
    # ---- fields that DIFFER between poc_mac and h100 (D8) ----
    device: str = "cpu"          # "mps" / "cpu" (PoC) ; "cuda" (spine)
    precision: str = "fp32"      # "fp32" (MPS bf16 flaky) ; "bf16" (cuda)
    seed: int = 0
    output_dir: str = "runs/poc"
    # ---- shared model / method definition ----
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    probing: ProbingConfig = field(default_factory=ProbingConfig)

    def resolve(self) -> "Config":
        self.model.resolve()
        return self

    def check_supported(self) -> None:
        """Fail loudly on config knobs that are declared but NOT yet implemented.

        `store_mode: stream` and `precision: bf16` appear in the cluster config, but the
        streaming producer-consumer and mixed precision are still W2 work. Silently
        ignoring them would mean a 300M-activation run tries to pre-cache ~192 GB, and
        a "bf16" run quietly executes in fp32 — so raise instead.
        """
        if self.data.store_mode not in ("precache",):
            raise NotImplementedError(
                f"data.store_mode={self.data.store_mode!r} is not implemented yet "
                "(only 'precache'). Streaming is a W2 task — see "
                "glp_plm/data/producer_consumer.py and PLAN §6.1."
            )
        if self.precision not in ("fp32",):
            raise NotImplementedError(
                f"precision={self.precision!r} is not implemented yet (only 'fp32'). "
                "Mixed precision is a W2 task — see PLAN §6.2."
            )


# ------------------------------------------------------------------ #
#  YAML <-> dataclass (no omegaconf dependency; stdlib + pyyaml only) #
# ------------------------------------------------------------------ #
def _from_dict(cls, data: Any):
    if not is_dataclass(cls) or data is None:
        return data
    kwargs = {}
    type_by_name = {f.name: f.type for f in fields(cls)}
    for key, val in data.items():
        if key not in type_by_name:
            raise KeyError(f"Unknown config key '{key}' for {cls.__name__}")
        ftype = type_by_name[key]
        # nested dataclass?
        nested = {
            "ModelConfig": ModelConfig, "DataConfig": DataConfig,
            "DiffusionConfig": DiffusionConfig, "SamplerConfig": SamplerConfig,
            "TrainConfig": TrainConfig, "ProbingConfig": ProbingConfig,
        }
        tname = ftype if isinstance(ftype, str) else getattr(ftype, "__name__", "")
        if tname in nested:
            kwargs[key] = _from_dict(nested[tname], val)
        else:
            kwargs[key] = val
    return cls(**kwargs)


def load_config(path: str | Path) -> Config:
    """Load a YAML file into a resolved Config."""
    import yaml  # lazy: dataclasses usable without pyyaml installed

    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return _from_dict(Config, raw).resolve()
