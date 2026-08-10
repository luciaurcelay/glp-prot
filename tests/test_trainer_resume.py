"""Checkpoint/resume guards for train.trainer.fit (the feature that lets a
time-capped cluster run be resubmitted and continue instead of restarting).

Kept synthetic and tiny (small model, random activations, CPU) so it's a fast unit
test with no weights/store dependency — same spirit as test_flow_matching_roundtrip.
"""

import torch

from glp_plm.config import ModelConfig
from glp_plm.model import GLPDenoiser
from glp_plm.train.trainer import fit

FIT_KW = dict(  # small, fixed schedule shared across a "kill" and its "resume"
    lr=1e-3, batch_size=16, num_epochs=1, warmup_ratio=0.1,
    lr_initial_factor=0.01, lr_final_factor=0.1, grad_clip=1.0,
    device="cpu", log_every_n_steps=1000, seed=0, logger=lambda *a, **k: None,
)


def _tiny_model():
    return GLPDenoiser(ModelConfig(activation_dim=8, d_model=16, d_mlp=16, n_blocks=1).resolve())


def _acts():
    torch.manual_seed(0)
    return torch.randn(16 * 20, 8)  # 20 steps/epoch


def test_resume_continues_from_checkpoint_step(tmp_path):
    ckpt = tmp_path / "glp_ckpt.pt"
    acts = _acts()

    # ckpt_every=6 over a 20-step run leaves the last checkpoint at step 18 (not the
    # end), so the resume has a real non-zero tail to run — with an evenly-dividing
    # cadence the "resume" would trivially run 0 steps and prove nothing.
    fit(_tiny_model(), acts, ckpt_path=ckpt, ckpt_every=6, resume=False, **FIT_KW)
    assert ckpt.exists(), "no periodic checkpoint written"

    state = torch.load(ckpt, map_location="cpu")
    assert state["step"] % 6 == 0 and 0 < state["step"] < state["total_steps"]
    assert "total_steps" in state, "total_steps must be recorded for the resume guard"

    # Resuming must not re-run from 0: only the remaining steps should execute/log.
    resumed_from = state["step"]
    model = _tiny_model()
    model.load_state_dict(state["model"])
    hist = fit(model, acts, ckpt_path=ckpt, ckpt_every=6, resume=True, **FIT_KW)
    tail = state["total_steps"] - resumed_from
    assert 0 < tail < state["total_steps"], "test should exercise a partial resume"
    assert len(hist["loss"]) == tail


def test_resume_refuses_on_total_steps_mismatch(tmp_path):
    ckpt = tmp_path / "glp_ckpt.pt"
    fit(_tiny_model(), _acts(), ckpt_path=ckpt, ckpt_every=5, resume=False, **FIT_KW)

    # A different activation count -> different total_steps -> a resume would run the
    # LR schedule against a mismatched total. Must be refused, not silently wrong.
    more_acts = torch.randn(16 * 40, 8)
    try:
        fit(_tiny_model(), more_acts, ckpt_path=ckpt, ckpt_every=5, resume=True, **FIT_KW)
    except ValueError as e:
        assert "total_steps" in str(e)
    else:
        raise AssertionError("resume with mismatched total_steps should raise ValueError")


def test_corrupt_checkpoint_falls_back_to_fresh_start(tmp_path):
    ckpt = tmp_path / "glp_ckpt.pt"
    ckpt.write_bytes(b"not a real torch checkpoint")  # truncated/garbage, as a kill mid-save leaves

    # Must not crash: warn-through and train from scratch (all steps run).
    hist = fit(_tiny_model(), _acts(), ckpt_path=ckpt, ckpt_every=5, resume=True, **FIT_KW)
    assert len(hist["loss"]) == 20  # full run, nothing resumed
    assert ckpt.exists() and torch.load(ckpt, map_location="cpu")["step"] > 0  # overwritten cleanly


def test_atomic_save_leaves_no_tmp_and_is_loadable(tmp_path):
    ckpt = tmp_path / "glp_ckpt.pt"
    fit(_tiny_model(), _acts(), ckpt_path=ckpt, ckpt_every=5, resume=False, **FIT_KW)
    # The sidecar the atomic write uses must not be left behind.
    assert not (tmp_path / "glp_ckpt.pt.tmp").exists()
    torch.load(ckpt, map_location="cpu")  # loads without error
