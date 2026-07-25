"""Flow-matching round-trip (PLAN §6.2 unit test; RECON §6).

A denoiser that returns the EXACT velocity u = eps - z0 must reconstruct z0 from
z1 = eps under the reverse sampler, to tolerance. Because z(t) = z0 + t*(eps - z0)
is linear in t, the Euler integrator is exact for a constant velocity field
regardless of step count — so this also checks the sampler's time bookkeeping.
"""

import torch

from glp_plm.diffusion import flow_matching
from glp_plm.diffusion.sampler import sample, sample_on_manifold


def test_generate_from_noise_reconstructs_z0():
    torch.manual_seed(0)
    d = 320
    n = 64
    z0 = torch.randn(n, d)
    eps = torch.randn(n, d)  # this is z1 (t=1)

    # Exact-velocity oracle denoiser: constant u = eps - z0 for all (z, t).
    u = flow_matching.velocity_target(z0, eps)

    def oracle(z, t):
        return u

    for num_steps in (1, 5, 20, 1000):
        z_rec = sample(oracle, eps, num_steps=num_steps, t_start=1.0)
        # Exact in real arithmetic; the only error is float32 round-off from summing
        # `num_steps` increments (~1e-4 at 1000 steps), so scale tolerance with steps.
        atol = 1e-6 + 2e-7 * num_steps
        assert torch.allclose(z_rec, z0, atol=atol), (
            f"reconstruction failed at num_steps={num_steps}: "
            f"max abs err {(z_rec - z0).abs().max().item():.2e} (atol={atol:.1e})"
        )


def test_interpolation_endpoints():
    z0 = torch.randn(8, 320)
    eps = torch.randn(8, 320)
    t0 = torch.zeros(8)
    t1 = torch.ones(8)
    assert torch.allclose(flow_matching.interpolate(z0, eps, t0), z0)
    assert torch.allclose(flow_matching.interpolate(z0, eps, t1), eps)


def test_sdedit_partial_start_reconstructs_for_oracle():
    """From t_start<1, noising z0 then reverse-integrating with the oracle returns z0."""
    torch.manual_seed(1)
    z0 = torch.randn(16, 320)

    # Oracle that always points along (its own injected) eps - z0 would need eps;
    # instead verify the algebra: sample_on_manifold noises with a fresh eps, and the
    # oracle must use THAT eps. We reproduce the internal noise with a fixed generator.
    gen = torch.Generator().manual_seed(123)
    t_start = 0.5
    eps = torch.randn(z0.shape, generator=gen)
    z_noisy = (1.0 - t_start) * z0 + t_start * eps
    u = eps - z0

    def oracle(z, t):
        return u

    z_rec = sample(oracle, z_noisy, num_steps=20, t_start=t_start)
    assert torch.allclose(z_rec, z0, atol=1e-5)


def test_denoiser_forward_shapes():
    """The real GLPDenoiser runs and preserves shape on both (N,d) and (B,S,d)."""
    from glp_plm.config import ModelConfig
    from glp_plm.model import GLPDenoiser

    model = GLPDenoiser(ModelConfig(activation_dim=320, n_blocks=2))
    z2 = torch.randn(10, 320)
    t = torch.rand(10)
    assert model(z2, t).shape == z2.shape

    z3 = torch.randn(3, 7, 320)
    assert model(z3, torch.rand(21)).shape == z3.shape

    mn = model.meta_neurons(z2, torch.full((10,), 0.9))
    # meta-neuron bank width = n_blocks * d_mlp = 2 * (4*320) = 2560
    assert mn.shape == (10, 2 * 4 * 320)
