"""Guards for the three steerable properties (glp_plm/eval/steering_properties.py).

The decomposition tests are the load-bearing ones: `analyze_steering_confound.py` claims
"this is where the drift came from", and that claim is only true if the components sum
back to the change they are decomposing.
"""
import numpy as np
import pytest

from glp_plm.eval.steering_properties import (PROPERTY_KEYS, charge_score, get_property,
                                              gravy_score, instability_score, row_delta,
                                              run_property, score_all)

UBIQUITIN = ("MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG")
LYSOZYME = ("KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLC"
            "NIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL")


def test_score_all_covers_every_property():
    s = score_all(UBIQUITIN)
    assert set(PROPERTY_KEYS) <= set(s)
    for k in PROPERTY_KEYS:
        assert s[k] == pytest.approx(get_property(k).score(UBIQUITIN))


def test_non_canonical_residues_are_refused():
    # X/B/Z/U/O have no entry in the Kyte-Doolittle or pK tables; a silent 0.0 here would
    # corrupt a contrast set rather than fail loudly.
    assert score_all("ACDXEFG") == {}


def test_charge_is_per_residue_not_net():
    """The whole reason charge is defined this way: a net-charge target would rank
    proteins substantially by length, and the diff-of-means vector would encode length."""
    doubled = LYSOZYME * 2
    assert charge_score(doubled) == pytest.approx(charge_score(LYSOZYME), abs=2e-3)
    net = score_all(LYSOZYME)["net_charge"]
    assert net == pytest.approx(charge_score(LYSOZYME) * len(LYSOZYME))
    assert abs(net) > 1.0                      # and it is genuinely a different scale


@pytest.mark.parametrize("key", ["gravy", "charge"])
def test_compositional_decomposition_is_exact(key):
    """GRAVY and charge are sum_a frequency(a) * weight(a), so the per-residue
    contributions must reconstruct the change (charge up to its two terminal terms)."""
    prop = get_property(key)
    assert prop.compositional
    total = sum(v for _, v in prop.decompose([UBIQUITIN], [LYSOZYME]))
    actual = prop.score(LYSOZYME) - prop.score(UBIQUITIN)
    assert total == pytest.approx(actual, abs=0.01)


def test_instability_decomposition_sums_over_dipeptides():
    """Instability is NOT compositional — its components are dipeptides — but the full
    (untruncated) decomposition must still reconstruct the change."""
    prop = get_property("instability")
    assert not prop.compositional
    dec = prop.decompose([UBIQUITIN], [LYSOZYME])
    assert len(dec) > 20                       # dipeptides, not 20 residues
    total = sum(v for _, v in dec)
    actual = instability_score(LYSOZYME) - instability_score(UBIQUITIN)
    assert total == pytest.approx(actual, abs=1e-6)
    assert dec == sorted(dec, key=lambda t: -abs(t[1]))


def test_instability_sees_order_but_gravy_does_not():
    """The point of including instability: a shuffle leaves composition — hence GRAVY and
    charge — untouched, but moves the dipeptide sum."""
    rng = np.random.default_rng(0)
    shuffled = "".join(rng.permutation(list(LYSOZYME)))
    assert gravy_score(shuffled) == pytest.approx(gravy_score(LYSOZYME))
    assert charge_score(shuffled) == pytest.approx(charge_score(LYSOZYME), abs=2e-3)
    assert instability_score(shuffled) != pytest.approx(instability_score(LYSOZYME), abs=1e-3)


def test_run_property_falls_back_to_gravy_for_legacy_rows():
    """Runs written before --property existed have no `property` column and were all
    GRAVY; their `delta_gravy` was the steered effect by construction."""
    legacy = [{"delta_gravy": "0.5"}]
    assert run_property(legacy).key == "gravy"
    assert row_delta(legacy[0]) == pytest.approx(0.5)
    # the generic column wins when both are present (a charge run also writes delta_gravy,
    # where it is an OFF-target measurement)
    both = {"delta_prop": "0.02", "delta_gravy": "0.5"}
    assert row_delta(both) == pytest.approx(0.02)
    assert run_property([{"property": "charge", **both}]).key == "charge"


def test_run_property_reads_the_csv_name_when_the_column_is_blank():
    assert run_property([{"property": ""}], "runs/x/expB_diffmean_instability.csv").key \
        == "instability"
