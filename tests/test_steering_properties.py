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


#: How close the components must sum to the change they explain. GRAVY and instability
#: are exact identities; charge carries a small residual because `charge_at_pH` adds two
#: terminal terms that composition cannot see (~0.005 charge/residue at these lengths).
DECOMPOSITION_TOL = {"gravy": 1e-9, "charge": 8e-3, "instability": 1e-9}


@pytest.mark.parametrize("key", ["gravy", "charge"])
def test_compositional_decomposition_is_exact(key):
    """GRAVY and charge are sum_a frequency(a) * weight(a), so the per-residue
    contributions must reconstruct the change (charge up to its two terminal terms)."""
    prop = get_property(key)
    assert prop.compositional
    total = sum(v for _, v in prop.decompose([UBIQUITIN], [LYSOZYME]))
    actual = prop.score(LYSOZYME) - prop.score(UBIQUITIN)
    assert total == pytest.approx(actual, abs=DECOMPOSITION_TOL[key])


@pytest.mark.parametrize("key", ["gravy", "charge", "instability"])
def test_decomposition_is_exact_across_sequences_of_different_lengths(key):
    """The real call site decomposes a *set* of sequences against a *set* of starts, and
    ours span 100-200 aa. Each sequence must be weighted equally, because what the
    components have to reconstruct is the change in the MEAN PER-SEQUENCE score. Pooling
    residues instead would weight long proteins more, and the error is invisible when every
    sequence has the same length — which is why this case is tested separately."""
    prop = get_property(key)
    before = [UBIQUITIN, LYSOZYME, UBIQUITIN[:40]]
    after = [LYSOZYME, UBIQUITIN, LYSOZYME[:150]]
    assert len({len(s) for s in before + after}) > 1        # the case that exposes it
    total = sum(v for _, v in prop.decompose(before, after))
    actual = (np.mean([prop.score(s) for s in after])
              - np.mean([prop.score(s) for s in before]))
    assert total == pytest.approx(actual, abs=DECOMPOSITION_TOL[key])


def test_instability_decomposition_sums_over_dipeptides():
    """Instability is NOT compositional — its components are dipeptides — but the full
    (untruncated) decomposition must still reconstruct the change."""
    prop = get_property("instability")
    assert not prop.compositional
    dec = prop.decompose([UBIQUITIN], [LYSOZYME])
    assert len(dec) > 20                       # dipeptides, not 20 residues
    total = sum(v for _, v in dec)
    actual = instability_score(LYSOZYME) - instability_score(UBIQUITIN)
    assert total == pytest.approx(actual, abs=DECOMPOSITION_TOL["instability"])
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


# --------------------------------------------------------------------------- #
#  Contrast construction (scripts/build_property_contrast.py)
# --------------------------------------------------------------------------- #
def _synthetic_pool(rng, n=600):
    """A pool where the property and LENGTH are deliberately confounded: high-GRAVY
    sequences are made systematically longer. That is the situation --match-length exists
    for, and it is not hypothetical — it is what a sum-valued score (net charge) does on
    its own."""
    from glp_plm.eval.steering_properties import score_all as _score_all

    pool = []
    for _ in range(n):
        hydrophobic = rng.random() < 0.5
        alphabet = "LVIAFM" if hydrophobic else "KRDENQ"
        # overlapping, not disjoint: real length distributions overlap, and a pool with no
        # shared support is refused outright by _length_matched_draw (correctly — there is
        # nothing to match on).
        length = int(rng.integers(140, 200) if hydrophobic else rng.integers(100, 165))
        seq = "".join(rng.choice(list(alphabet + "GSTP")) for _ in range(length))
        pool.append((seq, _score_all(seq)))
    return pool


def test_length_matching_equalizes_the_contrast_sets():
    import sys
    sys.path.insert(0, "scripts")
    from build_property_contrast import build_contrast

    rng = np.random.default_rng(0)
    pool = _synthetic_pool(rng)
    kw = dict(prop_key="gravy", n_per_set=30, n_heldout=10, seed=0, n_heldout_high=10)

    unmatched = build_contrast(pool, match_length=False, **kw)
    matched = build_contrast(pool, match_length=True, **kw)

    def gap(sets):
        return (np.mean([len(s) for s, _ in sets["high"]])
                - np.mean([len(s) for s, _ in sets["low_in"]]))

    assert abs(gap(unmatched)) > 20          # the confound is real in this pool
    assert abs(gap(matched)) < abs(gap(unmatched)) / 4


def test_length_matched_heldout_is_not_length_biased():
    """The matched draw is assembled bin by bin and the caller splits it by position, so
    without a shuffle the held-out starts would be the shortest proteins in the draw —
    a length confound between the starts and the vector, reintroduced by the fix for the
    length confound between the two contrast sets."""
    import sys
    sys.path.insert(0, "scripts")
    from build_property_contrast import build_contrast

    rng = np.random.default_rng(1)
    sets = build_contrast(_synthetic_pool(rng), prop_key="gravy", n_per_set=30,
                          n_heldout=10, seed=0, n_heldout_high=10, match_length=True)
    for held, kept in (("heldout", "low_in"), ("heldout_high", "high")):
        h = np.mean([len(s) for s, _ in sets[held]])
        k = np.mean([len(s) for s, _ in sets[kept]])
        assert abs(h - k) < 12, f"{held} mean length {h:.1f} vs {kept} {k:.1f}"


# --------------------------------------------------------------------------- #
#  The degeneracy gate
# --------------------------------------------------------------------------- #
def test_gate_agrees_between_the_sequence_and_row_forms():
    """Analyses read rows (which carry precomputed counts); other callers pass raw
    strings. They must not be able to disagree — the gate had seven independent copies at
    one point and one had drifted, so a script silently analysed a different row set from
    the summary.json numbers it printed beside."""
    from glp_plm.eval.steering_properties import (is_degenerate_row, is_degenerate_sequence,
                                                  max_homopolymer_run)
    cases = [LYSOZYME, UBIQUITIN, "L" * 130, "LV" * 65, UBIQUITIN[:30] + "A" * 20,
             "".join("ACDEFGHIKLMNPQRSTVWY"[i % 20] for i in range(140))]
    for seq in cases:
        row = {"sequence": seq, "unique_aa_count": len(set(seq)),
               "max_homopolymer_run": max_homopolymer_run(seq)}
        assert is_degenerate_row(row) == is_degenerate_sequence(seq), seq[:40]


def test_gate_rejects_junk_and_keeps_real_proteins():
    from glp_plm.eval.steering_properties import is_degenerate_sequence
    assert is_degenerate_sequence("L" * 130)            # homopolymer
    assert is_degenerate_sequence("LV" * 65)            # two residues only
    assert not is_degenerate_sequence(LYSOZYME)
    assert not is_degenerate_sequence(UBIQUITIN)


def test_gate_row_form_falls_back_when_the_counts_are_missing():
    """Natural-protein CSVs (heldout_starts.csv) have no count columns."""
    from glp_plm.eval.steering_properties import is_degenerate_row
    assert is_degenerate_row({"sequence": "L" * 130})
    assert not is_degenerate_row({"sequence": LYSOZYME})
    assert not is_degenerate_row({"sequence": LYSOZYME, "unique_aa_count": "",
                                  "max_homopolymer_run": ""})
