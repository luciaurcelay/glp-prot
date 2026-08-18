"""Check every number docs/paper_draft.tex asserts against the run metrics.

Run after regenerating any steering result, before submitting. The draft quotes rounded
values in prose and tables; this pins each of them to the metrics they came from, so a
re-run that shifts a number cannot leave a stale one in the text. Paths point at the MN5
run root and are meant to be edited when the canonical runs change.
"""
import json, sys
R = "/gpfs/projects/bsc70/bsc070332/glp-prot-runs/steering_crg/"
cp = json.load(open(R + "crossprop_rel_n40/metrics/summary.json"))
sp = json.load(open(R + "specificity_rel_n40/metrics/summary.json"))
ok = True
def chk(name, claimed, actual, tol):
    global ok
    good = abs(claimed - actual) <= tol
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} {name:<46} draft {claimed:>10} vs metrics {actual:>10.4f}")

print("efficacy @ alpha=0.256934, corrected divergence & SD units")
for k, div, sd in [("gravy", 1.581, 3.46), ("charge", 0.216, 3.61), ("instability", 25.04, 1.59)]:
    e = cp["efficacy"][k]["0.256934"]["corrected"]
    chk(f"{k} divergence", div, e["divergence"], 0.01)
    chk(f"{k} SD units", sd, e["sd_units"], 0.01)
print("naive @ same alpha")
for k, div in [("gravy", 0.077), ("charge", 0.055), ("instability", 5.30)]:
    chk(f"{k} naive", div, cp["efficacy"][k]["0.256934"]["naive"]["divergence"], 0.01)
print("naive @ top alpha (the 'not inert' claim)")
for k, div in [("gravy", 1.400), ("charge", 0.430), ("instability", 17.01)]:
    chk(f"{k} naive top", div, cp["efficacy"][k]["0.642336"]["naive"]["divergence"], 0.01)
print("decoding fixed point")
d = cp["decoding_fixed_point"]
chk("slope", 0.199, d["slope"], 0.001); chk("intercept", 0.172, d["intercept"], 0.001)
chk("fixed point", 0.215, d["fixed_point"], 0.001); chk("r", 0.80, d["r"], 0.005)
chk("r(loss)", -0.98, d["r_loss"], 0.005)
lo = min(x["natural"] for x in d["sets"]); hi = max(x["natural"] for x in d["sets"])
plo = min(x["post_loop"] for x in d["sets"]); phi = max(x["post_loop"] for x in d["sets"])
chk("natural range width", 0.281, hi - lo, 0.002)
chk("post range width", 0.088, phi - plo, 0.002)
print("alpha=0 fixed bias")
for k, u, dn in [("gravy", 0.798, 0.818), ("charge", 0.008, -0.032), ("instability", -10.33, -11.65)]:
    chk(f"{k} bias up", u, cp["alpha0_drift"][k]["fixed_bias_up"], 0.01)
    chk(f"{k} bias down", dn, cp["alpha0_drift"][k]["fixed_bias_down"], 0.01)
print("alpha=0 composition (pp), gravy run")
for a, v in [("L", 5.9), ("V", 5.4)]:
    chk(f"gravy {a}", v, cp["alpha0_composition_pp"]["gravy"][a], 0.05)
print("specificity on/off-target (SD units)")
cells = sp["specificity"]["corrected"]
for (sk, mk), v in {("gravy","gravy"):1.21, ("gravy","charge"):-0.03, ("gravy","instability"):0.07,
                    ("charge","charge"):0.67, ("charge","instability"):0.60,
                    ("instability","instability"):0.86, ("instability","gravy"):-0.54}.items():
    chk(f"{sk}->{mk}", v, cells[sk][mk]["standardized"], 0.005)
print("natural correlations")
c = sp["natural_correlations"]
for (a,b), v in {("gravy","instability"):-0.35, ("gravy","charge"):-0.03, ("charge","instability"):0.16}.items():
    chk(f"r({a},{b})", v, c[a][b], 0.005)
print("\nALL CONSISTENT" if ok else "\nMISMATCHES FOUND")
sys.exit(0 if ok else 1)
