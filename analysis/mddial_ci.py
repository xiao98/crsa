"""Paired bootstrap CIs for Table 1 (MDDial) + belief entropy diagnostic.
Usage: python analysis/mddial_ci.py <path/to/all_results.pkl>"""
import sys
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
df = pd.read_pickle(sys.argv[1])
df["nll"] = [-np.log(r.speaker_dist[r.true_utt_idx]) for r in df.itertuples()]

# per-dialog metrics, same definitions as compute_results in run_mddial.py
last = df.sort_values("turn").groupby(["model_name", "sample_id"]).tail(1).set_index(["model_name", "sample_id"])
ppl = df.groupby(["model_name", "sample_id"]).nll.mean().apply(np.exp).unstack(0)
acc = pd.Series([int(np.argmax(r.listener_dist) == r.category_idx) for r in last.itertuples()], index=last.index).unstack(0)
n_cat = len(last.listener_dist.iloc[0])
n = len(ppl)

def ci(x):
    b = x[rng.integers(0, n, (10000, n))].mean(1)
    return f"{x.mean():.4f} [{np.quantile(b, .025):.4f}, {np.quantile(b, .975):.4f}]"

print(f"dialogs={n}  categories={n_cat}  chance accuracy={1/n_cat:.4f}")
for m in ["crsa", "rsa", "literal"]:
    print(f"{m:8s} PPL {ci(ppl[m].values)}   TSR {ci(acc[m].values.astype(float))}")
print("paired diff rsa-crsa   PPL", ci((ppl.rsa - ppl.crsa).values), "  TSR", ci((acc.crsa - acc.rsa).values.astype(float)))
print("dialogs where CRSA and RSA argmax diagnosis differ:",
      int(sum(np.argmax(a) != np.argmax(b) for a, b in zip(last.loc["crsa"].listener_dist, last.loc["rsa"].listener_dist))))

# raw LLM diagnosis (no pragmatics)
llm_acc = np.array([int(np.argmax(r.category_distribution) == r.category_idx) for r in last.loc["crsa"].itertuples()], float)
print("raw LLM diagnosis TSR", ci(llm_acc))

# PPL by speaker role
by = df.groupby(["model_name", "speaker"]).nll.mean().apply(np.exp).unstack(0)
print("\nper-turn PPL by speaker role:\n", by.round(3))

# belief entropy over turns (CRSA only), normalised by log|M|
def H(p):
    p = np.asarray(p, float).ravel(); p = p / p.sum(); p = p[p > 0]
    return float(-(p * np.log(p)).sum())
c = df[df.model_name == "crsa"].copy()
for k in ["belief_A", "belief_B"]:
    size = np.asarray(c[k].iloc[0]).size
    c[k + "_H"] = [H(b) / np.log(size) if size > 1 else np.nan for b in c[k]]
    print(f"\n{k}: size={size}")
print(c.groupby("turn")[["belief_A_H", "belief_B_H"]].agg(["mean", "count"]).round(3).head(14))
print("\nlistener posterior normalised entropy at last turn:",
      {m: round(np.mean([H(d) / np.log(n_cat) for d in last.loc[m].listener_dist]), 3) for m in ["crsa", "rsa", "literal"]})
