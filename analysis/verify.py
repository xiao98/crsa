"""Independent re-checks of the MDDial findings."""
import pathlib, glob
import numpy as np, pandas as pd
from scipy.special import softmax
from crsa.src.llm_models import LLMCRSA
from crsa.src.datasets.mddial import MDDialDataset
from crsa.src.utils import Predictions
np.set_printoptions(precision=3, suppress=True, linewidth=200)

# (1) authors' shipped results: corrected Table 1 --------------------------------
df = pd.read_pickle("outputs/run_mddial/meta-llama/Llama-3.2-1B-Instruct/all_results.pkl")
df["p_true"] = [r.speaker_dist[r.true_utt_idx] for r in df.itertuples()]
df["p_true_norm"] = [r.speaker_dist[r.true_utt_idx] / np.sum(r.speaker_dist) for r in df.itertuples()]
df["p_unif"] = [1 / len(r.speaker_dist) for r in df.itertuples()]
ppl = lambda col, d: d.groupby("sample_id")[col].apply(lambda x: np.exp(-np.log(x).mean())).mean()
print("(1) Table 1 from the authors' own all_results.pkl  [released | speaker dist renormalised over utterances]")
for m in ["crsa", "rsa", "literal"]:
    d = df[df.model_name == m]
    print(f"    {m:8s} {ppl('p_true', d):8.3f} | {ppl('p_true_norm', d):8.3f}    sum of speaker dist: patient {d[d.speaker=='patient'].speaker_dist.map(np.sum).mean():.3f}, doctor {d[d.speaker=='doctor'].speaker_dist.map(np.sum).mean():.3f}")
print(f"    uniform  {ppl('p_unif', df[df.model_name=='crsa']):8.3f}   <- a speaker that ignores everything")
d = df[df.model_name == "literal"]
for sp in ["patient", "doctor"]:
    x = d[d.speaker == sp]
    print(f"    literal renormalised, {sp}: per-turn PPL {np.exp(-np.log(x.p_true_norm).mean()):.3f} vs uniform {np.exp(-np.log(x.p_unif).mean()):.0f}; accuracy {np.mean([np.argmax(r.speaker_dist)==r.true_utt_idx for r in x.itertuples()]):.3f}")

# (2) dialog counts --------------------------------------------------------------
files = glob.glob("outputs/run_mddial/meta-llama/Llama-3.2-1B-Instruct/predictions/*.npy")
empty = sum(len(np.load(f, allow_pickle=True).item()["turns"]) == 0 for f in files)
print(f"(2) prediction files {len(files)}, with zero modelled turns {empty}, dialogs in table {df.sample_id.nunique()}")

# (3) independent implementation of Eqs. (11)-(13) for one patient turn -----------
np.random.seed(0)
ds = MDDialDataset(prompt_style=None, split="train"); w = ds.world
preds = Predictions(pathlib.Path("outputs/run_mddial/meta-llama/Llama-3.2-1B-Instruct/predictions"))
P = w["prior"][:, 0, :]                                  # P(a, y), single doctor meaning
def yrsa(lex, alpha, iters=200):
    L = lex @ P; L /= L.sum(1, keepdims=True)           # Eq. 13: L0(y|u) ∝ sum_a P(a,y) lex(u,a)
    Pya = P / P.sum(1, keepdims=True)                    # P(y|a)
    for _ in range(iters):
        S = softmax(alpha * Pya @ np.log(L).T, axis=1)   # Eq. 11
        L = S.T @ P; L /= L.sum(1, keepdims=True)        # Eq. 12
    return S
rng = np.random.default_rng(0)
ids = rng.choice([i for i in ds.indices if preds[i]["turns"]], 200, replace=False)
print("(3) own implementation of Eqs. 11-13, patient turns of 200 random dialogs: mean max|S-0.5| over all 12 diseases")
turns = [(t, d) for i in ids for d, t in [(None, t) for t in preds[i]["turns"]] if t["speaker"] == "patient"]
for alpha in [2.5, 5, 10, 20, 40]:
    dev = [np.abs(yrsa(softmax(t["speaker_logprob"].ravel()).reshape(2, 12), alpha) - .5).max() for t, _ in turns[:400]]
    print(f"    alpha={alpha:5}: {np.mean(dev):.4f}")

# (4) released CRSA code, alpha sweep, LLM vs oracle lexicon -----------------------
print("(4) released CRSA code on the same 200 dialogs: patient PPL / mean max|S-0.5| / diagnosis accuracy at last turn")
for lex in ["llm", "oracle"]:
    for alpha in [2.5, 10, 40]:
        m = LLMCRSA(meanings_A=w["diseases"], meanings_B=["You are a doctor"], categories=w["diseases"], prior=w["prior"], alpha=alpha, max_depth=float("inf"), tolerance=1e-3)
        nll, dev, acc = [], [], []
        for i in ids:
            s, p = ds[i], preds[i]; m.reset(w["diseases"][s["disease"]], "You are a doctor"); asked = None
            for t in p["turns"]:
                sp = t["speaker"]; utts = w[f"{sp}_utterances"]; ll = t["speaker_logprob"]
                if sp == "doctor": asked = t["true_utt"]
                elif lex == "oracle":
                    py = np.where(w["disease2symptoms"][:, asked], .95, .05); ll = np.log(np.vstack([py, 1 - py]))
                m.run_turn(utts[t["true_utt"]], log_lexicon=ll, utterances=utts, speaker="A" if sp == "patient" else "B")
                if sp == "patient":
                    sd = m.past_speaker_dist[-1]; nll.append(-np.log(sd[t["true_utt"]])); dev.append(np.abs(sd - .5).max())
            acc.append(np.argmax(m.get_category_distribution()) == s["disease"])
        print(f"    {lex:6s} alpha={alpha:5}: {np.exp(np.mean(nll)):.3f} / {np.mean(dev):.4f} / {np.mean(acc):.3f}", flush=True)
