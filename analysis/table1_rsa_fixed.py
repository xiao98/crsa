"""Table 1 on all dialogs with the RSA speaker's P(b,y|a) normalised along the right axis,
plus a check of whether the same line matters in the reference game."""
import pathlib
import numpy as np
from scipy.special import softmax
import crsa.src.llm_models.rsa as rsa_mod
from crsa.src.llm_models import LLMRSA
from crsa.src.naive_models.utils import ZERO
from crsa.src.datasets.mddial import MDDialDataset
from crsa.src.datasets import FindA1Dataset
from crsa.src.utils import Predictions

def fixed_update(self, listener):
    prior = np.clip(self.prior, ZERO, None)
    prior_by_given_a = prior / prior.sum(axis=(1, 2), keepdims=True)
    log_listener = np.log(np.clip(listener.as_array, ZERO, None))
    s = softmax(self.alpha * np.einsum("aby,uby->au", prior_by_given_a, log_listener) - self.costs.reshape(1, -1), axis=1)
    self.history.append(np.clip(s, ZERO, None))
rsa_mod.Speaker.update = fixed_update

np.random.seed(1234)
ds = MDDialDataset(prompt_style=None, split="train"); w = ds.world
preds = Predictions(pathlib.Path("outputs/run_mddial/meta-llama/Llama-3.2-1B-Instruct/predictions"))
m = LLMRSA(meanings_A=w["diseases"], meanings_B=["You are a doctor"], categories=w["diseases"], prior=w["prior"], alpha=2.5, max_depth=float("inf"), tolerance=1e-3)
ppl, acc, dev = [], [], []
for i in ds.indices:
    s, p = ds[i], preds[i]
    if not p["turns"]: continue
    m.reset(w["diseases"][s["disease"]], "You are a doctor"); nll = []
    for t in p["turns"]:
        utts = w[f"{t['speaker']}_utterances"]
        m.run_turn(utts[t["true_utt"]], log_lexicon=t["speaker_logprob"], utterances=utts, speaker="A" if t["speaker"] == "patient" else "B")
        sd = m.past_speaker_dist[-1]; nll.append(-np.log(sd[t["true_utt"]])); dev.append(np.abs(sd - 1 / len(sd)).max())
    ppl.append(np.exp(np.mean(nll))); acc.append(np.argmax(m.get_category_distribution()) == s["disease"])
print(f"RSA fixed, {len(ppl)} dialogs: Speaker PPL {np.mean(ppl):.3f}  TSR {np.mean(acc):.3f}  max deviation from uniform {np.max(dev):.2e}   (sqrt(2*118) = {np.sqrt(236):.3f})")

g = FindA1Dataset(6, n_samples=1).world; P = np.asarray(g["prior"], float)
pa = P.sum(axis=(1, 2)); pb = P.sum(axis=(0, 2))
print("reference game prior", P.shape, " P(m_A) min/max", pa.min().round(5), pa.max().round(5), " P(m_B) min/max", pb.min().round(5), pb.max().round(5))
