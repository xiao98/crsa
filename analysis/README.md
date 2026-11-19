# Notes on the MDDial experiment (Table 1)

Scripts written while reproducing the paper. They only read the outputs shipped in this
repository (no GPU, no LLM call) and do not modify anything under `src/`.

```bash
# from the repository root, after `pip install -e .`
python analysis/mddial_ci.py outputs/run_mddial/meta-llama/Llama-3.2-1B-Instruct/all_results.pkl
python analysis/table1_rsa_fixed.py
python analysis/verify.py
```

| script | what it does |
|---|---|
| `mddial_ci.py` | Table 1 with paired bootstrap CIs, perplexity split by speaker role, belief entropy per turn |
| `table1_rsa_fixed.py` | RSA row of Table 1 with `P(b,y\|a)` normalised along the meaning axis |
| `verify.py` | Table 1 with speaker distributions renormalised over utterances; a standalone implementation of Eqs. 11-13; alpha sweep |

## Observations

Table 1, alpha = 2.5, 1703 dialogs with at least one intermediate turn:

| model | released | after the two changes below |
|---|---|---|
| CRSA | 15.362 | 15.362 |
| RSA | 15.395 | 15.362 |
| Literal | 60.835 | 16.232 |
| uniform speaker | | 15.362 = sqrt(2 * 118) |

1. `llm_models/rsa.py`, `Speaker.update`: `prior / prior_a` divides an (A, B, Y) array by an (A, 1)
   array, which broadcasts along B. On MDDial the result has shape (12, 12, 12) and sums to 6-22
   per meaning. With `prior / prior_a[:, :, np.newaxis]` RSA gives 15.362.
2. `llm_models/literal.py`, `LLMLiteral.run_turn`: the reported speaker distribution is a column of
   the jointly normalised lexicon; on patient turns it sums to about 0.08.
3. The CRSA speaker is uniform (exactly on doctor turns, within 3e-3 on patient turns) and the
   listener predicts the same class for every dialog. A standalone implementation of Eqs. 11-13
   gives a uniform patient speaker for alpha <= 10.

The line in (1) is also present in `naive_models`. Rerunning the reference game with the same
change leaves the conclusions of Figure 2 unchanged (CRSA reaches 1.0 at turn 6, YRSA stays near 0.59).

A write-up with the derivations is in `analysis/note/mddial_note.pdf`.
