# Repository for implementing the CRSA model

## Configuration

Create a new enviroment and install the project's requirements:

```bash
conda create --name crsa python=3.12.7
conda activate crsa
pip install -r requirements.txt
pip install -e .
```

## Scripts:

To run any of the following scripts run the following command: `python -m crsa.scripts.<script_name> --arg1 arg1 --arg2 arg2...`

- `naive_reference_game`: Produce the output for the "find A1" reference game along with the plots of the paper.
- `run_mddial`: Run Llama3.2-1B-Instruct on the MDDial, produce the literal speaker values for this dataset and outputs the table of the paper. Be sure of having the correct hardware on your machine to run the model. Hardware used in our case was NVIDIA Tesla V100 with 32 GiB of RAM.

## Citation

```
@inproceedings{estienne-etal-2025-collaborative,
    title = "Collaborative Rational Speech Act: Pragmatic Reasoning for Multi-Turn Dialog",
    author = "Estienne, Lautaro  and
      Zenou, Gabriel Ben  and
      Naderi, Nona  and
      Cheung, Jackie CK  and
      Piantanida, Pablo",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.1145/",
    doi = "10.18653/v1/2025.emnlp-main.1145",
    pages = "22520--22534",
    ISBN = "979-8-89176-332-6",
    abstract = "As AI systems take on collaborative roles, they must reason about shared goals and beliefs{---}not just generate fluent language. The Rational Speech Act (RSA) framework offers a principled approach to pragmatic reasoning, but existing extensions face challenges in scaling to multi-turn, collaborative scenarios. In this paper, we introduce Collaborative Rational Speech Act (CRSA), an information-theoretic (IT) extension of RSA that models multi-turn dialog by optimizing a gain function adapted from rate-distortion theory. This gain is an extension of the gain model that is maximized in the original RSA model but takes into account the scenario in which both agents in a conversation have private information and produce utterances conditioned on the dialog. We demonstrate the effectiveness of CRSA on referential games and template-based doctor{--}patient dialogs in the medical domain. Empirical results show that CRSA yields more consistent, interpretable, and collaborative behavior than existing baselines{---}paving the way for more pragmatic and socially aware language agents."
}
```

