# Article figures

The six SVGs are light and dark versions of three figures in the article. `generate.py` creates them with Matplotlib 3.11.2 and NumPy 2.5.4. Run `python writing/figures/generate.py` from the repository root after installing those packages.

The BANKING77 curves use the 77 aggregate F1 scores in [`results/BANKING77_PER_CLASS.json`](../../results/BANKING77_PER_CLASS.json). That file records the SHA-256 of the archived official evaluation report; it contains no query text, row predictions, or test references. The other figure values are published in [`results/BANKING77.md`](../../results/BANKING77.md) and [`results/SYNTHETIC_FINANCE.md`](../../results/SYNTHETIC_FINANCE.md). The threshold curve describes the fixed official test outcome, not training progress.
