# Paper data

The data cover Kuhn poker (3 players), Leduc poker, random-order Goofspiel
(4 cards), Sheriff, and Tiny Bridge.

| Directory | Runs | Parameters |
| --- | --- | --- |
| `raw/fixed/` | 30 | Six epsilon values per game: 0.1, 0.05, 0.01, 0.005, 0.001, 0 |
| `raw/adaptive/` | 5 | Initial epsilon 0.1, decay 0.5, minimum epsilon 1e-10 |

The iteration budget is 100,000 per algorithm. Files are gzip-compressed
`.ssv` records; `manifest.json` lists their games and parameters.

## Labels

Data labels and filenames are retained. Plot headings use the paper names:

| Data label | Paper name |
| --- | --- |
| A-EFR_IN | ACT-IN |
| CFR | CF |
| CFR_IN | CF-IN |
| CSPS-EFR | CSPS |
| CFPS-EFR | CFPS |

## Metrics

Fixed files contain `t = N` blocks with one tuple per algorithm.

| Plot field | Paper metric | Fixed-data conversion |
| --- | --- | --- |
| `posr` | Perturbed OSR(mu) | `max_full_phi_regrets_fus / (t + 1)` |
| `osr` | OSR(nu) | `max_full_phi_regret / (t + 1)` |
| `csr` | CSR(nu) | `max_phi_regret` |

Adaptive files are CSV. Use the `posr`, `osr`, and `csr` values in
`CHECKPOINT` rows directly, with `global_t` on the horizontal axis.

## Reproduce the figures

From the repository root:

```sh
python3 -m pip install -r requirements-plot.txt
python3 bin/plot_PBCE.py
```

The command reads the compressed files directly and creates five fixed and
five adaptive PDFs in `outputs/figures/{fixed,adaptive}/`.

Default fixed sampling keeps `t = 0, 100, ..., 99900`, selects 100 evenly
spaced points, and plots `t + 1`. Adaptive plots use all checkpoint rows.
