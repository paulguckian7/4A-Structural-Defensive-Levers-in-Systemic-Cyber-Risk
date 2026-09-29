# Paper 4A: Structural Defensive Levers in Systemic Cyber Risk

Analysis program, specification and run outputs for:

> P. Guckian, "Structural Defensive Levers in Systemic Cyber Risk: Evidence from the Extended Cyber Triangle on a Frozen Substrate", manuscript, 2026.

## Contents

| Path | What it is |
|---|---|
| `PhDPaper4A.py` | The analysis program: one self-contained file. The frozen `cemt_core` v0.8 (tag `docker-eval-v0.8`, digest `97dbae47`) is embedded byte for byte and hash-checked at load. |
| `spec/` | Paper 4A Specification v12 (20 September 2026), fixed before the confirmatory run. |
| `run/20260920_215757_p4a_all_v12/` | Confirmatory run summary outputs: `run_record.json`, `verdicts.json`, `verdicts.txt`, `P4A_summary.md`. |
| `run/.../exploratory_analysis_20260927_122758/` | Labelled post hoc re-analysis with the corrected H3 zero rule, and the publication figures. Not confirmatory. |

The full per-architecture data (`architectures.csv`, `outcomes.csv`, `nodes.csv.gz`, `runs.csv.gz`, `specs/`) are deposited with the Zenodo record for this repository because of their size.

## Versions

| Tag | Code | Use |
|---|---|---|
| `v12` | `PhDPaper4A v12 (2026-09-20)` | Frozen confirmatory code. Produced run `20260920_215757_p4a_all_v12`; its SHA-256 (run-configuration block excluded) is recorded in that run's `run_record.json`. |
| `v13` | `PhDPaper4A v13 (2026-09-27)` | v12 plus two changes that do not touch generation: the corrected H3 zero rule (22 September 2026, post hoc) and the FIGURES mode. Produced the exploratory re-analysis. |

## Running

Python 3.11 with numpy, pandas and scipy (matplotlib for FIGURES mode only). Run `python PhDPaper4A.py` and choose a mode:
PILOT, TIME, ALL (full confirmatory run, about 22 hours on 14 workers), GENERATE, ANALYSE or FIGURES.
Set the output folder in `RUN_CONFIG` at the top of the file.

To check that a copy of the v12 file is the one that produced the run, compare its hash with `code_sha256` in `run_record.json`:

```
python -c "import re,hashlib;s=open('PhDPaper4A.py','rb').read().replace(b'\r\n',b'\n').decode();s=re.sub(r'# >>> RUN_CONFIG.*?# <<< RUN_CONFIG','',s,flags=re.S);print(hashlib.sha256(s.encode()).hexdigest())"
```

## Licence

Code: MIT (see `LICENSE`). Data and run outputs: CC BY 4.0.
