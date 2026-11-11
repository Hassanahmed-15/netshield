# NetShield

Intrusion detection that works on a network it has never seen, with zero labels from that network.

Trained on four labelled NetFlow datasets and tested once on hidden CICIDS2017, the final method caught 94% of attacks with 1.7% false alarms (balanced accuracy 0.96, ROC-AUC 0.98, F1 0.86). It catches floods, port scans and FTP password guessing, and still misses quiet attacks such as web attacks, slow DoS and SSH password guessing.

## Where to start

- `reports/NetShield_Mid-Semester_Report.pdf`: mid-semester report
- `reports/NetShield_Team_Briefing.pdf`: everything the team went through, including where each dataset came from
- `REPORT.md`: full technical write-up, sections 1 to 10
- `notebooks/20_final_results.ipynb`: displays all saved results without retraining

## Code

| Script | What it does |
|---|---|
| `src/prepare_data.py`, `src/features.py` | Load UNSW-NB15 and CICIDS2017 and build shared features |
| `src/run_final.py`, `src/run_joint_full.py`, `src/extra_fixes.py`, `src/make_plots.py` | Fixed UNSW-NB15 and CICIDS2017 pipeline (report sections 1 to 7) |
| `src/nf_transfer.py` | Same-features test on NetFlow v3 (section 8) |
| `src/nf_zero_label.py` | Zero-label version 1, hidden test on BoT-IoT (section 9) |
| `src/zero_label_v2.py` | Zero-label version 2, hidden test on CICIDS2017 (section 10) |

Results tables are in `outputs/`. Notebooks 01 to 19 are the earlier exploration phases.

## Data (not in this repo)

The datasets total several gigabytes, so they are not committed.

| Data | Source | Expected location |
|---|---|---|
| UNSW-NB15 CSVs | [UNSW Canberra](https://research.unsw.edu.au/projects/unsw-nb15-dataset) | `data/raw/UNSW-NB15/` |
| CICIDS2017 MachineLearningCVE CSVs | [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html) | `data/raw/CICIDS2017/` |
| CICIDS2017 with IP addresses (`GeneratedLabelledFlows.zip`) | Official release above, or the public copy [bencorn/CICIDS2017](https://huggingface.co/datasets/bencorn/CICIDS2017) | `../cic2017_ip/` |
| NF-UNSW-NB15-v3, NF-ToN-IoT-v3, NF-CSE-CIC-IDS2018-v3, NF-BoT-IoT-v3 | [UQ NIDS datasets](https://staff.itee.uq.edu.au/marius/NIDS_datasets/), or the public Parquet copy [keys-i/netFlow](https://huggingface.co/datasets/keys-i/netFlow) | `../nf_data/` |

## Setup

```
python -m venv ../venv && ../venv/bin/pip install -r requirements.txt
../venv/bin/python -I src/zero_label_v2.py build
../venv/bin/python -I src/zero_label_v2.py dev
```

The `hidden` step of each zero-label script has already run once and refuses to run again, so the hidden-test scores stay honest. See section 10 of `REPORT.md`.
