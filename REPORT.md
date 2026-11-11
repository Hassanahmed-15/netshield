# NetShield: intrusion detection across UNSW-NB15 and CICIDS2017

## Summary

The goal was one model that performs well on two different network datasets.

- **With no labels from the second dataset, the problem is not solved.** Training on UNSW and testing on CICIDS stayed below chance (ROC-AUC 0.41). Training on CICIDS and testing on UNSW improved to ROC-AUC 0.78, but with 42% false alarms.
- **Best zero-label result: CICIDS2017, kept hidden for version 2.** With no CICIDS2017 labels, version 2 reached 96% balanced accuracy, ROC-AUC 0.98 and F1 0.86, with 94% of attacks caught and 1.7% false alarms. It caught PortScan at 99.6%, where every earlier method failed. It missed slow DoS, SSH password guessing, web attacks, Bot and infiltration (section 10).
- **A zero-label method did work on a dataset kept back until the end, for flood attacks.** It describes each computer's behaviour per minute and learns from several other datasets. On BoT-IoT it scored ROC-AUC 0.97, with 93% of attacks caught and 4.7% false alarms, using no BoT-IoT labels. It missed most scanning, and it is unreliable on some other datasets (section 9).
- **One model trained with labels from both datasets works on both**: F1 0.99 on CICIDS and 0.88 on the full UNSW test file. It matches a separate model built for each dataset, but it false-alarms on 31% of UNSW's normal traffic.
- **A small labelled sample helps only if it contains the attacks being tested.** With 500 labelled flows from the same period, F1 reaches 0.89 (UNSW to CICIDS) and 0.88 (CICIDS to UNSW). With 500 labels from earlier days only, UNSW to CICIDS falls to F1 0.30 (section 6).
- **Why earlier work stalled:** several pipeline faults (section 2) and, more fundamentally, the two datasets label different attacks. Fixing the faults does not remove that difference.

## 1. How the evaluation was protected

Each dataset was split into a development part and a final part. All method choices used the development part. The final part was run once, after the method was locked.

| | Development | Final |
|---|---|---|
| CICIDS as test data | Monday to Wednesday | Thursday and Friday (different attack types) |
| UNSW as test data | Official training file | Official test file |

Exact duplicate rows were removed first: 337,154 from CICIDS (on the columns used) and 480,633 from the full UNSW capture. In the no-label results, no labels from the test dataset were used for fitting, alignment or thresholds. The method was chosen using labels from the development part of each dataset, never the final rows that were scored. Section 8 removes that dependence as well.

## 2. Faults found in the earlier notebooks

| Fault | Where | Effect |
|---|---|---|
| Byte counts mean different things. CICIDS "Total Length" is payload only; UNSW `sbytes` includes headers. | Phases 7 to 19 | The most common CICIDS payload per packet is 0; the UNSW minimum is about 40. Adding headers back raised development ROC-AUC by 0.02 to 0.17 in all ten comparisons. |
| "Expanded" features are maths on the same 11 columns. | Phases 14, 16 | They add almost no new information, so no real gain was possible. |
| CORAL applied to unscaled features. | Phases 18, 19 (BBSE) | The model flagged 100% of traffic as attack. |
| DANN trained without a log transform. | Phase 19 (DANN) | Training loss started at 45 and the model did worse than chance. |
| Random splits with duplicates left in. | Phases 5, 16, 17 | Within-dataset scores are likely inflated. |
| Official UNSW files are 64% attack, while the full capture is 12.6%. | All phases | Source thresholds do not carry over, and F1 misleads: flagging everything scores 0.81 on the training file. |

## 3. Results on the final data

ROC-AUC of 0.5 is a coin flip. "Flag-all F1" is the score from marking every flow as an attack, so an F1 below it adds nothing.

### No labels from the test dataset

**Train on UNSW, test on CICIDS Thursday and Friday (22% attack, flag-all F1 0.36)**

| Method | ROC-AUC | PR-AUC | F1 | Recall | False alarms |
|---|---|---|---|---|---|
| Earlier Random Forest | 0.41 | 0.19 | 0.22 | 37% | 57% |
| Earlier CORAL + balanced | 0.34 | 0.16 | 0.30 | 58% | 65% |
| **New, selected method** | **0.41** | **0.20** | **0.28** | **50%** | **57%** |
| Normalized ensemble | 0.68 | 0.35 | 0.40 | 54% | 33% |
| Same as selected, full UNSW capture as source (not selected) | 0.74 | 0.60 | 0.47 | 59% | 26% |

**Train on CICIDS, test on UNSW official test file (55% attack, flag-all F1 0.71)**

| Method | ROC-AUC | PR-AUC | F1 | Recall | False alarms |
|---|---|---|---|---|---|
| Earlier Random Forest | 0.55 | 0.56 | 0.00 | 0% | 0% |
| Earlier CORAL + balanced | 0.74 | 0.76 | 0.00 | 0% | 0% |
| **New, selected method** | **0.78** | **0.79** | **0.75** | **80%** | **42%** |
| Normalized ensemble | 0.48 | 0.52 | 0.35 | 30% | 46% |

On the full UNSW capture (8.3% attack, flag-all F1 0.15), the selected method reaches ROC-AUC 0.83, F1 0.41, recall 69% and 15% false alarms.

The selected method was fixed by a rule set in advance. The full-capture variant beat it in the first table, but choosing it after seeing final scores would be selection on the test data, so it is reported only as an observation.

### One model trained on both datasets

| Test data | ROC-AUC | F1 | Recall | False alarms |
|---|---|---|---|---|
| CICIDS, random 30%, no duplicates | 1.00 | 0.99 | 99% | 0.1% |
| UNSW official test file, all rows | 0.97 | 0.88 | 98% | 31% |
| UNSW official test, rows also in training removed | 0.96 | 0.76 | 95% | 28% |
| CICIDS Thursday and Friday, those days left out of training | 0.76 | 0.53 | 37% | 0.3% |

Separate per-dataset models score almost the same: UNSW F1 0.88 (all rows) and 0.76 (overlap removed), CICIDS F1 0.996 against 0.994 for the joint model. The last row is the harder test, because those attack types were never seen in training.

### Adding a small labelled sample (mean of 3 repeats)

| Labelled flows | UNSW to CICIDS: F1 | ROC-AUC | False alarms | CICIDS to UNSW: F1 | ROC-AUC | False alarms |
|---|---|---|---|---|---|---|
| 100 | 0.71 | 0.98 | 9% | 0.82 | 0.91 | 21% |
| 500 | 0.89 | 0.99 | 1% | 0.88 | 0.96 | 11% |
| 2000 | 0.92 | 0.995 | 3% | 0.91 | 0.97 | 11% |

With only 100 labels, results vary a lot between repeats.

## 4. Why no-label transfer fails

The two datasets disagree about what an attack looks like.

- PortScan is 41% of the CICIDS final attacks. The UNSW-trained model catches 0.4% of it and scores it as more normal than normal traffic (ROC-AUC 0.23 against benign). Without PortScan, ROC-AUC would be about 0.53, which is still close to chance.
- UNSW attacks are mostly short flows with no reply and near-zero duration. CICIDS attacks are longer floods and scans.
- Results depend heavily on which unlabeled traffic is used for alignment. The same method moved from ROC-AUC 0.79 to 0.68 on development data when only that pool changed.

See `outputs/final_model/plots/03_recall_by_attack_family.png`. Its high recall bars come with the false-alarm rates above.

## 5. Limits to keep in mind

- Development scores did not hold for the selected method. UNSW to CICIDS fell from ROC-AUC 0.88 to 0.41, and CICIDS to UNSW from 0.82 to 0.78.
- The header correction adds 20 bytes of IP header per packet. This value is inferred from the minimum packet sizes in the data, not taken from tool documentation.
- The alignment step uses all unlabeled traffic from the test dataset, including the rows later scored. It uses no labels from them, but it requires that traffic to be available in advance.
- Labelled samples in section 3's few-label test come from the same period as the rows they are tested on. Section 6 shows the score drops sharply when labels come only from earlier days.
- `outputs/final_model/tables/results.csv` holds joint and within-dataset rows scored on half of each test set. The official UNSW test file is sorted in blocks, so those halves are not comparable. `joint_full_test.csv` replaces them and uses the same models and thresholds. It covers rows that the development run also scored, but no decision used those scores.
- CICIDS's own labels were taken as given.

## 6. Follow-up tests

These were added after the final data had been scored once, so the final scores below are a second look and are weaker evidence than section 3. Choices were still made on development data only, by highest mean ROC-AUC across both directions. Only the locked choices were scored on final data. Ports, TTL and TCP flags were already excluded from the transfer features.

**What the development data showed**

- Adversarial validation: a classifier separates the two datasets almost perfectly even after per-dataset scaling (AUC 1.00). Removing the most separating features down to 6 only reached 0.997 to 0.998.
- Dropping absolute counts did not help overall. In the all-attacks test, mean ROC-AUC fell from 0.81 to 0.76 with CORAL.
- Leave-one-feature-out removed nothing in the all-attacks test, and 11 of 18 features in the shared-attacks test. That reduced set was chosen using development labels from the target, so it is not a zero-label result.

**No labels, final data**

"Shared attacks" means DoS and scanning. UNSW has no brute-force category, so brute force cannot be shared.

| Test | Direction | ROC-AUC | F1 (unlabeled threshold) | Flag-all F1 | False alarms |
|---|---|---|---|---|---|
| All attacks | UNSW to CICIDS | 0.60 | 0.39 | 0.36 | 27% |
| All attacks | CICIDS to UNSW, official test | 0.78 | 0.75 | 0.71 | 28% |
| Shared attacks | UNSW to CICIDS | 0.68 | 0.60 | 0.36 | 8% |
| Shared attacks | CICIDS to UNSW, official test | 0.69 | 0.40 | 0.29 | 55% |
| Shared attacks, features chosen with development labels | UNSW to CICIDS | 0.79 | 0.52 | 0.36 | 2% |
| Shared attacks, features chosen with development labels | CICIDS to UNSW, official test | 0.77 | 0.46 | 0.29 | 37% |

In the shared test, UNSW to CICIDS catches 94% of DDoS (ROC-AUC 0.98 against benign) and 0.3% of PortScan.

**Thresholds.** Matching the threshold to the true attack rate changed all-attacks F1 very little (0.39 to 0.41 for UNSW to CICIDS, 0.75 either way for CICIDS to UNSW). In the shared CICIDS to UNSW test it made F1 worse (0.40 to 0.20).

**Benign-only anomaly detection, final data**

| Trained on | Tested on | Method | ROC-AUC | F1 | False alarms |
|---|---|---|---|---|---|
| UNSW benign | CICIDS Thu-Fri | Isolation Forest | 0.71 | 0.28 | 19% |
| UNSW benign | CICIDS Thu-Fri | Autoencoder | 0.77 | 0.31 | 4% |
| CICIDS benign | UNSW official test | Isolation Forest | 0.42 | 0.38 | 59% |
| CICIDS benign | UNSW official test | Autoencoder | 0.56 | 0.05 | 13% |
| CICIDS Monday (target's own clean traffic) | CICIDS Thu-Fri | Isolation Forest | 0.80 | 0.55 | 16% |
| CICIDS Monday (target's own clean traffic) | CICIDS Thu-Fri | Autoencoder | 0.66 | 0.51 | 6% |

**Labelled target flows, final data (mean of 3 repeats)**

| Labels | UNSW to CICIDS: F1 | ROC-AUC | False alarms | CICIDS to UNSW: F1 | ROC-AUC | False alarms |
|---|---|---|---|---|---|---|
| 1%, same period | 0.93 | 0.995 | 4% | 0.90 | 0.96 | 11% |
| 5%, same period | 0.93 | 0.996 | 3% | 0.92 | 0.97 | 9% |
| 500, earlier period only | 0.30 | 0.78 | 0.3% | 0.85 | 0.93 | 39% |
| 2000, earlier period only | 0.55 | 0.82 | 0.6% | 0.86 | 0.95 | 38% |

Earlier period means CICIDS Monday to Wednesday, or the official UNSW training file. For UNSW to CICIDS, those labels contain no PortScan or DDoS. The 2000-label F1 varies widely between repeats (standard deviation 0.25). Flag-all F1 is 0.36 and 0.71.

**Not tested.** The corrected CICIDS2017 from Engelen et al. is public, but the download was too slow to finish in this session.

Tables are in `outputs/final_model/extra/`.

## 7. How to reproduce

```
python -m venv venv && venv/bin/pip install -r requirements.txt
venv/bin/python src/prepare_data.py ../cache
venv/bin/python src/run_final.py dev ../cache
venv/bin/python src/run_final.py final ../cache
venv/bin/python src/run_joint_full.py ../cache
venv/bin/python src/make_plots.py
venv/bin/python src/extra_fixes.py grid ../cache
venv/bin/python src/extra_fixes.py final ../cache
```

`src/features.py` holds the shared feature definitions. Results are in `outputs/final_model/`, and `notebooks/20_final_results.ipynb` displays them.

## 8. No-label test on three datasets with the same features

This tested whether a shared feature format solves the problem. It used NF-UNSW-NB15-v3, NF-ToN-IoT-v3 and NF-CSE-CIC-IDS2018-v3, from a public Parquet copy of the University of Queensland release. Row and attack counts match the official page for all three files, but checksums were not compared with the originals.

**Rules, fixed before any score was computed**

- Duplicates were removed after sampling up to 3 million flows per file. ToN and CIC2018 were sampled, UNSW was used whole.
- Each target was split by alternating clock hours. One set of hours was the unlabeled calibration pool, used for alignment and the Otsu threshold with no label filter. The other set was scored.
- Target labels were used only to score. The method for each target was chosen using only the other two datasets, by highest mean ROC-AUC between them. This is stricter than section 3, so the two sets of numbers are not directly comparable.
- Features: durations, packet sizes, rates, inter-arrival times and ratios, with or without absolute counts, and optionally TCP flags, TTL and window size. IP addresses, ports and timestamps were not used.

**Results, all attacks, no target labels**

| Train | Test | ROC-AUC | PR-AUC | F1 | Flag-all F1 | Recall | False alarms |
|---|---|---|---|---|---|---|---|
| UNSW | CIC2018 | 0.83 | 0.53 | 0.45 | 0.21 | 92% | 30% |
| ToN | UNSW | 0.49 | 0.15 | 0.16 | 0.18 | 38% | 39% |
| ToN | CIC2018 | 0.43 | 0.10 | 0.11 | 0.21 | 23% | 42% |
| CIC2018 | ToN | 0.36 | 0.72 | 0.34 | 0.87 | 23% | 35% |
| UNSW | ToN | 0.35 | 0.68 | 0.27 | 0.87 | 17% | 36% |
| CIC2018 | UNSW | 0.20 | 0.06 | 0.03 | 0.18 | 8% | 43% |

**Same model trained and tested on one dataset (separate hours)**

| Dataset | ROC-AUC | F1 | False alarms |
|---|---|---|---|
| UNSW | 1.00 | 0.998 | 0.04% |
| ToN | 0.98 | 0.96 | 8% |
| CIC2018 | 0.97 | 0.81 | 0.4% |

**Shared-attack test.** Training and scoring were limited to attack groups both datasets contain. ROC-AUC was 0.66 for UNSW to CIC2018 (DoS only), 0.56 for ToN to CIC2018, and 0.26 to 0.45 in the other four directions. Limiting to shared attacks did not help.

**What this shows**

- A shared feature format did not solve no-label transfer. Five of six directions were at or below chance, while each dataset separates its own attacks well.
- UNSW to CIC2018 worked for some attacks: 81% to 100% recall on DoS and DDoS and 100% on SSH brute force, with 30% false alarms. It caught almost no Bot traffic and 18% of infiltration.
- The same attack name does not guarantee transfer. DDoS recall was 100% from UNSW to CIC2018's HOIC attack, 3% from UNSW to ToN, and 0% from ToN to CIC2018.
- For a given direction, ROC-AUC across the 9 configurations ranged widely, for example 0.10 to 0.86 for CIC2018 to UNSW. The best configuration cannot be identified without target labels, and the choice made from other datasets did not carry over.
- After duplicates were removed, ToN is 78% attack in the scored hours, so ROC-AUC is more informative than F1 for that dataset.

Results are in `outputs/netflow_v3/`. To reproduce, run `src/nf_transfer.py prep` and then `src/nf_transfer.py run`, with the three Parquet files in `../nf_data`.

## 9. Zero-label method with per-computer behaviour

The project requires zero labels from the test dataset. Section 8 used features of single flows. This section adds what a computer does over time, because attacks such as floods and scans are defined by that behaviour.

**Method**

- Eleven features per flow, computed in one-minute windows on the whole file: how many flows, destination computers and ports a source contacted, the share of its flows that got no reply, and the matching counts for the destination and for each source and destination pair. IP addresses are used only for counting and are never model inputs.
- LightGBM trained on several labelled source datasets, each counting equally, with attack and normal traffic balanced inside each.
- Optional self-training: the model marks the target's unlabeled calibration hours, and retrains on the flows it is most confident about. These are model guesses, so no target labels are involved.
- The attack cut-off is the Otsu threshold on unlabeled calibration hours.

**Protocol, fixed before any score was computed**

- UNSW, ToN and CIC2018 had already been used in section 8, so they served as development data. Twelve configurations were tried: three feature sets, with or without quantile normalization, with or without self-training.
- The locked configuration had the highest mean ROC-AUC over the three leave-one-dataset-out tests. It was per-computer features, no normalization, with self-training.
- NF-BoT-IoT-v3 was not scored until the method was locked, and the script refuses a second run. BoT's file has 16,933,808 rows, while the official page lists 16,993,808. The attack rate agrees (99.7%), so this is likely a typo on that page.

**Held-out result: trained on UNSW, ToN and CIC2018, tested on BoT-IoT, no BoT-IoT labels used**

| ROC-AUC | PR-AUC | Attacks caught | False alarms | F1 | Flag-all F1 |
|---|---|---|---|---|---|
| 0.97 | 0.999 | 93% | 4.7% | 0.96 | 0.99 |

| Attack | Flows scored | Recall | ROC-AUC against normal |
|---|---|---|---|
| DoS | 149,131 | 95% | 0.99 |
| DDoS | 77,650 | 94% | 0.97 |
| Reconnaissance | 3,293 | 7% | 0.51 |
| Theft | 59 | 34% | 0.88 |

To measure false alarms well, every benign BoT-IoT flow was kept and only attacks were sampled, leaving 5,179 benign flows in the scored hours. BoT-IoT is almost all attack traffic, so F1 tells little here. Marking everything scores F1 0.99 but has 100% false alarms.

**Three development datasets, no target labels (train on the other two)**

| Test | Strict: method chosen only from the other two | ROC-AUC | F1 | False alarms | Locked method: ROC-AUC | F1 | False alarms | Flag-all F1 |
|---|---|---|---|---|---|---|---|---|
| UNSW | per-flow + per-computer, self-training | 0.77 | 0.21 | 25% | 0.99 | 0.53 | 0% | 0.18 |
| ToN | per-computer, quantile, self-training | 0.39 | 0.24 | 8% | 0.83 | 0.83 | 36% | 0.87 |
| CIC2018 | per-computer, no self-training | 0.78 | 0.25 | 8% | 0.80 | 0.41 | 22% | 0.21 |

The locked-method columns are optimistic for these three datasets, because each dataset's labels helped choose that method. The strict columns have no such advantage. The same models using per-flow features only scored ROC-AUC 0.14, 0.56 and 0.58 (no normalization, no self-training).

Isolation Forest fitted on the target's own unlabeled hours, with no source dataset and no labels, scored ROC-AUC 0.98 on UNSW (per-computer features, F1 0.72, 8% false alarms) and 0.80 on CIC2018. On ToN it scored 0.14, because after duplicates are removed attacks are the majority there, so normal traffic looks unusual. Results are in `dev_isolation_forest.csv`.

**Limits**

- There is one held-out dataset, and it is a lab IoT capture dominated by floods, which are the easiest attacks to see this way. It says little about subtle attacks.
- Scanning remained weak on BoT-IoT (7% recall). Results on the development datasets vary by method, and the strict ToN result is below chance.
- Windows count flows within the same minute, including later flows, so this is offline analysis.
- Lab attacks often come from a few computers, which makes per-computer patterns stronger than they may be on a real network.
- It needs IP addresses and timestamps. The CICIDS2017 file used in sections 1 to 7 has neither, so this method could not be run on the original pair. The corrected CICIDS2017 files include both, but they could not be fully downloaded here.

Results are in `outputs/netflow_v3/zero_label/`. To reproduce, place the four Parquet files in `../nf_data` and run `src/nf_zero_label.py context`, `dev` and `holdout`. The holdout step runs once.

## 10. Zero-label method, version 2, tested on hidden CICIDS2017

Version 1 was tested on BoT-IoT, so BoT-IoT could not serve as a hidden test again. Version 2 used all four NetFlow datasets for development, and CICIDS2017 with IP addresses as the new hidden test.

**Data.** `GeneratedLabelledFlows.zip`, from a public Hugging Face copy of the official CICIDS2017 release, includes source and destination IP addresses and timestamps. Its 2,830,743 rows match the file used in sections 1 to 7 exactly. The timestamps print afternoon hours on a 12-hour clock without AM or PM. Hours before 8 were moved forward 12 hours, which gives working hours 8 to 17. Duplicates were removed (2,522,592 rows kept), and hours alternate between unlabeled calibration and marked test, as in sections 8 and 9.

**What version 2 adds.** Every feature comes from five fields: source IP, destination IP, destination port, start time and whether the flow got a reply. So CICFlowMeter and NetFlow records get identical features.

- Each computer's behaviour in 1-, 10- and 60-minute windows, which can show slower attacks.
- A count of how many computers a source reached on the same port, which is how a scan looks across the network.
- How unusual a minute is for that computer, compared with its own typical minute.
- An option to keep only features that point the same way, clearly, in every training dataset. This targets the features shown in section 9 to flip direction between networks.
- An Isolation Forest fitted on the test network's own unlabeled hours, used alone or averaged with the model.
- Computer smoothing: each flow's score is averaged with the mean score of the same computer's flows in that 10-minute window.

**Protocol, fixed in the script before any score.** The development test marked each NetFlow dataset after training on the other three. There were 80 combinations: 4 feature sets, consistent features on or off, self-training on or off, and 5 ways of scoring. The locked combination had the highest mean balanced accuracy. It was all windows plus relative features, consistent features only, self-training on, and computer smoothing. Only 10 features survived the consistency rule. CICIDS2017 was then marked once, for version 2 and for version 1 as locked in section 9, and the script refuses a second run.

**Hidden result, CICIDS2017, no CICIDS2017 labels (1,000,000 marked flows, 6.2% attack)**

| Method | Balanced accuracy | Accuracy | ROC-AUC | PR-AUC | F1 | Flag-all F1 | Attacks caught | False alarms |
|---|---|---|---|---|---|---|---|---|
| Version 1 (section 9) | 0.82 | 0.85 | 0.92 | 0.72 | 0.39 | 0.12 | 78% | 14.5% |
| **Version 2** | **0.96** | **0.98** | **0.98** | **0.95** | **0.86** | 0.12 | **94%** | **1.7%** |

| Attack in the marked hours | Flows | Version 1 recall | Version 2 recall |
|---|---|---|---|
| DoS Hulk | 38,089 | 100% | 100% |
| DoS GoldenEye | 9,493 | 78% | 100% |
| PortScan | 7,844 | 0% | 99.6% |
| FTP password guessing | 3,137 | 58% | 100% |
| DoS slowloris | 2,048 | 54% | 0% |
| SSH password guessing | 793 | 7% | 0% |
| Web attacks (brute force, XSS, SQL injection) | 758 | 0% | 0% |
| Bot | 280 | 2% | 0% |
| Infiltration and Heartbleed | 27 | 0% | 0% |

Flow counts are from version 2's sample, and version 1's differ by under 2%.

**Development result, each NetFlow dataset marked after training on the other three**

| Test | Version 1 settings: balanced accuracy | Version 2 locked: balanced accuracy | Version 2 ROC-AUC | Version 2 false alarms |
|---|---|---|---|---|
| UNSW | 0.47 | 0.85 | 1.00 | 0% |
| ToN | 0.71 | 0.80 | 0.89 | 35% |
| CIC2018 | 0.94 | 0.96 | 0.98 | 0.6% |
| BoT-IoT | 0.94 | 0.98 | 0.99 | 2.8% |

These development numbers are optimistic, because version 2 was the best of 80 combinations on these same datasets. The hidden CICIDS2017 numbers have no such advantage.

**Limits**

- CICIDS2017 was hidden from version 2, but not from the project. Sections 1 to 7 used its labels for other methods, and the PortScan failure found there was one reason to add longer windows. No CICIDS2017 label or score was used to build, choose or tune version 2.
- Because hours alternate, all DDoS (128,016 flows) and DoS Slowhttptest (5,228) fell in calibration hours and were not marked. Most PortScan (82,328 of 90,819) and DoS Hulk flows also fell there.
- Recall is dominated by DoS Hulk, which is 61% of the marked attacks. Quiet attacks were all missed: web attacks, Bot, infiltration, and slow DoS (slowloris).
- Version 2 kept only port-count, port-spread, repeated-service and relative-traffic features. It is a detector for scanning, floods and repeated login attempts, not a general one.
- CICIDS2017's own labels contain known errors, which the corrected release by Engelen et al. fixes. They were taken as given.
- Windows count flows within the same window, including later flows, so this is offline analysis.

Results are in `outputs/zero_label_v2/`. To reproduce, place the NetFlow Parquet files in `../nf_data` and `GeneratedLabelledFlows.zip` in `../cic2017_ip`, then run `src/zero_label_v2.py build`, `dev` and `hidden`. The hidden step runs once.
