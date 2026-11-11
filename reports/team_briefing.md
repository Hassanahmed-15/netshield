---
title: "NetShield: Team Briefing"
date: "November 11, 2026"
---

## Where we are

Our detector now works on a network it has never seen, with zero labels from that network. On CICIDS2017, kept hidden until a single final run, it caught 94% of attacks with 1.7% false alarms, which is 96% balanced accuracy and F1 0.86. It catches floods, port scans and FTP password guessing. It still misses quiet attacks such as web attacks, slow DoS and SSH password guessing. The mid-semester report is a separate document. This one explains how we got here, so anyone on the team can pick up the work.

## The datasets: what they are and where we got them

We used seven data files covering five lab networks. Two came in the original project zip. The rest we downloaded, because the official sites either needed a sign-in or were too slow.

| Dataset | Made by | What it is | Where we got it | How we checked it |
|---|---|---|---|---|
| UNSW-NB15 (original CSVs) | UNSW Canberra Cyber Range Lab, 2015 [1] | Lab network with 9 attack types: Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms. Four full-capture files plus the official training and testing files. | Project zip, `NetShield/data/raw/UNSW-NB15` | Used as given. The official files are 64% attack; the full capture is 12.6%. |
| CICIDS2017 (MachineLearningCVE CSVs) | Canadian Institute for Cybersecurity, University of New Brunswick, 5 working days in July 2017 [2] | Office network with DoS, DDoS, PortScan, FTP and SSH password guessing, web attacks, Bot, infiltration and Heartbleed. No IP addresses or times. | Project zip, `NetShield/data/raw/CICIDS2017` | 2,830,743 rows in 8 files. |
| CICIDS2017 with IP addresses (GeneratedLabelledFlows) | Same release as above [2] | The same flows, plus source and destination IP, ports and timestamp. Needed for per-computer features. | The official site redirects downloads to a registration page. We used the public Hugging Face copy [bencorn/CICIDS2017](https://huggingface.co/datasets/bencorn/CICIDS2017) (284 MB zip). | Zip integrity test passed. Its 2,830,743 rows match the project zip exactly, file by file. |
| NF-UNSW-NB15-v3 | University of Queensland, NetFlow v3 release [3] | UNSW-NB15 re-extracted with the nProbe tool into 53 NetFlow features, with IPs and timestamps. 2,365,424 flows, 5.4% attack. | Official links on the [UQ NIDS datasets page](https://staff.itee.uq.edu.au/marius/NIDS_datasets/) return "Unauthorized" without a sign-in. We used the public Parquet copy [keys-i/netFlow](https://huggingface.co/datasets/keys-i/netFlow). | Row and attack counts match the official page. |
| NF-ToN-IoT-v3 | University of Queensland [3]; original capture UNSW Canberra [4] | Smart devices plus normal computers, 2019. Backdoor, DoS, DDoS, injection, man-in-the-middle, password, ransomware, scanning, XSS. 27,520,260 flows, 39% attack. | Same Hugging Face copy | Row and attack counts match the official page. |
| NF-CSE-CIC-IDS2018-v3 | University of Queensland [3]; original capture by the Communications Security Establishment and the Canadian Institute for Cybersecurity [2] | Large lab network on Amazon's cloud, 2018. DoS, DDoS, brute force, Bot, infiltration, web attacks. 20,115,529 flows, 12.9% attack. | Same Hugging Face copy | Row and attack counts match the official page. |
| NF-BoT-IoT-v3 | University of Queensland [3]; original capture UNSW Canberra [5] | Smart-device network almost entirely under attack: DoS, DDoS, reconnaissance, theft. 16,933,808 flows, 99.7% attack. | Same Hugging Face copy | The attack share matches. The official page says 16,993,808 rows, which looks like a typo. |

We also tried the corrected CICIDS2017 release by Engelen et al. [6], from [distrinet-research.be](https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/). The server sent about 18 KB per second and stopped at 57 of 344 MB, so it is not used yet. The partial file is in `corrected_cicids/`.

The Hugging Face NetFlow copy states that it passed the University of Queensland checksum checks before conversion. We did not compare checksums ourselves, only row and attack counts.

**What we did to the data**

- **Removed exact duplicate flows.** Lab tools repeat identical flows, and duplicates inflate scores. CICIDS2017 went from 2,830,743 to 2,522,592 rows.
- **Sampled the largest files** to at most 3 million flows before removing duplicates. For BoT-IoT we kept every normal flow, because only 0.3% of it is normal.
- **Split by clock hour,** alternating unlabelled calibration hours and scored hours. Never a random split.
- **Fixed two format problems.** UNSW's NetFlow file stores the byte rate of zero-length flows as infinity, so we treat those as missing. CICIDS2017 prints afternoon times without AM or PM, so hours before 8 were moved forward 12 hours.
- **Labels:** every file has a 0 or 1 attack label and an attack-type name. The model predicts only the 0 or 1. We used the names only to report which attack types were caught.

## The journey, step by step

1. **Where we started.** Notebooks 01 to 19 trained on UNSW-NB15 and tested on CICIDS2017, using 11 shared flow features plus CORAL, DANN, BBSE, threshold tricks and self-training. Nothing improved, and we did not know why.
2. **We audited the pipeline first.** We found real bugs. CICIDS2017 counts bytes without headers and UNSW-NB15 counts them with headers. CORAL ran on unscaled features and flagged every flow. DANN ran without a log transform and diverged. Duplicates and random splits inflated scores. The official UNSW files are 64% attack, so flagging everything already scored F1 0.81. After the fixes, the best no-label result was still poor: ROC-AUC 0.41 from UNSW to CICIDS and 0.78 in the other direction.
    - **Lesson:** fix the measuring before trying new methods.
3. **We tested what labels can do.** One model trained with labels from both datasets scored F1 0.99 on CICIDS and 0.88 on UNSW. Adding just 500 labelled flows from the new dataset gave F1 about 0.89. So the model can learn; the hard part is having no labels.
4. **We tried the standard fixes for dataset shift.** Separate scaling per dataset, testing only on attack types both datasets share, dropping features that identify the dataset, smarter thresholds, and detectors trained on normal traffic only. The best was 0.68 ROC-AUC on shared attacks. A classifier could still tell the two datasets apart almost perfectly (AUC 1.00).
    - **Lesson:** the datasets differ in what an attack looks like, not just in scale.
5. **We caught our own mistake.** Some early settings were chosen using labels from another part of the test dataset. That breaks the zero-label rule, and those scores did not hold on the final data. From then on, choices could use only other datasets, and the final test stays hidden until one run.
6. **We tried datasets with identical features.** The NetFlow v3 versions of UNSW, ToN-IoT and CSE-CIC-IDS2018 share the same 53 features. Five of six train and test directions were still at or below chance, while each dataset scored 0.97 to 1.00 on itself.
    - **Lesson:** same features are not enough. Each lab has its own normal traffic and its own attack tools.
7. **Version 1: look at computers, not single flows.** Our teacher confirmed zero labels is a hard requirement. Instead of one flow at a time, we described what each computer did in one minute: how many computers and ports it contacted, how many connections got no reply, how often it hit the same port again. We trained on three networks at once and kept BoT-IoT hidden. It scored ROC-AUC 0.97, catching 93% of attacks with 4.7% false alarms, but only 7% of scans.
8. **Error analysis showed two problems.** Many clues flip direction between networks: attackers made more connections than normal computers on three networks, but fewer on UNSW. And the training attacks hit few ports, so the model learned that few ports means attack, which is the opposite of a scan.
9. **Version 2 fixed both.** We added 10- and 60-minute windows, a check on how many computers a source hits on the same port, and a check on how unusual a minute is for that computer. We also kept only clues that point the same way on every training network, and spread suspicion across a computer's flows in the same 10 minutes. We developed it on four NetFlow datasets and locked the best of 80 setups. Then we ran it once on CICIDS2017, using a release with IP addresses that matched our copy row for row. Result: 96% balanced accuracy, PortScan caught 99.6% of the time (it was 0% before).

## The final method, simply

The model answers one question per connection: attack or normal. It never predicts the attack type.

1. **Read five fields from every connection:** sender IP, receiver IP, receiver port, start time, and whether a reply came back. Every dataset has these, whichever tool recorded it.
2. **Describe each computer's behaviour** over the last 1, 10 and 60 minutes: how many ports it tried, how spread out they were, how often it hit the same port again, and how unusual this minute is for that computer. IP addresses are only used for counting, never as clues.
3. **Keep only trustworthy clues.** A clue stays only if it points the same way on every training network. Of 40 candidates, 10 survived.
4. **Train one LightGBM model** on four labelled networks (UNSW, ToN, CIC2018, BoT-IoT), each network weighted equally.
5. **Adapt to the new network without labels.** The model scores the new network's odd clock hours, then retrains on its most confident guesses there (self-training).
6. **Smooth by computer.** Each connection's score is averaged with its computer's mean score over the same 10 minutes.
7. **Pick the cut-off without labels,** using Otsu's method on the odd-hour scores. Even hours are the ones we mark.

We also tried an Isolation Forest that learns the new network's normal traffic by itself. It did worse on development data, so it is not in the final method.

## Results and how to read them

| Hidden test | Method | Balanced accuracy | F1 | Attacks caught | False alarms |
|---|---|---|---|---|---|
| CICIDS2017 | Version 2 | 0.96 | 0.86 | 94% | 1.7% |
| CICIDS2017 | Version 1 | 0.82 | 0.39 | 78% | 14.5% |
| BoT-IoT | Version 1 | 0.94 | 0.96 | 93% | 4.7% |

**What the scores mean**

- **Balanced accuracy:** the average of "attacks caught" and "normal traffic left alone". Guessing, or flagging everything, scores 0.50. Use this as the headline number.
- **Plain accuracy:** misleading when one class dominates. BoT-IoT is 98% attacks, so flagging everything there scores 98% accuracy and is useless.
- **F1:** combines how many attacks we caught with how often a flag is right. On CICIDS2017 only 6 in 100 scored flows are attacks, so even a few false alarms pull F1 down. Flagging everything scores 0.12 there.
- **ROC-AUC:** how well the model ranks attacks above normal traffic, ignoring the cut-off. 0.5 is a coin flip.

**How it does by attack type (version 2, CICIDS2017)**

- **Caught almost all:** DoS Hulk and DoS GoldenEye (100%), PortScan (99.6%), FTP password guessing (100%).
- **Missed completely:** slowloris (slow DoS), SSH password guessing, web attacks (brute force, XSS, SQL injection), Bot, infiltration and Heartbleed.
- **Careful when presenting:** DoS Hulk is 61% of the attacks in the test hours, so the overall score leans on it. DDoS and DoS Slowhttptest fell entirely in the unlabelled calibration hours, so they were not scored at all.
- **One regression:** version 1 caught 54% of slowloris and version 2 caught none. We have not found which change caused it.

## Files and how to re-run

Everything sits in one working folder, `netshield-work`, outside any other project.

| Path | What it holds |
|---|---|
| `NetShield/REPORT.md` | Full technical write-up, sections 1 to 10, every number with its caveats |
| `NetShield/notebooks/20_final_results.ipynb` | Displays all saved results, including the version 2 hidden test and a per-attack chart. It does not retrain. |
| `NetShield/src/run_final.py`, `run_joint_full.py`, `extra_fixes.py` | Fixed UNSW-NB15 and CICIDS2017 pipeline (report sections 1 to 7) |
| `NetShield/src/nf_transfer.py` | Same-features test on NetFlow v3 (section 8) |
| `NetShield/src/nf_zero_label.py` | Version 1, hidden test on BoT-IoT (section 9) |
| `NetShield/src/zero_label_v2.py` | Version 2, hidden test on CICIDS2017 (section 10) |
| `NetShield/outputs/` | Every results table, one folder per stage |
| `NetShield/data/raw/` | The original UNSW-NB15 and CICIDS2017 CSVs from the project zip |
| `nf_data/` | Four NetFlow v3 Parquet files, from the Hugging Face copy `keys-i/netFlow` |
| `cic2017_ip/` | `GeneratedLabelledFlows.zip`, CICIDS2017 with IP addresses, from the Hugging Face copy `bencorn/CICIDS2017` |
| `corrected_cicids/` | Partial download of corrected CICIDS2017 (57 of 344 MB, not usable yet) |
| `venv/` | Python 3.12 environment; `NetShield/requirements.txt` lists the packages |

**Re-running version 2** (from `netshield-work`):

```
# features for all five datasets, about 3 minutes
venv/bin/python -I NetShield/src/zero_label_v2.py build

# 80 setups on the four NetFlow datasets, about 10 minutes
venv/bin/python -I NetShield/src/zero_label_v2.py dev
```

The `hidden` step has already run, and the script refuses to run it again. That is on purpose (next section).

## Rules that keep our results honest, and what is left

**Rules**

- **Test labels only mark answers.** No training, scaling, cut-off or method choice may use labels from the dataset being tested.
- **A hidden test is used once.** After you see its score, any change you make is tuned to it. BoT-IoT and CICIDS2017 are now both spent as hidden tests. A new method needs a new untouched dataset.
- **Always show the trivial baseline.** Put the flag-everything score beside every F1 or accuracy.
- **Report each attack type,** not only the total, so the misses are visible.
- **Remove duplicates and split by time,** never randomly.

**Still to do**

- Find a fresh dataset for the next hidden test. Corrected CICIDS2017 from Engelen et al. is the first candidate, but its server was very slow.
- Add per-computer clues for quiet attacks: long-open connections (slow DoS) and repeated failed logins (SSH).
- Find out why version 2 lost slowloris, which version 1 caught 54% of the time.
- Run UNSW-NB15 alone to CICIDS2017 with the new features, clearly marked as a second look at CICIDS2017.
- Before submitting, check the reference list in the mid-semester report against the original papers, and fill in team names.

## Dataset sources

1. N. Moustafa and J. Slay, "UNSW-NB15: a comprehensive data set for network intrusion detection systems," MilCIS, 2015.
2. I. Sharafaldin, A. Habibi Lashkari and A. A. Ghorbani, "Toward generating a new intrusion detection dataset and intrusion traffic characterization," ICISSP, 2018. Covers CICIDS2017 and CSE-CIC-IDS2018.
3. M. Luay et al., "Time matters: temporal NetFlow features for ML-based network intrusion detection," IEEE Access, vol. 14, 2026, and the [UQ NIDS datasets page](https://staff.itee.uq.edu.au/marius/NIDS_datasets/).
4. N. Moustafa, "A new distributed architecture for evaluating AI-based security systems at the edge: Network TON_IoT datasets," Sustainable Cities and Society, vol. 72, 2021.
5. N. Koroniotis et al., "Towards the development of realistic botnet dataset in the Internet of Things for network forensic analytics: Bot-IoT dataset," Future Generation Computer Systems, vol. 100, 2019.
6. G. Engelen, V. Rimmer and W. Joosen, "Troubleshooting an intrusion detection dataset: the CICIDS2017 case study," IEEE Security and Privacy Workshops, 2021.
