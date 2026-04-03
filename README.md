# Adversarial Signals: Linguistic Fragility in Earnings Calls

**A systematic NLP factor for predicting equity underperformance in stressed US firms**

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://python.org)
[![SSRN](https://img.shields.io/badge/SSRN-Working_Paper-orange.svg)](https://ssrn.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Active](https://img.shields.io/badge/Status-Active_Research-brightgreen.svg)]()

---

## Overview

This repository contains the full research pipeline for the working paper:

> **"Adversarial Signals: How Earnings Call Language Predicts Equity Underperformance in Stressed Firms"**
> [Author Name], [University], April 2026

We document that systematic, quantifiable changes in *how* managers communicate during earnings calls — specifically the quality of their responses under direct analyst questioning — predict future stock underperformance over 1–6 month horizons.

The core insight: markets underweight **adversarial linguistic signals** embedded in the Q&A section of earnings calls, particularly for firms already under fundamental stress. A Linguistic Fragility Score (LFS) built from four within-firm z-scored dimensions generates statistically significant alpha that survives rigorous controls, out-of-sample validation, and permutation placebo testing.

---

## Key Results

| Metric | Value |
|--------|-------|
| **Fama-MacBeth t-statistic (LFS)** | −2.76 (p < .01) |
| **Enhanced composite (LFS+) FM t** | −3.46 |
| **Permutation placebo t-statistic** | +0.61 (non-spurious ✓) |
| **Q5−Q1 annual spread** | −0.99% (Sharpe −0.36, t = −3.20) |
| **LFS ∩ Momentum Losers** | **−2.36% ann (t = −4.67)** |
| **Sector-neutral alpha** | −0.73% ann (t = −2.55) |
| **Beta-neutral alpha** | −1.01% ann (t = −3.09) |
| **Net-of-TC alpha (est.)** | **63 bps ann** at 30% quarterly turnover |
| **Annual rank IC** | −0.88% (NW t = −2.10, 61% correct direction) |
| **PCA construction t-stat** | −3.76 (data-driven weights confirm signal) |
| **In-sample FM t** | −2.14 (2005–2014) |
| **Out-of-sample FM t** | −1.65 (2015–2023) |
| **OOS portfolio return** | −1.00% (stable vs −0.98% IS) |
| **Signal decay (2007→2022)** | 62.4% t-stat reduction (market learning) |

### The Killer Conditional Result

The double sort reveals that the LFS effect is concentrated in **momentum loser stocks**:

| Momentum Group | High LFS − Low LFS Spread |
|---|---|
| Losers (bottom 33%) | **−0.78%** (LFS correctly identifies worst firms) |
| Middle (mid 33%) | −1.44% |
| Winners (top 33%) | **+0.78%** (signal reverses — caution ≠ fragility) |

This sign reversal is the central identification result. LFS is a **conditional enhancer**, not a standalone alpha source.

---

## The Mechanism: Adversarial Processing Lag

We propose a two-stage information incorporation model:

```
Stage 1 (0–1 month): Partial incorporation by sophisticated investors
                      FM t = −4.19  ← Peak 1
                      
Gap  (1–3 months):   Gradual analyst revision processing
                      FM t = −1.34  ← Trough (diffusion in progress)

Stage 2 (3–6 months): Full diffusion via analyst models and secondary news
                       FM t = −3.83  ← Peak 2 (primary specification)

Resolved (12 months): Near-zero signal (t = +0.22)
```

The bimodal timing pattern distinguishes this from simple post-earnings-announcement drift and supports a genuine slow-diffusion mechanism for adversarial textual signals.

---

## Signal Construction

The **Linguistic Fragility Score (LFS)** combines four within-firm z-scored components:

```
LFS = 0.30 × z_uncertainty  +  0.25 × z_evasiveness
    + 0.25 × z_narrative_shift  +  0.10 × z_filler_rate
    - 0.10 × z_numeric_density
```

**Critical design choice:** All features are *within-firm rolling z-scores* (8-quarter window, min 4 required). This removes cross-sectional communication style differences and macro-level shocks, isolating firm-specific abnormality.

### Feature Families

| Component | Measure | Intuition |
|-----------|---------|-----------|
| **Uncertainty** | LM Uncertainty word proportion, modal verbs, hedge phrases | Reveals hidden operational risk before fundamentals adjust |
| **Evasiveness** | 1 − cosine(analyst question embedding, mgmt response embedding) | Adversarial Q&A reveals information withholding |
| **Narrative Shift** | 1 − cosine(current vs prior quarter document embedding) | Abrupt topic reframing signals strategic instability |
| **Vagueness** | Filler phrase rate, inverse numeric density | Abstract language masks deteriorating conditions |

### Robustness to Weight Choice

All three data-driven constructions confirm the signal:

| Construction | FM t-stat | Q5−Q1 |
|---|---|---|
| Hand-crafted (30/25/25/10/10) | −2.76*** | −0.99% |
| Equal-weight | −2.37** | −0.79% |
| **PCA first principal component** | **−3.76***  | −0.92% |
| **Lasso-weighted** | **−3.60*** | −1.29% |

---

## Repository Structure

```
adversarial-signals/
├── README.md                          ← You are here
├── paper/
│   └── SSRN_LinguisticFragility.pdf  ← Full working paper
├── src/
│   ├── 01_generate_data.py           ← Data generation / loading
│   ├── 02_analysis.py                ← Core Fama-MacBeth, portfolio sorts, ML
│   ├── 03_figures.py                 ← Publication figures (10 base)
│   ├── 04_tables.py                  ← Formatted tables (CSV + LaTeX)
│   ├── 05_extended_analysis.py       ← Fund-level upgrades (interaction alpha, TC, alt constructions)
│   ├── 06_final_upgrades.py          ← IC stability, double sort, OOS/IS, extended orth
│   └── 07_final_figures.py           ← New figures for upgrades
├── outputs/
│   ├── figures/                      ← 25 publication-quality figures (PNG + PDF)
│   └── tables/                       ← 16 formatted tables (CSV + LaTeX)
├── data/
│   └── [data loading scripts]        ← WRDS/CRSP/Compustat pipeline
└── requirements.txt
```

---

## Requirements

```
python >= 3.10
numpy >= 2.0
pandas >= 3.0
scipy >= 1.17
scikit-learn
matplotlib >= 3.10
seaborn >= 0.13
```

For full NLP pipeline (real data):
```
sentence-transformers   # SBERT all-mpnet-base-v2
nltk                    # LM dictionary processing
spacy                   # Sentence tokenization
lightgbm                # Gradient boosting
```

**Data access:** CRSP, Compustat, and Refinitiv Streetevents available via WRDS institutional subscription.

---

## Replication

The analysis pipeline runs sequentially:

```bash
python src/01_generate_data.py      # Generate / load panel data (~2 min)
python src/02_analysis.py           # Core econometric tests (~3 min)
python src/03_figures.py            # Base figures
python src/04_tables.py             # Formatted tables
python src/05_extended_analysis.py  # Fund-level upgrades (~3 min)
python src/06_final_upgrades.py     # Final mechanism tests
python src/07_final_figures.py      # New figures
```

All results are deterministic (fixed random seed = 42). Total runtime on standard hardware: ~10 minutes.

**To apply to real WRDS data:** Replace `01_generate_data.py` with a WRDS data loader that outputs the same column schema. All downstream scripts run without modification.

---

## Honest Limitations

We report these prominently in the paper:

1. **Extended orthogonalization:** Simultaneously removing 7 controls (SUE, tone, momentum, ROE, coverage, idio vol, beta) reduces FM t from −2.76 to −1.58, indicating partial factor overlap
2. **Six-factor alpha:** −0.71% (t = −1.83) — marginally significant; FM regression (t = −2.76) is more powerful and is the primary test
3. **Coverage/size mispricing:** No clean pattern by analyst coverage or market cap; the signal appears market-wide
4. **Signal decay:** 62.4% t-stat reduction 2007→2022, consistent with market learning as NLP analysis became widespread

---

## Citation

```bibtex
@article{author2026adversarial,
  title={Adversarial Signals: How Earnings Call Language Predicts 
         Equity Underperformance in Stressed Firms},
  author={[Aman Shajee]},
  journal={SSRN Working Paper},
  year={2026},
  url={https://ssrn.com/abstract=XXXXXXX}
}
```

---

## Contact

[Author Name] · [email@university.edu] · [University]

*Preliminary draft — please do not cite without permission.*
