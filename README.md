# Credit Risk Portfolio Monitoring

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Download the `accepted_2007_to_2018Q4.csv` Lending Club dataset from
   Kaggle: https://www.kaggle.com/datasets/wordsforthewise/lending-club
3. Place the downloaded file at `data/raw/accepted_2007_to_2018Q4.csv`.

## AI-Augmented Development

<!-- TODO: move this section to the end of the file, after the
     Business Problem / Data & Methodology / Key Findings sections are
     written, so it's not the first thing after the title. -->

This project was built independently, using AI coding agents as a
productivity and quality-assurance tool. I made every substantive
decision: the data source, the variable selection (validated against
LendingClub's official data dictionary), the grade-balanced sampling
design, the decision to simulate monthly delinquency via a
grade-calibrated Markov chain (disclosed in `docs/charter.md`), and the
analytical methods used — vintage analysis, roll-rate analysis, and a
logistic regression PD model.

AI agents also served as a review layer: working independently, without
a manager or code reviewer, I used them to catch things I might
otherwise miss — for example, a step that looked complete but wasn't, or
a `.gitignore` rule silently excluding a file I needed tracked. Every
change was reviewed and approved by me before being committed; these
working principles are documented in `CLAUDE.md`.
