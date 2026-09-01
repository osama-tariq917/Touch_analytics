# Cross-Device Touch-Gesture Biometrics

Code and results for *Touch-Gesture Biometrics Do Not Transfer Across Handsets:
A Within-Subject Cross-Device Evaluation* (ICCST 2026).

The paper asks a question the usual evaluation setup cannot answer: if someone
enrols a touch-behaviour template on their own phone and later switches to a
different one, does the template still recognise them?

Enrolling 40 participants on one handset and testing them on another, within
subject, the median equal error rate rises from **0.23 to 0.47** against a
chance floor of 0.50. Calibrating the raw sensor channels recovers part of that,
reaching **0.34**.

## Headline numbers

| Condition | Median EER |
|---|---|
| Same handset | 0.23 |
| Different handset, no correction | 0.47 |
| Different handset, feature normalisation (transductive) | 0.41 |
| Different handset, feature normalisation (inductive) | 0.46 |
| Different handset, raw-sensor calibration | 0.34 |

Chance is 0.50. Raw calibration improved 34 of the 40 participants.

Two further findings:

- **Fusing strokes helps only when the device matches.** Accumulating twenty
  strokes drives the error rate to near zero on the enrolment handset, but to
  0.64 on a different one. A device mismatch is a systematic bias, and averaging
  sharpens it rather than cancelling it.
- **Handsets are not interchangeable as calibration targets.** Per-handset
  five-fold cross-validation gives error rates from 0.05 to 0.10, and the best
  reference frame (handset 11, at 0.14) is not the device contributing the most
  data (handset 7, at 0.33).

## Datasets

Neither corpus is redistributed here. Both are public and must be obtained from
their authors.

- **Touchalytics** — 41 users, 5 handsets, 912,133 events.
  Frank et al., *IEEE TIFS* 8(1), 2013.
- **BioIdent** — 71 users, 11 handsets, 231,371 events, and the reason this study
  is possible: 45 participants contribute strokes from two or three different
  devices. Antal et al., *Pattern Recognition Letters* 56, 2015.

A note on the Touchalytics file layout that is easy to get wrong: **column 0 is
the handset and column 1 is the participant.** Training on column 0 produces a
five-class handset classifier that looks accurate and measures nothing about
people.

## Layout

```
code/
  touchauth.py           stroke segmentation, 36 features, device normalisation
  per_device.py          per-handset k-fold, ROC, best-template analysis
  reviewer_response.py   inductive protocol, calibration stability, fusion curves
  test_calibration.py    raw-sensor quantile calibration
  export_quantiles.py    builds per-handset quantile tables
  export_tflite.py       trains and exports a quantised on-device model
  make_golden.py         parity fixture for the Android port
  make_diagrams.py       schematic figures
  make_review_figs.py    device, normalisation and ROC figures
results/
  tables/                every CSV behind the paper's tables
  figures/               every figure, including unused variants
paper/
  ICCST2026_paper.pdf
```

## Reproducing

```bash
pip install numpy pandas scikit-learn matplotlib

# 1. extract strokes from the raw corpora
python code/touchauth.py            # see load_touchalytics / load_bioident

# 2. main experiments
python code/reviewer_response.py    # protocol, stability, fusion
python code/per_device.py           # per-handset k-fold and template choice

# 3. figures
python code/make_diagrams.py
python code/make_review_figs.py
```

Scripts expect `strokes_touchalytics.csv` and `strokes_bioident.csv` in the
working directory, produced by `touchauth.build_strokes()` from the raw files.

## Notes on the protocol

Three choices matter more than the model, and reproducing the numbers depends on
keeping them:

1. **Session-disjoint splits.** Enrolment and test never share a session.
   Splitting at the level of individual touch samples puts neighbouring points of
   the same stroke on both sides and inflates results.
2. **Absolute screen coordinates are excluded.** Keeping them lets a model
   memorise *where* a gesture happened instead of *how* it was made.
3. **The device-signal probe is user-disjoint.** Since each Touchalytics
   participant used one handset, the device label is a deterministic function of
   the user; an ordinary split lets a model score highly by recognising the
   person rather than the hardware.

One caveat carried over from the paper: the per-handset cross-validation in
`per_device.py` draws folds from shuffled strokes rather than disjoint sessions.
That is the more permissive protocol, which is why those error rates (0.05–0.10)
are much lower than the 0.20 reported elsewhere. The two are not comparable.

## Citation

```
@inproceedings{salik2026crossdevice,
  title     = {Touch-Gesture Biometrics Do Not Transfer Across Handsets:
               A Within-Subject Cross-Device Evaluation},
  author    = {Salik, Osama Tariq and Ramalingam, Soodamani and
               Garzia, Fabio and Ramzan, Muhammad},
  booktitle = {Proc. IEEE International Carnahan Conference on Security
               Technology (ICCST)},
  year      = {2026}
}
```
