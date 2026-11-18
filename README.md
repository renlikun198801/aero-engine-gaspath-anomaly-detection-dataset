# Aero-Engine Gas-Path Anomaly Detection Dataset

Simulated gas-path data from a twin-spool, separate-exhaust turbofan, intended for **one-class / semi-supervised anomaly detection** under measurement noise. Every file is a flat CSV with 2,000 samples, 12 detector input features, 8 ground-truth health parameters, and a binary label.

Companion dataset for the manuscript *Flight-Envelope-Adaptive One-Class Anomaly Detection for Aircraft Engine Gas-Path Health Monitoring* (under review).

## Contents

```
data/
├── train_2000_noise0x.csv     # 2000 healthy samples
├── train_2000_noise1x.csv
├── train_2000_noise2x.csv
├── train_2000_noise5x.csv
├── train_2000_noise10x.csv
├── val_2000_noise{0,1,2,5,10}x.csv    # 1000 healthy + 1000 anomalous
└── test_2000_noise{0,1,2,5,10}x.csv   # 1000 healthy + 1000 anomalous
```

- Total: 15 CSV files, 2,000 rows each, about 12 MB.
- The suffix `noisex` gives the measurement-noise level, where `1x` is the baseline sensor noise and `10x` is ten times that standard deviation.
- **Files of the same split are row-aligned**: row *i* has the same health state, the same degraded parameter, and the same label in every `noisex` file. This makes noise-level comparisons paired, so per-row differences isolate the effect of noise. Flight conditions (`alt`, `Ma`, `dtamb`) are re-drawn per file and therefore differ slightly between noise levels.

## Columns

| Group | Columns | Notes |
|---|---|---|
| Flight condition | `alt`, `Ma`, `dtamb` | Altitude (m), Mach number (–), ambient temperature offset (K) |
| Gas-path measurements | `T2`, `P2`, `N1`, `N2`, `Wf`, `T3`, `P3`, `T5`, `P5` | Station temperatures (K), pressures (Pa), spool speeds (fraction of design), fuel flow (fraction of design) |
| Extra measurements | `N1_corr`, `FN` | Corrected fan speed and net thrust; **not** used as detector inputs in the paper |
| Health parameters | `SE_LPC`, `SW_LPC`, `SE_HPC`, `SW_HPC`, `SE_HPT`, `SW_HPT`, `SE_LPT`, `SW_LPT` | Ground-truth isentropic efficiency (`SE_`) and flow capacity (`SW_`) scalars per component, normalized so that 1 denotes the degradation-free engine |
| Label | `label` | `1` = healthy, `-1` = anomalous |

The 12 detector input features used in the paper are:

```
alt, Ma, dtamb, N1, T2, P2, N2, Wf, T3, P3, T5, P5
```

The three distance features used by the slack network are `T2`, `N2`, `N1`.

## Label convention

- **Training files contain healthy data only** (`label = 1`, 2,000 rows). This reflects the deployment situation in which fault samples are unavailable.
- **Validation and test files are balanced** (1,000 healthy, 1,000 anomalous), so that recall- and false-alarm-oriented metrics can be computed without resampling.
- The label follows the LIBSVM convention (`-1` for the negative/anomalous class).

## Released operating envelope

The released files cover the cruise band: `alt` 3,000–4,000 m, `Ma` 0.4–0.6, `dtamb` 5–10 K. Within this band, healthy sample-to-sample variation in `T2`, `N2`, and `N1` is dominated by the flight condition rather than by health state, which is what motivates condition-dependent decision boundaries.

## Degradation modes

Eight single-component degradation modes are injected, one per sample: exactly one of the eight health parameters is displaced, leaving the other seven at their baseline values.

| Component | Efficiency parameter | Flow-capacity parameter |
|---|---|---|
| Low-pressure compressor (LPC) | `SE_LPC` | `SW_LPC` |
| High-pressure compressor (HPC) | `SE_HPC` | `SW_HPC` |
| High-pressure turbine (HPT) | `SE_HPT` | `SW_HPT` |
| Low-pressure turbine (LPT) | `SE_LPT` | `SW_LPT` |

Sign conventions and magnitudes, as verified from the files:

- Every engine — healthy or not — carries a small baseline loss drawn uniformly from 0 to 1 in the degradation direction: efficiencies and compressor flow capacities lie in `[0.99, 1.00]`, turbine flow capacities in `[1.00, 1.01]`.
- An anomaly adds a further 4–6% displacement to **one** parameter: efficiency drops by 4–6%, compressor flow capacity drops by 4–6%, and turbine flow capacity rises by 4–6%.
- The eight modes are approximately balanced: in a 1,000-anomaly test file each mode accounts for roughly 110–140 samples.

The parameter values are simulation ground truth; they are therefore **targets, not inputs**, and must be excluded from the feature matrix. They are also the natural way to check per-component detection, i.e. to attribute an alarm to the component that actually degraded.

## How the data were generated

A zero-dimensional, component-level thermodynamic model of a twin-spool, separate-exhaust turbofan:

1. Flight condition (`alt`, `Ma`, `dtamb`) sets the inlet total temperature and pressure.
2. Component maps and isentropic relations give the LPC, HPC, HPT, and LPT outlet states.
3. Spool work balance and mass-flow continuity close the cycle.
4. An off-design operating point is obtained by a Newton–Raphson solve on the spool speeds and fuel flow.
5. Health parameters are applied as multiplicative deviations on component efficiency and flow capacity before the cycle is re-solved.
6. Measurement noise is multiplicative Gaussian, applied to the sensed quantities; the `noisex` factor multiplies the baseline standard deviation.

Because the engine model is analytic and deterministic, the labels are exact: there is no labelling ambiguity, only measurement noise.

## Loading example

```python
import pandas as pd

FEATURES = ["alt", "Ma", "dtamb", "N1", "T2", "P2", "N2", "Wf", "T3", "P3", "T5", "P5"]
HEALTH   = ["SE_LPC", "SW_LPC", "SE_HPC", "SW_HPC",
            "SE_HPT", "SW_HPT", "SE_LPT", "SW_LPT"]

train = pd.read_csv("train_2000_noise1x.csv")
test  = pd.read_csv("test_2000_noise1x.csv")

X_train = train[FEATURES].to_numpy()          # normal-only training set
X_test, y_test = test[FEATURES].to_numpy(), test["label"].to_numpy()
```

## Suggested evaluation protocol

- Fit on `train_*` only; use `val_*` to calibrate the decision threshold, and report on `test_*`.
- Report precision, recall, F1, false-positive rate, and the recall-weighted $F_{10}$ score, which penalizes missed faults ten times more than false alarms.
- Because the data are balanced in test but normal-only in training, do **not** train a supervised classifier on the test labels.
- Report results per noise level; `0x` is noise-free and is not a realistic operating condition.

## Intended use and limitations

The data are simulated. They reproduce the qualitative behaviour of gas-path measurements (nonlinear coupling, condition-dependent spread, weak fault signatures under noise) but they are not a substitute for run-to-failure or fleet data. The degradation modes are single-component and steady-state; transient operation, multi-fault combinations, sensor faults, and real engine-to-engine variation are not represented.

## Citation

If you use this dataset, please cite the accompanying manuscript:

```bibtex
@article{ren_nsdeepsvdd,
  author  = {Ren, Likun and Yu, Xiaolin and Xie, Zhenbo and Qin, Qinhai and Xie, Jing},
  title   = {Flight-Envelope-Adaptive One-Class Anomaly Detection for Aircraft Engine Gas-Path Health Monitoring},
  journal = {IEEE Aerospace and Electronic Systems Magazine},
  note    = {under review},
  year    = {2026}
}
```

## License

Released for research use. A `LICENSE` file should be added before publication; CC BY 4.0 is recommended for the data.
