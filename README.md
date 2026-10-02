# hmm-from-scratch

Do hidden states earn their parameters? Markov chains and hidden Markov models written from scratch in NumPy, judged only on held-out data: 761 days of 5-minute rainfall, and spoken words from speakers the model has never heard. For rain, yes: an 11-parameter HMM predicts held-out days better than a 64-parameter chain. For speech, time structure lifts unseen-speaker digit accuracy from 55 % to 64 %, and to 74 % with mean normalisation.

[![ci](https://github.com/Pchambet/hmm-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/Pchambet/hmm-from-scratch/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-0d9488)
[![License: MIT](https://img.shields.io/badge/license-MIT-64748b)](LICENSE)
[![Report](https://img.shields.io/badge/report-online-0f172a)](https://pchambet.github.io/hmm-from-scratch/)

![Left: survival of dry-spell lengths, observed vs Markov chain vs 3-state HMM. Right: unseen-speaker digit accuracy vs number of hidden states, with and without mean normalisation.](docs/figures/hero.png)

## TL;DR

- **Rain spells.** A first-order Markov chain fitted to 532 days implies that about 0.1 % of dry spells last more than 3 days (0.096 %, the exact tail of its geometric law). Over the next 229 days, 3.8 % did, about 40 times more. A 3-state HMM fitted to the same data predicts 3.7 %. The Kolmogorov-Smirnov distance to the held-out dry spells drops from 0.53 to 0.07.
- **It is not just extra parameters.** On held-out data, a 3-state HMM (11 parameters) gains 2.9 nats per day over the first-order chain; the best observable chain, of order 6 with 64 parameters, gains 2.3. Hidden states are not a free win, though: the 2-state HMM (5 parameters, +1.13) loses to the order-3 chain (8 parameters, +1.55).
- **Speech, seen vs unseen speakers.** One HMM per word, with the feature set and state count chosen on a separate corpus. On the Free Spoken Digit Dataset it scores 93.0 % on the official split, where test speakers also appear in training. With the test speaker left out entirely, it drops to 64.2 %.
- **What closes part of that gap.** Removing each recording's mean cepstrum lifts unseen-speaker accuracy to 74.1 % with 3 states (worst speaker: 53.4 %). Normalisation only pays off with time structure: with 1 state it drops accuracy from 55.4 % to 42.8 %; with 3 states it lifts it from 64.2 % to 74.1 %.
- **Noise cuts both ways** (fruit-word corpus, one speaker, 5-fold CV). Under test-time white noise, the 3-state MFCC model holds 98 % at 20 dB, against 60 % for filterbank features and 48 % for the raw spectrum. At 10 dB and below, the time-blind 1-state model degrades more gracefully (69 % vs 55 % at 10 dB).

## Why it matters

Hidden-state models are the standard tool for anything that switches regime: weather, machine health, demand, speech. A model with more parameters fits its training data better almost by construction, so in-sample fit proves nothing. In planning, the cost of rain or downtime sits in the long spells, and a model that gets the mean right but the tail wrong understates that risk. Here every model is judged on held-out spells and held-out likelihood, against observable chains of equal or larger size, and the cases where hidden states lose are reported too.

## Approach

```mermaid
flowchart LR
  A[5-min rain record<br/>761 days] --> B[Chronological split<br/>532 / 229 days]
  B --> C[Markov chains, order 1-6<br/>HMMs, 2-4 states<br/>Baum-Welch, 4 starts]
  C --> D[Held-out log-likelihood<br/>on identical steps]
  C --> E[Simulated spell lengths<br/>vs held-out spells]
  F[Fruit words, 1 speaker] --> G[CV: feature x states<br/>model selection]
  G --> H[FSDD, 6 speakers<br/>official split + leave-one-speaker-out]
  G --> I[Test-time noise sweep]
```

1. **Models.** Forward-backward, Baum-Welch and Viterbi in about 250 lines of NumPy (`src/hmm_markov/hmm.py`). Sequences of unequal length run as one padded batch, and the recursions stay stable for left-to-right models with extreme likelihood ratios. Tests check them against brute-force enumeration of every state path.
2. **Rain.** Binary wet/dry series: a 5-minute step counts as wet when its accumulation exceeds 0.1 (presumably mm; the course file does not state the unit). The record is cut into weeks. Every model scores the same held-out steps, conditioned on the same six preceding symbols, so chains of order 1 to 6 and HMMs share one likelihood scale. A test checks that an HMM with directly observed states scores exactly like a Markov chain.
3. **Words.** Log spectrum (129 dims), log mel filterbank (26) or MFCC (12) frames. One left-to-right diagonal-Gaussian HMM per word, with a flat-start initialisation. A recording is assigned to the word whose model gives the highest likelihood. The selection uses repeated 5-fold CV on the fruit corpus only, with ties going to fewer states; the selected setup is then run on FSDD, which played no part in the choice.

## Results

![Held-out log-likelihood gain over a first-order chain](docs/figures/rain_models.png)
Every HMM with 3 or more states beats every observable chain on the 229 held-out days, including the order-6 chain with almost six times as many parameters. The 2-state HMM sits between the order-2 and order-3 chains. The 4-state HMM (19 parameters) gains 3.19 nats per day: the held-out record becomes about 24 times more likely each day, and held-out log-loss falls from 79.8 to 63.8 millibits per 5-minute step (-20 %).

| Model | params | held-out log-loss (millibits/step) | KS dry | KS wet | P(dry > 3 days) |
|---|---:|---:|---:|---:|---:|
| Markov chain, order 1 | 2 | 79.8 | 0.527 | 0.233 | 0.1 % (exact) |
| Markov chain, order 3 | 8 | 72.0 | 0.350 | 0.124 | 0.5 % |
| Markov chain, order 6 | 64 | 68.4 | 0.226 | 0.036 | 1.2 % |
| HMM, 3 states | 11 | 65.2 | 0.073 | 0.170 | 3.7 % |
| HMM, 4 states | 19 | 63.8 | 0.069 | 0.039 | 4.3 % |
| Observed, held-out | | | | | 3.8 % |

The fourth state is spent on rain itself: two wet regimes instead of one bring the wet-spell KS distance from 0.17 down to 0.04. Long-memory chains also get wet spells right (order 6: 0.04), but their dry-spell tail stays far too thin.

![Viterbi decoding of the wettest held-out week](docs/figures/rain_week.png)
The three states read as dry weather (it rains 0.017 % of the time), showery weather (2.2 %) and rain (98 %). In this week, many short dry gaps between rain bursts are decoded as showery weather rather than dry weather. That is how the model separates a pause in a storm from a genuinely dry period.

![Fruit-word accuracy vs test-time SNR](docs/figures/words_noise.png)
On the fruit-word corpus (one speaker, 5-fold CV), MFCC is the most noise-robust representation at every SNR tested. The ranking between 1 and 3 states flips below 20 dB, which is worth knowing before tuning a model on clean data only.

![Fruit corpus CV grid](docs/figures/words_grid.png)
The selection corpus is saturated: MFCC reaches 99.4 % with a single state and 100 % with 3 or more. It picks a feature set, but it cannot separate model sizes. That is why the speaker-independent test on FSDD is the number to quote.

Full tables and interactive charts are in the [report](https://pchambet.github.io/hmm-from-scratch/); per-speaker numbers and every other output are in `results/*.json`.

## Reproduce

```bash
make setup    # uv sync --locked (Python 3.12)
make data     # checks committed data, downloads FSDD v1.0.10 once (16 MB, verified)
make run      # both experiments: 4 to 6 min in total on an Apple-silicon laptop
make report   # docs/figures/*.png and site/index.html from results/*.json
make test     # 25 tests, < 10 s, no network
```

Disk: about 190 MB for the virtual environment and 26 MB for the downloaded FSDD recordings in `data/raw/` (gitignored). All randomness is seeded (`--seed`, default 0), and the committed results came from that seed.

## Repository layout

```
src/hmm_markov/
  hmm.py        forward-backward, Baum-Welch, Viterbi; categorical and diagonal-Gaussian emissions
  markov.py     k-th order chains, stationary law, spell lengths
  features.py   log spectrum, log mel filterbank, MFCC
  rain.py       experiment 1 (spells, held-out likelihood)
  words.py      experiment 2 (CV selection, FSDD, noise)
  report.py     figures and the static report page
  cli.py        `uv run hmm-markov {data,rain,words,report,all}`
tests/          brute-force checks, parameter recovery, hand-computed cases
data/           RR5MN.mat (rain record), fruits/ (7 words x 15 recordings); raw/ is downloaded
results/        experiment outputs (JSON) behind every number above
coursework/     the original course notebooks and instructor scripts (archived)
```

## Methodology notes and limitations

- **One rain gauge, location undocumented.** 761 consecutive days from the course material; the indicator is re-derived from the stored accumulations and checked. The held-out period is the last 229 days, so it covers a different mix of seasons than the training period. That is a harder test than a random split, and also a noisier one.
- **The HMM tail is better, not right.** Beyond about 3 days, the 3-state HMM thins out faster than the record: dry spells longer than 7 days are 0.3 % simulated vs 1.4 % observed. Its slowest state still has a geometric tail. Explicit-duration (semi-Markov) models would be the next step.
- **EM finds local optima.** Each HMM is the best of 4 starts. For every state count, the best and worst final training log-likelihood differ by less than 0.1 nats, but a global optimum is not guaranteed.
- **Spell-length metrics compare simulations with one realisation.** Simulations run for 3x the record length; only the first-order chain's tail is computed exactly, from its geometric law (its simulation holds just 3 dry spells over 3 days, too few to quote). The KS distances and tail probabilities carry sampling noise of the held-out record itself (423 dry spells), so differences of a few hundredths of KS are not meaningful.
- **The state count was fixed before FSDD was used.** The fruit corpus chose 3 states. On FSDD, more states help slightly (with mean normalisation: 75.6 % with 5 states, 75.8 % with 8), but the 3-state figure chosen in advance is the one quoted.
- **Speech corpora are small.** 105 fruit recordings from one speaker, and 3,000 FSDD recordings from 6 speakers: the leave-one-speaker-out figure averages 6 numbers, ranging from 53 % to 96 %. The pipeline has no deltas, no mixture emissions and no language model. It is a clean baseline, not a competitive recogniser.
- **The noise test uses synthetic white noise** at the utterance's average power, which includes silence. Real noise is coloured and non-stationary.

### About the coursework

This started as a Télécom SudParis lab on Markov models (2025). The instructor provided the audio feature code, the HMM wrapper around `pomegranate`, the rain record and the fruit recordings (their original source is not documented in the course material). `coursework/code/` is instructor-provided course material, partly adapted from [python_speech_features](https://github.com/jameslyons/python_speech_features) (J. Lyons, MIT licence); the one exception is `exercise1_markov_chain.py`, the instructor's fill-in template completed by me. The folder is not covered by this repository's licence. The archived notebooks in `coursework/notebooks/` are my lab answers and are kept as submitted; they depend on the old `pomegranate` API and on data paths that no longer exist, so they do not run as they are. They contain errors that this package fixes, listed here for transparency:

- Part 1 reuses variable names across cells, so its final stationary-distribution check mixes two parameterisations.
- In part 2, the `pomegranate` Baum-Welch stopped after 3 iterations, the last with a negative improvement. The resulting model rains 1.0 % of the time, against 4.2 % observed.
- Part 3 fails on a `pomegranate` API mismatch and never produced accuracy numbers.

Everything above the coursework section is new code written for this repository, and it does not depend on `pomegranate`.

## References

- L. R. Rabiner, "A tutorial on hidden Markov models and selected applications in speech recognition", *Proc. IEEE* 77(2), 1989.
- K. R. Gabriel and J. Neumann, "A Markov chain model for daily rainfall occurrence at Tel Aviv", *Q. J. R. Meteorol. Soc.* 88, 1962.
- J. P. Hughes, P. Guttorp and S. P. Charles, "A non-homogeneous hidden Markov model for precipitation occurrence", *J. R. Stat. Soc. C* 48(1), 1999.
- S. B. Davis and P. Mermelstein, "Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences", *IEEE Trans. ASSP* 28(4), 1980.
- J. Lyons et al., python_speech_features, [github.com/jameslyons/python_speech_features](https://github.com/jameslyons/python_speech_features), MIT licence (basis of the instructor's feature code in `coursework/code/`).
- Z. Jackson et al., Free Spoken Digit Dataset v1.0.10, [github.com/Jakobovski/free-spoken-digit-dataset](https://github.com/Jakobovski/free-spoken-digit-dataset), CC BY-SA 4.0 (downloaded, not redistributed here).

---

Built by [Pierre Chambet](https://github.com/Pchambet) — decision science for operations under uncertainty.
