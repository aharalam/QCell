# QCell: Key Findings

A plain record of what was built, what was measured, what it means, and where the project could go next. It includes the results that are flattering and the ones that are not.

## 1. The question

Does a quantum feature map give any measurable benefit for rare-cell classification on single-cell RNA data, after fair, leakage-free tuning and held-out evaluation?

Four models are compared on identical folds, cells and features: linear, degree-2 polynomial and RBF support vector machines, and a quantum-kernel SVM (QSVM).

## 2. Short version

- On real cell-type labels, the quantum model **ties** the best classical models. It does not beat them.
- On a simple planted interaction (XOR), a classical RBF kernel is **slightly ahead**.
- On an engineered signal with the structure of the quantum circuit itself, the quantum model **wins clearly**: 0.874 against 0.725 for the best classical model on Quantum Rings, and ahead in all six further samples tested (section 14).
- The quantum model costs roughly 400 to 800 times more compute than any classical model here.
- The pipeline is reproducible: an identical rerun matched every prediction and metric exactly.

## 3. Where the project stands

| Piece | Status |
|---|---|
| Data loading, labels, sampling | Built and tested |
| Fold-safe 8-feature pipeline | Built and tested |
| Synthetic XOR control | Built and tested |
| Classical SVM benchmarks with nested cross-validation | Built, tested, run |
| Quantum kernel on the Quantum Rings SDK | Built, verified against a reference simulation, run |
| End-to-end runner and bundle exporter | Built, tested, run three times |
| Test suite | 84 tests passing |
| Pull request | Open: https://github.com/aharalam/QCell/pull/1 |
| Base44 dashboard | Prompts written (`base44_prompts.md`); app not built |
| Seqera / nf-core run | Not done; bundles mark that stage as skipped |
| Circuit-structured target (the quantum win) | Built, tested, run on Quantum Rings, and checked on six further samples (section 14) |
| Kernel-collapse explorer data | Built, tested, exported in the bundle (section 14) |
| Demo video, Devpost submission | Not done |

Three validated bundles exist under `artifacts/` (git-ignored):

| Run | Method | Notes |
|---|---|---|
| `qcell-run-01` | Spec's retry rule (`reps=2`, then `reps=3`) | Kept for comparison |
| `qcell-run-02` | Repetitions and input scale tuned on inner folds | Two targets only |
| `qcell-run-03` | Identical rerun of run 02 | Matches run 02 exactly |
| `qcell-run-04` | Run 02's method plus the third target and the collapse explorer | The one to present |

## 4. Main results

Run 02: 200 cells (all 37 dendritic cells plus 163 others), 5 outer folds, 3 inner folds, seed 42, Quantum Rings simulator. Values are means across outer folds, with the fold-to-fold standard deviation for macro F1.

| Target | Model | Macro F1 | Balanced accuracy | Rare recall | Rare precision |
|---|---|---|---|---|---|
| Dendritic vs rest | Linear | 0.916 ± 0.052 | 0.934 | 0.918 | 0.828 |
| Dendritic vs rest | Poly-2 | 0.922 ± 0.023 | 0.947 | 0.943 | 0.823 |
| Dendritic vs rest | RBF | 0.887 ± 0.056 | 0.939 | 0.971 | 0.733 |
| Dendritic vs rest | QSVM | 0.910 ± 0.067 | 0.960 | 1.000 | 0.771 |
| Synthetic XOR | Linear | 0.552 ± 0.038 | 0.560 | 0.425 | 0.555 |
| Synthetic XOR | Poly-2 | 0.805 ± 0.044 | 0.811 | 0.826 | 0.810 |
| Synthetic XOR | RBF | 0.899 ± 0.032 | 0.901 | 0.912 | 0.887 |
| Synthetic XOR | QSVM | 0.878 ± 0.045 | 0.882 | 0.912 | 0.858 |

Runtime per target: QSVM about 820 to 830 seconds; each classical model 2 seconds or less.

Where RBF and QSVM disagree in this run: 9 dendritic-target cells (QSVM right on 6, RBF on 3) and 12 synthetic-target cells (QSVM right on 4, RBF on 8).

## 5. The good

- **The pipeline is sound.** Every fitted statistic comes from the training fold only. Tests confirm that corrupting held-out labels changes neither the tuning nor the predictions.
- **The quantum simulation is verified.** Quantum Rings states match Qiskit's exact reference to within 1e-7 at 1, 2 and 3 repetitions on 8 qubits.
- **Results are exactly reproducible.** Run 03 reproduced run 02 with zero differing predictions, identical tuned settings and identical metrics for all four models. Only measured runtimes differed.
- **The quantum model is competitive on real data.** Averaged over six samples of cells it ties the best classical model (see section 7).
- **The quantum kernel does learn nonlinear structure.** On XOR it reaches 0.88 where a linear model gets 0.55.
- **There is a real quantum win available.** See section 8.
- **No credentials leaked.** Neither the API key nor the account email appears in any bundle, log or commit.

## 6. The bad

- **No quantum advantage on real cell labels.** Every gap between QSVM and the best classical model is inside the fold-to-fold spread.
- **The headline number from run 02 was luck.** QSVM's 0.960 balanced accuracy and perfect recall on dendritic cells did not hold across other samples (section 7). It should not be presented as an advantage.
- **RBF beats QSVM on XOR** in 4 of 6 samples, with one tie and one loss.
- **Cost.** QSVM takes roughly 400 to 800 times longer than any classical model, for the same or worse accuracy.
- **The untuned quantum kernel fails completely.** With the spec's original settings it scored 0.50 balanced accuracy, predicting "other" for every cell.
- **The winning quantum settings are the least quantum.** Most folds chose the shallowest circuit (`reps=1`) with heavily shrunk inputs. In that regime the kernel behaves much like an RBF kernel, which is why the two agree on about 95% of cells.
- **Classical scores are below the team's earlier figures.** RBF balanced accuracy here is 0.939, against a team-reported 0.980 that was never verified and likely used a different sample or preprocessing.

## 7. Robustness across samples

The whole benchmark was repeated on five other samples (seeds 1 to 5), which draw different "other" cells and different folds. Those five used Qiskit's reference simulator, which matches Quantum Rings to 7 decimal places. Seed 42 is the Quantum Rings run.

**Dendritic vs rest, macro F1 / balanced accuracy**

| Seed | Linear | Poly-2 | RBF | QSVM |
|---|---|---|---|---|
| 1 | 0.917 / 0.930 | 0.950 / 0.952 | 0.916 / 0.936 | 0.898 / 0.919 |
| 2 | 0.931 / 0.914 | 0.948 / 0.931 | 0.917 / 0.947 | 0.944 / 0.958 |
| 3 | 0.899 / 0.922 | 0.889 / 0.942 | 0.948 / 0.978 | 0.947 / 0.969 |
| 4 | 0.915 / 0.934 | 0.937 / 0.957 | 0.941 / 0.976 | 0.956 / 0.982 |
| 5 | 0.865 / 0.878 | 0.912 / 0.945 | 0.917 / 0.927 | 0.902 / 0.905 |
| 42 | 0.916 / 0.934 | 0.922 / 0.947 | 0.887 / 0.939 | 0.910 / 0.960 |
| **Mean** | 0.907 / 0.919 | 0.926 / 0.945 | 0.921 / 0.951 | 0.926 / 0.949 |

QSVM beat RBF in 3 of 6 samples and lost in 3. Mean difference QSVM minus RBF: +0.005 macro F1, -0.002 balanced accuracy. That is a tie.

**Synthetic XOR, macro F1**

| Seed | Linear | Poly-2 | RBF | QSVM |
|---|---|---|---|---|
| 1 | 0.568 | 0.863 | 0.903 | 0.868 |
| 2 | 0.472 | 0.849 | 0.929 | 0.889 |
| 3 | 0.535 | 0.898 | 0.893 | 0.879 |
| 4 | 0.542 | 0.861 | 0.893 | 0.893 |
| 5 | 0.468 | 0.884 | 0.900 | 0.910 |
| 42 | 0.552 | 0.805 | 0.899 | 0.878 |
| **Mean** | 0.523 | 0.860 | 0.903 | 0.886 |

RBF is ahead in 4 samples, tied in 1, behind in 1. Mean difference QSVM minus RBF: -0.017.

## 8. Where quantum does win (exploratory)

In these tests the label is defined by measuring each real cell's own quantum state: the cell's features are loaded into the spec's original circuit (`reps=2`, input scale 1.0), a fixed random measurement is applied, and cells above the median are labelled positive. The features are real; the signal is planted.

| Qubits (features used) | Linear | Poly-2 | RBF | QSVM |
|---|---|---|---|---|
| 2 | 0.673 | 0.682 | 0.950 ± 0.036 | **0.970 ± 0.021** |
| 3 | 0.521 | 0.521 | 0.717 ± 0.056 | **0.865 ± 0.052** |
| 4 | 0.463 | 0.569 | 0.630 ± 0.079 | **0.684 ± 0.057** |

Values are macro F1, 200 cells, 5 outer folds.

- At 3 qubits QSVM leads the best classical model by about 15 points, roughly three times the fold spread. Linear and poly-2 are at chance.
- The inner-fold tuning selected the generating circuit (`reps=2`, scale 1.0) in every fold at 2 and 3 qubits, without being told.
- The advantage sits in a window. At 2 qubits RBF nearly catches up. At 4 qubits 200 cells are too few for any model.

**Caveats, all of which matter:**

- One measurement setting, one sample, local reference simulator, not Quantum Rings.
- Labels were defined from features fitted on all cells.
- The classical side had only the three standard kernels.
- This is an engineered signal. It shows what kind of structure a quantum kernel detects and classical kernels miss at this sample size. It does not show that such signals exist in biology.

Tests of 3-way and 4-way planted interactions were started but stalled on slow classical fits and were stopped. No result.

## 9. Technical findings worth showing

1. **Kernel concentration.** With unscaled inputs on 8 qubits, the average similarity between two different cells was 0.0043, essentially 1/256. The kernel matrix is nearly the identity, so the SVM cannot generalize. Shrinking the input angles restores usable similarities (average 0.13 at scale 0.1, 0.60 at scale 0.05).
2. **Simulator accuracy mode.** The Quantum Rings simulator's default efficiency mode returned states with only 0.71 overlap with the exact reference for this circuit. The pipeline pins it to highest accuracy with double precision. Without the reference check this would have gone unnoticed.
3. **SDK incompatibility.** Current Qiskit builds the ZZ feature map with qubits that have no register, which the Quantum Rings translator cannot map. The gates are copied onto a registered circuit.
4. **Throughput.** One 8-qubit circuit takes about 120 ms on the simulator. Six worker processes bring the effective cost to about 50 ms. Run 02 used about 34,000 circuits and its quantum stage took 1,652 seconds.
5. **The spec's retry rule leaks.** "Rerun at `reps=3` if `reps=2` loses to RBF on the synthetic control" selects a setting from held-out scores. Run 01 followed it (QSVM 0.895 on dendritic cells, 0.873 on XOR). Run 02 replaced it with inner-fold tuning (0.910 and 0.878).

## 10. Departures from the original spec

| Spec | What was done | Why |
|---|---|---|
| Streamlit dashboard | Base44 app reading an exported JSON bundle | Base44 cannot run Python |
| Fixed `reps=2`, `entanglement='full'` | Repetitions (1, 2, 3) tuned on inner folds | Fair tuning, as RBF's gamma gets |
| No input scaling | Input scale tuned on inner folds | The unscaled kernel collapses |
| Retry at `reps=3` on a loss | Dropped | It uses held-out scores |
| Kernel-PCA as "visual proof" of separation | Shown as a projection with a caveat | A projection cannot prove separation |
| Prior reported numbers | Not used anywhere | Unverified |

## 11. What can and cannot be claimed

**Can say**

- The benchmark is leakage-free, verified against a reference simulation, and exactly reproducible.
- On real rare-cell labels, a quantum-kernel SVM matched the best tuned classical kernels across six samples.
- An untuned 8-qubit quantum kernel fails on this data, and here is why and how it was fixed.
- If Route A below is built and holds up: on circuit-structured signals planted in real cell features, the quantum kernel beat every classical kernel tested.

**Cannot say**

- That quantum is better at finding dendritic cells, or any real cell type.
- That the 0.960 from run 02 reflects a real advantage.
- That quantum-structured signals exist in biology.
- Anything about speed or behaviour on quantum hardware. Everything here is simulated.
- Anything clinical. The data is from a healthy donor.

## 12. Potential directions

### A. "Where quantum kernels win, and where they don't" (recommended)

Add the 3-qubit circuit-structured target as a third benchmark and present a map with three regions: real labels are a tie, simple interactions go to classical, quantum-structured signals go to quantum.

- **Gives:** a true "quantum is better here" result from the same fair pipeline.
- **Needs:** a third target in the pipeline, the bundle contract and the Base44 prompts; a Quantum Rings run (short at 3 qubits); repeats across several seeds and measurement settings; stronger classical baselines so the comparison cannot be called soft.
- **Risk:** the advantage could shrink under those checks. It must be labelled as engineered.

### B. Parity on real data (supporting line)

"A quantum kernel matches tuned classical kernels on real rare-cell detection." True across six samples. It is parity, not advantage, and it comes with the cost figure.

### C. Kernel-collapse explorer (presentation layer)

An interactive view where the viewer changes the input scale and watches the similarity heatmap wash out and accuracy fall to chance. It suits a track that asks for a visual explainer, and it turns the main obstacle into the most memorable part of the demo.

- **Needs:** kernel matrices exported at each scale (a few hundred circuits) and one new dashboard section.

### D. Multi-way gene interactions (uncertain)

Targets that depend on three or four features jointly, in the spirit of genes that matter only in combination. Plausible ground for a quantum kernel at small sample sizes, but untested here, and higher-degree polynomial kernels are a direct classical competitor.

### E. Seqera, for the Code of Life track

- **Minimal:** a demonstration run of `nf-core/scrnaseq` on its test profile, attached to the bundle and labelled as not the source of these counts.
- **Better fit:** run this benchmark itself as a small Nextflow pipeline in the Seqera workspace.
- **Full spec flow:** real PBMC 3k reads through nf-core into this pipeline. Heaviest and riskiest.

### F. Honest null result, as is

Present runs 01 to 03 and the six-sample table as a careful negative result with three concrete findings (collapse, simulator mode, retry-rule leak). Credible, lowest effort, least exciting.

### Not viable

- Claiming an advantage on dendritic cells from run 02 alone.
- Weakening classical baselines, choosing quantum settings by held-out score, or picking a favourable seed.

## 13. Limits of everything above

- One dataset. 200 cells per sample, of which 37 are dendritic and are the same 37 in every sample.
- Cell-type labels are inferred from clustering of the same dataset and are not externally validated.
- The rare class is deliberately enriched (18.5% of the sample against 1.4% of the dataset), so precision is higher than it would be at natural prevalence.
- Six samples is enough to see that the dendritic result is a tie, not enough for fine statistical claims.
- All quantum results are exact simulations of 8 or fewer qubits, which any laptop can compute.

## 14. Update: third target and collapse explorer on Quantum Rings

Run `qcell-run-04` added the circuit-structured target (Direction A) and the kernel-collapse explorer data (Direction C) to the pipeline. The dendritic and XOR results in this run are identical, prediction for prediction, to run 02.

### Circuit-structured target, Quantum Rings

200 cells (100 positive by construction), 5 outer folds, seed 42. Three input features, 3 qubits.

| Model | Macro F1 | Balanced accuracy | Rare recall | Rare precision |
|---|---|---|---|---|
| Linear | 0.643 ± 0.126 | 0.650 | 0.680 | 0.633 |
| Poly-2 | 0.674 ± 0.086 | 0.680 | 0.770 | 0.652 |
| RBF | 0.725 ± 0.035 | 0.725 | 0.700 | 0.739 |
| QSVM | **0.874 ± 0.025** | **0.875** | 0.900 | 0.863 |

- QSVM leads the best classical model by 0.149 macro F1. Its worst fold (0.848) is above RBF's best fold (0.774), and the bootstrap intervals do not overlap (QSVM 0.825 to 0.919, RBF 0.660 to 0.785).
- RBF and QSVM disagree on 56 cells. QSVM is right on 43 of them, RBF on 13.
- Inner-fold tuning chose `reps=2` at input scale 1.0 in every fold, which is the circuit that generated the labels.
- QSVM took 381 seconds on this target.

### Robustness across six further samples

Each seed changes the cells drawn, the folds, and the random measurement that defines the labels. These used the local reference simulator. "RBF wide" is RBF with a larger tuning grid, added so the classical side cannot be said to have been held back. Values are macro F1.

| Seed | Linear | Poly-2 | RBF | RBF wide | QSVM | QSVM lead over best classical |
|---|---|---|---|---|---|---|
| 1 | 0.627 | 0.651 | 0.748 | 0.761 | 0.833 | +0.072 |
| 2 | 0.575 | 0.647 | 0.714 | 0.719 | 0.874 | +0.155 |
| 3 | 0.619 | 0.671 | 0.797 | 0.801 | 0.910 | +0.109 |
| 4 | 0.439 | 0.667 | 0.728 | 0.733 | 0.885 | +0.152 |
| 5 | 0.430 | 0.632 | 0.795 | 0.780 | 0.930 | +0.135 |
| 6 | 0.600 | 0.608 | 0.800 | 0.794 | 0.885 | +0.085 |
| **Mean** | 0.548 | 0.646 | 0.764 | 0.765 | 0.886 | +0.118 |

QSVM is ahead in 6 of 6 samples. The wider RBF grid made no real difference (0.765 against 0.764). Tuning chose `reps=2` at scale 1.0 in all 30 folds.

**What this supports:** on a signal with quantum-circuit structure planted in real cell features, the quantum kernel beat every classical kernel tested, consistently.

**What it does not support:** any claim about real cell types, or that such signals occur in biology. The signal is engineered, and the three input features are fitted once on all cells because the labels are defined from them.

### Kernel-collapse explorer

The 8-qubit kernel (`reps=2`) on a fixed subset of 60 cells (20 dendritic), with the held-out accuracy of a QSVM forced to each input scale on all 200 cells.

| Input scale | Mean similarity between different cells | Balanced accuracy | Macro F1 | Cells predicted dendritic |
|---|---|---|---|---|
| 1.0 | 0.0055 | 0.500 | 0.449 | 0 of 200 |
| 0.1 | 0.0963 | 0.860 | 0.848 | 39 |
| 0.05 | 0.4484 | 0.855 | 0.820 | 45 |

A fully collapsed 8-qubit kernel would give every pair 1/256, about 0.0039. At scale 1.0 the kernel is essentially there, and the model predicts no dendritic cells at all.

### Notes on this run

- Validation: 0 errors, 0 warnings. No credentials in the bundle or log.
- QSVM runtimes for the dendritic and XOR targets (1,246 and 954 seconds) are higher than in run 02 because the robustness check shared the processor during the run. The predictions are unaffected.
- Files for the dashboard are in `artifacts/qcell-run-04/dashboard_bundle/`: `results.json`, `cells.json`, `kernels.json`, `run_manifest.json`, `collapse_explorer.json`, and two circuit images.
