# Quantum vs. Classical Rare-Cell Discovery — Base44 Build Prompts

Prompts to paste into Base44, in order, to build the dashboard described in `quantum_vs_classical_spec.md` with the Hack The Nest 2026 toolset.

**Fastest path to a demo:** Prompts 1 to 5 are the demo. Prompts 6 to 8 are polish. Publish after Prompt 3 so a live link exists early, then keep shipping.

---

## 1. Read this first: what Base44 builds and what it cannot

Base44 generates a hosted web app (React front end, built-in database, login, file storage, hosting). Its backend runs JavaScript, not Python. **It cannot run Scanpy, scikit-learn, Qiskit, the Quantum Rings SDK, or Nextflow.**

So the project splits in two:

| Part | Runs where | Tool from the hacker packet | Track it serves |
|---|---|---|---|
| Raw data to processed counts | Seqera Platform workspace, `nf-core/scrnaseq` (only if starting from FASTQ) | Seqera, nf-core, Co-Scientist | Code of Life |
| QC, annotation, split-safe features, classical SVMs | Your laptop or Seqera, Python 3.11+ (Scanpy, scikit-learn) | — | Code of Life |
| Quantum kernel (ZZFeatureMap, QSVM) | Python with `QuantumRingsLib` + `quantumrings-toolkit-qiskit` | Quantum Rings SDK | Quantum Leap |
| Dashboard, artifact upload, validation, live link | **Base44** | Base44 | Best Use of Base44 |

The Python pipeline exports a small bundle of JSON files. A team admin uploads the bundle into the Base44 app. The app only displays what was uploaded. That is what keeps every number on screen a measured number.

This replaces the Streamlit dashboard in the spec (sections 2.2, 9, 12). Everything else in the spec stands.

**Do not paste into Base44, commit, or screenshot:** the Quantum Rings API key, the Quantum Rings promo code, the Base44 access code, or any other partner code. The Base44 app needs no secrets at all.

---

## 2. How to use this file

1. Create a new Base44 app. Paste **Prompt 1** as the very first message. It is long on purpose: it carries the rules and the data contract that every later prompt relies on.
2. Paste one prompt at a time. Open the preview and check the "Done when" list under each prompt before sending the next one.
3. If something is wrong, fix it with a short, specific follow-up ("the confusion matrix rows are swapped: rows must be true labels") rather than re-sending the whole prompt.
4. If your Base44 workspace has a plan or discuss mode, use it for questions so you do not spend build messages on them.
5. Until the real bundle exists, build against the **placeholder bundle** (Prompt 1 creates the loader). The app marks placeholder data with a banner on every page.

---

## 3. For whoever owns the Python pipeline

The app reads four JSON files and two images. The exact shapes are in the **DATA CONTRACT** block inside Prompt 1. That block is the single source of truth: if the pipeline's export changes, change it there and tell Base44.

`modeling/src/export_results.py` writes the bundle for you. Call `export_bundle(out_dir, manifest, targets)` from the pipeline with each target's labels, folds, per-model out-of-fold predictions, measured timings, display coordinates and the three display kernel matrices. It computes the metrics, confusion matrices, bootstrap intervals and Kernel-PCA coordinates, runs the same checks the app runs on upload, and writes nothing if any check fails. Two commands are useful on their own. Both need `numpy` and `scikit-learn` installed in the Python you run them with.

Write a random bundle flagged `is_placeholder`, for testing the app's upload page before real results exist:

```bash
python3 modeling/src/export_results.py placeholder placeholder_bundle
```

Check any bundle folder the way the app will. This checks the placeholder one; for a real run, replace `placeholder_bundle` with that run's bundle folder:

```bash
python3 modeling/src/export_results.py validate placeholder_bundle
```

Things the export must get right, because the app checks them on upload:

- Every benchmark cell appears exactly once in `cells.json`, with the outer fold where it was held out and one out-of-fold prediction per model.
- All four models use the same cells and the same folds.
- Fold metrics in `results.json` match what the app recomputes from the per-cell predictions.
- Kernel matrices are square, share one `cell_ids` ordering, and are computed on the same cells for RBF, poly-2 and quantum.
- Kernel-PCA coordinates are computed in Python (`sklearn.decomposition.KernelPCA(kernel="precomputed", n_components=3)`) on those same matrices. The app does not do linear algebra.
- Kernels and Kernel-PCA shown in the app come from a display-only fit. Say so in `fit_scope`. They are never the matrices used for scoring.
- The QSVM's repetitions and input scale are tuned on inner folds, so each fold's choice is in `best_params` (`reps`, `scale`, `C`). If a run ever tries several fixed configurations instead, export every attempt. The app shows all of them.

---

## 4. The prompts

### Prompt 1 — Foundation: rules, data model, upload and validation, app shell

```text
Build a web app called "Quantum vs Classical: Finding Rare Cells".

WHAT IT IS
A research benchmark dashboard for computational biologists and hackathon judges. It compares classical support vector machines (linear, degree-2 polynomial, RBF) with a quantum-kernel SVM (QSVM) on two tasks:
1. Biological: dendritic cells vs all other cells in the 10x Genomics PBMC 3k dataset (healthy-donor blood cells).
2. Synthetic control: a planted two-feature interaction (XOR-like) vs background.
All modeling happens outside this app in Python. This app never computes model results. It only displays measured results that an admin uploads as files.

HARD RULES (apply to every page, now and in every later change)
R1. Never invent, seed, hard-code or auto-generate any score, cell, prediction, runtime, label or kernel value. Every number shown must come from an uploaded bundle. If data is missing, show an empty state that says what is missing.
R2. Never say or imply that quantum beats classical, or the reverse, except in a sentence generated from the uploaded numbers.
R3. The data is from a healthy donor. Never use the words sick, diseased, cancer, tumor, patient, diagnosis, resistant or drug target for these cells. The positive biological class is "Dendritic cell". The app is not a clinical or diagnostic tool.
R4. Anything from the synthetic control must carry a visible "Synthetic control" tag. Never describe it as a discovery in real data.
R5. A 2D or 3D plot is a projection. Never caption a plot as proof that a model separated the cells. Kernel similarity is not a causal explanation.
R6. When a bundle is flagged as a placeholder, show a fixed banner on every page: "PLACEHOLDER DATA: not measured results", and add a "placeholder" watermark to every chart.
R7. The app stores no API keys or secrets.

DATA MODEL
Create one entity, Run:
- run_id (text, required, unique)
- label (text)
- status (enum: draft, complete, failed; default draft)
- is_placeholder (boolean, default false)
- is_featured (boolean, default false)
- manifest_url, results_url, cells_url, kernels_url (file URLs to uploaded JSON files)
- circuit_image_url, seqera_screenshot_url (file URLs to uploaded images, optional)
- validation_report (object: errors array, warnings array, checked_at)
- notes (long text)
Permissions: anyone, including visitors who are not logged in, can read Run records. Only admins can create, update or delete them.

DATA CONTRACT (the four JSON files in a bundle)
"float" means a real number, "int" an integer. Class labels are always 0 or 1, where 1 is the rare/positive class.

File run_manifest.json:
  schema_version: "1.0"
  run_id: text
  created_at: ISO-8601 timestamp
  is_placeholder: boolean
  dataset: { name, source_url, n_cells_raw: int, n_genes_raw: int, n_cells_after_qc: int, donor_status }
  annotation: { method, positive_class_name, limitations }
  sampling: { strategy, n_benchmark_cells: int, n_positive: int, seed: int }
  preprocessing: { feature_pipeline, n_features: int, fitted_within_training_folds_only: boolean }
  cv: { outer_folds: int, inner_folds: int, selection_metric, seed: int }
  quantum: { sdk, sdk_version, backend, n_qubits: int, feature_map, entanglement, attempts: [ { reps: int, is_final: boolean, reason } ],
             input_scale_grid: [float], reps_grid: [int], input_scale_note, simulator_settings, parallel_workers: int, smoke_test: object }   (the last six are optional)
  synthetic_control: { rule_description, features_used: [text], seed: int }
  pipeline_stages: [ { id, name, tool, status: "succeeded" | "failed" | "skipped" | "not_run", started_at, duration_seconds: float, artifact, link } ]
  seqera: { run_url, pipeline, status }   (optional)
  software_versions: object of name -> version

File results.json:
  schema_version, run_id
  targets: { biological: T, synthetic: T }
  T = {
    positive_label_name, negative_label_name, n_cells: int, n_positive: int,
    models: { linear: M, poly2: M, rbf: M, qsvm: M }
  }
  M = {
    display_name,
    config: object,
    aggregate: { balanced_accuracy: {mean, std}, macro_f1: {mean, std}, rare_recall: {mean, std}, rare_precision: {mean, std} },
    ci95: { balanced_accuracy: [low, high], macro_f1: [low, high] }   (optional)
    pooled_confusion_matrix: [[tn, fp], [fn, tp]],
    folds: [ { fold: int, balanced_accuracy, macro_f1, rare_recall, rare_precision, confusion_matrix: [[tn, fp],[fn, tp]], best_params: object, fit_seconds: float, predict_seconds: float } ],
    total_runtime_seconds: float,
    attempts: [ { reps: int, is_final: boolean, aggregate: same shape as above, total_runtime_seconds } ]   (qsvm only, optional)
  }

File cells.json:
  schema_version, run_id
  targets: { biological: C, synthetic: C }
  C = {
    feature_names: [text],
    cells: [ {
      id: text,
      fold: int,                      (the outer fold where this cell was held out)
      y_true: 0 | 1,
      annotation: text,               (cell type name; for synthetic, "planted" or "background")
      coords: { umap3: [x,y,z], pca3: [x,y,z] },
      kpca: { rbf: [x,y,z], poly2: [x,y,z], qsvm: [x,y,z] },
      features: [float],              (same order as feature_names)
      pred: { linear: 0|1, poly2: 0|1, rbf: 0|1, qsvm: 0|1 },
      score: { linear: float, poly2: float, rbf: float, qsvm: float },   (optional, uncalibrated SVM decision values)
      planted: { x1: float, x2: float }                                 (synthetic only)
    } ],
    decision_regions: { x1_range: [min,max], x2_range: [min,max], resolution: int, grids: { linear: [[0|1]], poly2: [[0|1]], rbf: [[0|1]], qsvm: [[0|1]] } }   (synthetic only, optional)
  }

File kernels.json:
  schema_version, run_id
  targets: { biological: K, synthetic: K }
  K = {
    cell_ids: [text],                 (one ordering shared by all three matrices)
    fit_scope: text,                  (states that these are display-only fits, not the matrices used for scoring)
    params: { rbf: object, poly2: object, qsvm: object },
    matrices: { rbf: [[float]], poly2: [[float]], qsvm: [[float]] }
  }

ADMIN PAGE: "Upload run" (admins only)
- Six file inputs: run_manifest.json, results.json, cells.json, kernels.json, circuit image (optional), Seqera screenshot (optional).
- On upload, read and parse the JSON in the browser and run these checks before saving:
  E1. All four files parse and have the same run_id and schema_version "1.0".
  E2. Every required field in the contract is present with the right type.
  E3. Every metric is between 0 and 1. Every confusion matrix is 2x2 with non-negative integers.
  E4. In cells.json, cell ids are unique per target, and every cell has pred for linear, poly2, rbf and qsvm.
  E5. For each target and model, each fold's confusion matrix in results.json equals the one recomputed from cells.json (cells with that fold, y_true vs pred).
  E6. For each target and model, each fold's balanced_accuracy and macro_f1 match the values recomputed from the per-cell predictions within 0.005.
  E7. Each kernel matrix is square with size equal to cell_ids length, and every id in cell_ids exists in cells.json.
  W1 (warning only). A kernel matrix is not symmetric within 1e-6, or its diagonal is not close to 1 for rbf or qsvm.
  W2 (warning only). manifest.preprocessing.fitted_within_training_folds_only is false.
  W3 (warning only). A fold has fewer than 3 positive cells.
- Show the report as a list of passed, warning and failed checks. Save it to validation_report.
- A run can only be set to status "complete" when there are zero errors. Otherwise it stays "draft".
- Store the files using the built-in file upload and save their URLs on the Run record.
- A "Load placeholder bundle" button that creates a small, obviously fake run (40 cells per target, is_placeholder true, run_id starting with "PLACEHOLDER-") by generating random values in the browser. This is the only place random data may ever be generated, and R6 applies to it.
- A list of existing runs with status, placeholder flag, a "feature this run" toggle (only one featured run at a time) and delete.

APP SHELL
- Pages: Overview, Compare Models, Quantum Lab, Pipeline, plus the admin-only Upload run page. For now each public page shows only its title and an empty state; later prompts fill them in.
- A data layer that loads the active run: the featured run, or else the most recent run with status "complete". It fetches the four JSON files from their URLs once and shares them with all pages. Show a loading state and a clear error state if a file cannot be fetched.
- Global controls in the header, kept in the URL query string so links are shareable:
  - Target: "Biological: Dendritic vs Rest" or "Synthetic: Planted Interaction vs Background" (default biological).
  - Baseline to compare against QSVM: Linear, Poly-2 or RBF (default RBF).
- Header title "Quantum vs Classical: Finding Rare Cells", subtitle "Reproducible single-cell SVM benchmarks on healthy-donor PBMC data."
- Status chips under the header, all read from the manifest: dataset name, "Biological" or "Synthetic control", run id, execution status, run timestamp.
- Footer on every page: "Research and education benchmark on healthy-donor data. Not a clinical or diagnostic tool."

DESIGN
- Dark scientific dashboard. Background near #0B1020, panels #131A2E, text #E5E7EB.
- Classical accent blue #3B82F6. Quantum accent purple #8B5CF6. Use them only to label which model is which, never to signal better or worse.
- Outcome colors, colorblind-safe, always paired with a marker shape and a text label: true positive #009E73 circle, false negative #D55E00 diamond, false positive #E69F00 square, true negative #6B7280 small circle.
- Class colors: rare/positive class #F0E442, other class #56B4E9.
- Responsive down to phone width. Visible keyboard focus. Body text contrast at least 4.5:1.
- Charts: use Plotly (react-plotly.js with plotly.js) for 3D scatter plots and heatmaps. If you cannot install it, load Plotly from its CDN. If that is also not possible, use three.js for 3D and canvas for heatmaps. Tell me which one you used.

When you are done, list the pages you created and confirm that no sample scores or cells exist anywhere except inside the "Load placeholder bundle" action.
```

**Done when**

- [ ] Admin page uploads the four JSON files and shows a pass/warn/fail report.
- [ ] "Load placeholder bundle" works and the placeholder banner appears on every page.
- [ ] Public pages show empty states with no numbers on them when no run exists.
- [ ] Logged-out visitors can open the app but cannot reach the upload page.

---

### Prompt 2 — Overview page

```text
Build the Overview page. Rules R1 to R7 still apply. Read everything from the active run; the Target and Baseline controls in the header drive this page.

1. Dataset strip (from run_manifest.json): dataset name with source link, cells and genes before QC, cells after QC, benchmark sample size, number and percentage of rare-class cells, how the cell-type labels were obtained (annotation.method) with annotation.limitations in a small "Label caveat" note. For the synthetic target, show synthetic_control.rule_description instead and a "Synthetic control" tag.

2. Model comparison cards, above the fold. Two large cards side by side: the selected baseline (blue label, for example "RBF SVM") and "QSVM" (purple label). Each card shows balanced accuracy, macro F1, rare-class recall, rare-class precision as mean ± std across outer folds, plus total runtime in seconds. Below them, a compact row with the two other baselines so all four models are always visible.

3. One generated sentence under the cards, built only from the numbers, using macro F1 (or the manifest's cv.selection_metric if it is balanced_accuracy):
   - If the absolute difference of means is smaller than the larger of the two standard deviations: "QSVM and [baseline] are not distinguishable at this sample size ([metric] [a] vs [b], difference within fold-to-fold spread)."
   - Otherwise: "[Higher model] measured higher [metric] than [lower model] on this run ([a] vs [b])."
   Add in small text: "One run, [n] cells, [k] outer folds. Not a general claim about quantum or classical methods."
   Never show a trophy, crown, "winner" badge or green/red better/worse coloring.

4. Side-by-side Kernel-PCA 3D plots: left uses cells[].kpca for the selected baseline's kernel (rbf or poly2; if the baseline is Linear, show rbf and say so), right uses cells[].kpca.qsvm. Same cells, same colors, same marker sizes, colored by true label. Rotating one plot rotates the other. Caption: "Kernel-PCA projection of each kernel's similarity structure, first three components. A projection for visual comparison, not evidence of separation. Scores are in the cards above." Also show kernels.json fit_scope under the plots.

5. QSVM attempts note: if results.targets[target].models.qsvm.attempts has more than one entry, show a small table of every attempt (reps, macro F1, balanced accuracy, runtime, which one is final) and the reason text from the manifest. Never hide a non-final attempt.

6. Links to the other three pages with one line each on what is there.
```

**Done when**

- [ ] Switching Target or Baseline updates cards, sentence and plots together.
- [ ] The sentence changes correctly when you load placeholder data several times.
- [ ] No "winner" styling anywhere.

---

### Prompt 3 — Compare Models page

```text
Build the Compare Models page. Rules R1 to R7 still apply.

1. Cell Explorer: two 3D scatter plots side by side, left for the selected baseline, right for QSVM. Both use exactly the same coordinates for the same cells.
   - Coordinates switch: UMAP 3D, PCA 3D (cells[].coords), or Kernel-PCA (cells[].kpca for that side's kernel). In UMAP and PCA modes both plots share identical coordinates; say that in the caption.
   - Color switch: "True label", "Predicted label", "Classification outcome" (true positive, false negative, false positive, true negative, using the outcome colors and shapes from the design rules).
   - Toggle chips to show or hide each of the four outcomes, with live counts on each chip.
   - Linked cameras, and a "2D view" switch for small screens.
   - Hover shows cell id, annotation, true label, that model's prediction and fold.
   - Clicking a cell selects it in both plots and opens the Cell Inspector.

2. Cell Inspector panel for the selected cell: id, annotation, true label, held-out fold, the prediction of all four models in a row with a match or mismatch mark, the SVM decision value per model only if cells[].score exists (label it "SVM decision value, uncalibrated", never "probability" or "confidence"), and the input features as a small bar chart using feature_names. If the cell is in kernels.json cell_ids, show its five most similar cells under the baseline kernel and under the quantum kernel with their similarity values and true labels. Caption: "Kernel similarity shows which cells the model treats as alike. It is not a causal or biological explanation."

3. Where the models disagree: a table of every cell where the baseline and QSVM predictions differ, with columns id, annotation, true label, baseline prediction, QSVM prediction, which one was right. Sortable, with counts at the top: "Baseline right, QSVM wrong: n", "QSVM right, baseline wrong: n", "Both wrong: n". Clicking a row selects that cell in the plots.

4. Confusion matrices: one per model (all four), as labeled 2x2 grids from pooled_confusion_matrix, rows are true labels and columns are predicted labels, with the class names from results.json. A toggle to show or hide.

5. Fold-level metrics: for each of the four metrics, a strip plot with one dot per outer fold per model plus the mean, so the fold-to-fold spread is visible. If ci95 exists, draw it as an interval. A toggle to show or hide.

6. Metrics table: all four models by balanced accuracy, macro F1, rare recall, rare precision (mean ± std) and total runtime. Below it, each model's tuned hyperparameters per fold from folds[].best_params.

7. A line at the top of the page stating the comparison is fair by construction, filled from the manifest: same cells, same [outer_folds] outer folds, [n_features] input features for every model, tuning by [selection_metric] on inner folds only.
```

**Done when**

- [ ] The same cell is highlighted in both plots when clicked.
- [ ] Outcome counts on the chips add up to the number of cells.
- [ ] Confusion matrix rows are true labels, columns are predictions.

**Publish now** so a live link exists. Keep building after.

---

### Prompt 4 — Quantum Lab page

```text
Build the Quantum Lab page. Rules R1 to R7 still apply.

1. "The quantum part, in plain words" panel at the top, written for a judge who is not a physicist. Use this text, filling the brackets from the manifest:
   "Each cell is described by [n_features] numbers. A classical kernel compares two cells with a formula. The quantum kernel compares them a different way: it loads each cell's numbers, shrunk by a tuned scale, into a [n_qubits]-qubit circuit as rotation angles, and measures how much the two resulting quantum states overlap. That overlap is the similarity score. A classical SVM then does the rest, exactly as it does for the other models. The circuit was run on the [quantum.sdk] simulator, not on quantum hardware. Whether this different notion of similarity helps is an empirical question, and the Compare Models page shows what we measured."

2. Circuit panel: show the uploaded circuit image (Run.circuit_image_url) with a settings table from manifest.quantum: feature map, qubits, repetitions for each attempt, entanglement pattern, SDK and version, backend. Add: "One qubit carries one input feature. If features are principal components, a qubit represents a component, not an individual gene." If no image was uploaded, show an empty state, not a drawn circuit.
   Below the table, when the manifest has them: show manifest.quantum.input_scale_note as a "Why the inputs are scaled" note, the candidate scales from input_scale_grid, the repetitions and scale chosen in each outer fold (results.targets[target].models.qsvm.folds[].best_params.reps and .scale) as a small table, the candidate repetitions from manifest.quantum.reps_grid when present, simulator_settings, and the smoke_test values labeled "Agreement with a reference simulation (1.0 is identical)". State that the scale was chosen on inner folds only, like gamma for RBF.

3. Kernel heatmaps: three heatmaps side by side for rbf, poly2 and qsvm from kernels.json, same cell order, same size, each with its own color scale and a visible color bar. Draw a thin divider and labels where the cell order changes from the other class to the rare class, using y_true from cells.json. Hovering shows the two cell ids, their true labels and the similarity value. Show fit_scope and each kernel's params under the plots. Caption: "Brighter means the kernel treats two cells as more alike. Block structure shows how each kernel organizes the classes; it does not by itself mean better classification."

4. Kernel comparison: a scatter plot of quantum kernel value vs RBF kernel value for every cell pair in the upper triangle (sample at most 5,000 pairs at random if there are more, and say so), colored by pair type: both rare, both other, mixed.

5. Synthetic control section, shown only when Target is Synthetic and tagged "Synthetic control":
   - The rule in words from manifest.synthetic_control.rule_description and features_used.
   - A 2D scatter of cells[].planted x1 vs x2 colored by true label.
   - If decision_regions exists, four small 2D panels (linear, poly2, rbf, qsvm) showing each model's predicted region as a shaded background from the exported grid with the true points on top. If it does not exist, show the scatter only and do not draw any boundary.
   - Caption: "This target was engineered by us to require a feature interaction. A result here says how a model handles that planted pattern, nothing about real cells."

6. A "Why quantum here?" note at the bottom: "A classical kernel could do this job, and on this run [the generated comparison sentence from the Overview page]. We used a quantum kernel to test, under identical conditions, whether its similarity measure changes rare-cell classification. That measured answer is the point of the project."
```

**Done when**

- [ ] All three heatmaps use the same cell order and the class divider lines up across them.
- [ ] The synthetic section is hidden on the biological target.
- [ ] No circuit is drawn unless an image was uploaded.

---

### Prompt 5 — Pipeline and provenance page

```text
Build the Pipeline page. Rules R1 to R7 still apply.

1. Lineage graph, left to right, drawn from manifest.pipeline_stages in order. Each stage is a node showing name, tool, a status pill (succeeded, failed, skipped, not run), duration and artifact name, with the link opening in a new tab when present. Stages with status "not_run" or "skipped" are drawn dimmed and say so in words. Never show a stage as succeeded unless the manifest says so. The expected stages are: raw data, Seqera / nf-core scrnaseq, Scanpy QC and preprocessing, split-aware features, classical SVM benchmarks, Qiskit feature map on Quantum Rings, evaluation, dashboard bundle.

2. Seqera panel: pipeline name, status and a "Open run on Seqera Platform" button from manifest.seqera.run_url, plus the uploaded Seqera screenshot if present. If neither exists, an empty state: "No Seqera run recorded for this bundle."

3. Reproducibility card from the manifest: random seeds, outer and inner fold counts, selection metric, sampling strategy with benchmark size and number of rare cells, feature pipeline, and a clear yes/no line: "Feature selection, scaling and PCA fitted inside training folds only". If no, show it as a warning.

4. Validation card: the saved validation_report for this run with every check and its result, and when it was checked.

5. Software versions table from manifest.software_versions.

6. Downloads: buttons to download run_manifest.json, results.json, cells.json and kernels.json exactly as uploaded, and a "Download PNG" control on every chart in the app.

7. Run switcher: a dropdown listing all runs with status "complete" so a visitor can view an earlier run. The choice goes into the URL query string.
```

**Done when**

- [ ] A stage marked `not_run` in the manifest is visibly dimmed.
- [ ] Downloaded JSON is byte-for-byte what was uploaded.

---

### Prompt 6 — Honesty and accessibility audit

```text
Audit the whole app against rules R1 to R7 and fix every violation. Report what you changed.

Check specifically:
1. Search all pages, components and code for hard-coded numbers that look like scores, counts, runtimes or coordinates. The only allowed source of generated data is the "Load placeholder bundle" action. Remove anything else.
2. Search all text for these words and remove or rewrite them: sick, diseased, cancer, tumor, patient, diagnosis, clinical, resistant, drug target, proves, breakthrough, quantum advantage, quantum supremacy, outperforms, superior, untangles, winner.
3. Every chart built from the synthetic target has the "Synthetic control" tag.
4. Every 3D or 2D projection has its projection caption.
5. With a placeholder run active, the banner shows on every page and every chart has the watermark, including in downloaded PNGs.
6. With no run at all, every page shows an empty state and no numbers.
7. Outcome categories are distinguishable without color (shape and label), every chart has a text summary for screen readers, all controls work by keyboard, and the layout works at 375px wide.
8. Every metric is shown with at most 3 decimal places and the same rounding everywhere.
```

---

### Prompt 7 (optional) — "Beat the line" interactive for the community vote

Base44 winners are chosen by community vote, and Quantum Leap asks for something a non-physicist understands in a minute. This adds a 30-second hands-on moment built only on real exported points.

```text
Add a page called "Try it" and link it from the Overview page. Rules R1 to R7 still apply. It uses only the synthetic target's cells and is tagged "Synthetic control".

1. Show the 2D scatter of cells[].planted x1 vs x2 colored by true label.
2. Challenge text: "Can you separate the two groups with one straight line? Drag the two handles."
3. The visitor drags two handles that define a straight line, and can flip which side counts as the planted class. Compute their balanced accuracy and macro F1 live in the browser from the actual points and show them as large numbers.
4. Next to their score, show the measured scores of the four models for the synthetic target from results.json, labeled "measured on held-out cells". Add a note: "Your line is scored on all points you can see. The models were scored on cells they had never seen, so this is a demonstration, not a fair contest."
5. After 20 seconds or when the visitor presses "Why is this hard?", reveal: "The planted class depends on two features together, so no single straight line can capture it. That is why linear models score lower here, and why we compare curved classical kernels with a quantum kernel." Then show the decision_regions panels if they exist.
6. A "Share my score" button that copies a link to this page. Do not store visitor scores.
```

---

### Prompt 8 — Publish prep

```text
Prepare the app for public demo. Rules R1 to R7 still apply.

1. Add an "About and methods" page: what the project asks, the two targets, how labels were derived (from the manifest), the fairness rules (same folds, same cells, same feature budget, inner-fold tuning only), what the quantum kernel is, the limits (one run, small sample, inferred labels, simulator not hardware, projections are not proof), and the tools used: Seqera Platform and nf-core, Scanpy, scikit-learn, Qiskit with the Quantum Rings SDK, Base44.
2. Make every public page work for visitors who are not logged in. Only the Upload run page requires an admin login.
3. Add a page title, description and social preview text: "Quantum vs Classical: Finding Rare Cells. A reproducible single-cell SVM benchmark."
4. The app opens on the Overview page with the featured run loaded. Target and Baseline choices in the URL are respected on first load.
5. Add a small "How to read this in one minute" guide on the Overview page: three numbered steps pointing at the cards, the disagreement table and the Quantum Lab explainer.
6. Check first-load performance with a 500-cell run: parse the JSON files once, and render heatmaps without freezing the page.
```

After this prompt: in the app's settings, set visibility to public (no login required), publish, and open the live link in a private browser window to confirm it loads logged out.

---

## 5. Fix-it prompts

Paste one when you hit the matching problem.

**Base44 added sample numbers or demo cells**

```text
You added data that did not come from an uploaded bundle. This breaks rule R1. Remove every hard-coded or generated score, cell, prediction and runtime outside the "Load placeholder bundle" action, and replace each with an empty state. List every file you changed.
```

**3D plots are blank or Plotly failed to load**

```text
The 3D plots do not render. Replace the charting approach for 3D scatter plots only: load Plotly from its CDN with a script tag inside a React effect and render once it is available, with a visible loading state. Keep every plot's data, colors, captions and linked selection exactly as they are.
```

**Uploaded JSON cannot be fetched from its file URL**

```text
Fetching the uploaded JSON files by URL fails in the published app. Change storage so the parsed content is saved in the database at upload time instead: keep run_manifest and results as object fields on Run, store cells in a new entity CellRecord (run_id, target, plus the cell fields, created in batches), and store each kernel matrix in a new entity KernelMatrix (run_id, target, kernel name, cell_ids, rows). Update the data layer to read from these. Keep the original files downloadable.
```

**The page freezes on a large run**

```text
The app freezes when a run has many cells. Parse each JSON file once and cache it, render heatmaps on canvas or with Plotly's WebGL heatmap, cap the kernel comparison scatter at 5,000 sampled pairs, and virtualize the disagreement table. Do not drop or subsample any cell from the scatter plots or the metrics.
```

**Wording drifted toward hype**

```text
Re-read rules R2, R3, R4 and R5 and rewrite any heading, caption, tooltip or button that makes a claim the uploaded numbers do not support. Use neutral, descriptive wording. List each string you changed, before and after.
```

---

## 6. Before the demo

- [ ] A real (non-placeholder) run is uploaded, passes validation with zero errors, and is the featured run.
- [ ] The placeholder run is deleted, or at least not featured.
- [ ] Live Base44 link opens logged out (Base44 track entry is the live link).
- [ ] Seqera run link or screenshot is on the Pipeline page (Code of Life requires one).
- [ ] The "plain words" panel and "Why quantum here?" note read correctly with the real numbers (Quantum Leap quality bar).
- [ ] No API key, promo code or access code appears in the app, the repo, or the demo video.
- [ ] Devpost: public GitHub repo with the Python pipeline, demo video of 3 minutes or less, every teammate added, tracks selected, AI tools listed (Base44 and any coding assistants).
- [ ] Everyone on the team can explain the bundle contract and one page of the app.

---

## 7. Where this file departs from the spec, and why

| Spec says | This file does | Reason |
|---|---|---|
| Streamlit dashboard, no React | Base44 app (which generates React) | You asked for Base44. Python cannot run there, so the dashboard reads exported JSON. |
| `kernel_*.npz`, `features.npz`, `processed.h5ad` feed the dashboard | `cells.json` and `kernels.json` feed the dashboard; the NPZ and H5AD files stay in the pipeline's artifact folder | A browser cannot read NPZ or H5AD. |
| Kernel-PCA is "the visual proof" that quantum space "untangles" rare cells (2.4) | Shown as a projection with a caption saying it is not evidence of separation | Section 1 of the same spec says a 3D projection cannot prove separation. The stricter rule wins. |
| Retry with `reps=3` if QSVM loses to RBF on the planted control (5) | `reps` (1, 2 or 3) is chosen on inner folds, with the input scale and C | Choosing `reps` after seeing held-out scores is tuning on the holdout, which section 5 also forbids. An earlier run that followed the retry rule is kept as `qcell-run-01` for comparison. |
| Prior reported numbers table (7) | Not in the app anywhere | They are unverified until their artifacts are loaded. |
| Matplotlib/seaborn heatmaps and circuit drawing | Heatmaps drawn in the app from `kernels.json`; circuit is an image exported from Qiskit | No Python in Base44. |
