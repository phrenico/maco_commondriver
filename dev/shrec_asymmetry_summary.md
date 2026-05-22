# ShRec Metric Asymmetry

## The Problem
The `run_shrec_method` in `scripts/experiments/method_runner.py` currently evaluates datasets inconsistently due to limitations in the `shrec.models.RecurrenceManifold` API:
- **Logmaps & Tentmaps:** The evaluation logic explicitly operates on the **training set** (`X_train`), ignoring the test partition.
- **Lorenz:** The logic calls `.fit_predict(X_test)` solely on the **test set**.

Because `shrec` is based on recurrence manifolds and lacks a distinct out-of-sample `.predict()` method (it uses `.fit_predict()`), it evaluates a manifold built entirely in isolation. Evaluating it purely on test data deprives it of the historical context, while evaluating it on training data constitutes a fundamental testing violation (grading on seen data). This introduces a severe baseline skew against other methods that cleanly separate `fit(train)` and `predict(test)`.

## The Solution
To fairly resolve this while respecting chronological integrity and the limitations of the `shrec` API:
1. **Concatenate the Sets:** Combine `X_train` and `X_test` chronologically into a single, contiguous array (`X_comb = np.concatenate([X_train, X_test], axis=0)`).
2. **Global Fit:** Run `z_pred_comb = model.fit_predict(X_comb)` so the recurrence manifold encompasses the full temporal dynamics.
3. **Strict Slicing (Fair Grading):** Slice the resulting predictions (`z_pred = z_pred_comb[-len(X_test):]`) to extract only the predictions corresponding to the test partition.
4. **Compare:** Run the metric evaluation (`comp_ccorr`) matching `z_test` strictly against the sliced `z_pred`. 

This mathematical workaround provides ShRec with the required global topology while ensuring it is algorithmically penalised and strictly graded on its out-of-sample test behavior.