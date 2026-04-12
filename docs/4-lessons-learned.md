# Lessons Learned

Four decisions that shaped the final pipeline — including two reversals from the initial design. Each jump is measured on the same 1,000-session offline validation set ([predictions/predictions_3.json](../predictions/predictions_3.json)).

## 1. Training distribution must match test distribution

**What I tried.** First model trained only on sessions with 5+ cart additions (the evaluation guarantee for test data). Seemed safe — these are the "clean" sessions with the most signal.

**Why it failed.** 5+ cart sessions average **53 interactions**; test sessions average **4**. The LambdaRank learned patterns that only exist in deep sessions: dense co-visitation graphs, strong family concentration, rich user trajectories. At inference on short test sessions, those features were mostly zero, and the model had never learned to rank well under that regime.

**What I changed.** Dropped the filter to **≥ 1 cart addition** and **truncated viewed products to the last 10** per training session ([train_model.py:499-590](../src/models/train_model.py#L499-L590)). This made training sessions distributionally identical to test.

**Result.** NDCG@5: **0.214 → 0.377** (+76% relative). Largest single lift in the project.

## 2. Excluding viewed products from candidates wrecked recall

**What I tried.** Filtered out products the user already viewed in the session from the candidate pool. Intuition: "the user saw it and didn't cart it — move on."

**Why it failed.** EDA revealed **24.2% of cart adds in training are products viewed earlier in the same session**. The "view → consider → cart" loop is the dominant purchase pattern, not the exception. By filtering viewed products, I was removing the single most predictive candidate class before the reranker even saw them.

**What I changed.** Kept all viewed products as candidates and added a `is_viewed_in_session` binary feature (#15 of 20) to let LambdaRank learn when re-viewing predicts carting vs. when it signals rejection.

**Result.** Hit Rate@5: **~45% → 76%**. `is_viewed_in_session` became a top-5 feature by importance.

## 3. Adaptive diversification beat hard per-family caps

**What I tried.** Hard cap of max 3 products per family in the top 5 — standard diversification to avoid showing the user "5 of the same thing."

**Why it failed.** For sessions where a user is clearly shopping within one family (e.g., a user viewing 8 dresses), the hard cap forced 2 slots to unrelated families with weaker scores. The displaced predictions were genuine leaders in the session-specific ranking.

**What I changed.** Adaptive cap in `diversified_top_k()` ([predict_model.py:629-668](../src/models/predict_model.py#L629-L668)): if the dominant family has ≥ 3 candidates scoring above the session median, raise its cap from 3 to 4. Median-based threshold ensures this only triggers when the dominant family legitimately has high-quality candidates, not just many low-scoring ones.

**Result.** NDCG@5: **+0.015** over hard cap. Not the biggest single lift, but removed a class of systematic errors on focused sessions.

## 4. LambdaRank over pointwise classification

**What I tried.** Pointwise LightGBM binary classifier (cart = 1, not-cart = 0) with `scale_pos_weight` to handle class imbalance. Simple, fast, and the go-to for tabular ML.

**Why it failed.** NDCG@5 is a ranking metric — it only cares about the relative order of products within a session, not absolute probabilities. A well-calibrated binary classifier can still produce the wrong ordering if calibration drifts across sessions with different density. Also: `scale_pos_weight` uniformly boosts all positives regardless of difficulty; popularity-weighted negatives (which LambdaRank handles via group-aware pair sampling) matter more here.

**What I changed.** Switched to `LGBMRanker` with `objective='lambdarank'`, `metric='ndcg'`, `eval_at=[5]`, and session-level groups ([train_model.py:593-622](../src/models/train_model.py#L593-L622)). Same features, same candidates, same training data — just a different loss.

**Result.** NDCG@5: **+0.05** over the tuned binary classifier. More importantly, it simplified hyperparameter search — direct NDCG optimization means validation score is the thing you're optimizing, not a proxy.

---

## What I'd do differently next time

- **Validate on a proper holdout from the start.** The current offline validation overlaps with the co-visitation matrix construction, so reported scores are optimistic. A strict time-based holdout (e.g., last 3 days of training as validation) would give honest numbers.
- **Train co-visitation matrices per-device.** Device 3 has a 9.0% cart rate vs Device 1's 5.7%; the co-visitation structure likely differs. One matrix per device might push NDCG@5 another 0.02-0.03.
- **Treat the 80.7% anonymous sessions as a separate regime.** Returning users get personalized signals; anonymous users only get in-session + content. Training two specialist rankers and routing at inference could beat one general model.
- **Replace CV cosine with learned product embeddings.** The 1280-dim CV embeddings are raw ResNet-style features. A small two-tower model trained on co-cart pairs would likely yield a better similarity space for ranking.
