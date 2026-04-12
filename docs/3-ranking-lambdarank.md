# LambdaRank Ranking

The second stage: score every candidate from [stage 1](2-candidate-generation.md) with a LightGBM LambdaRank model and select the top 5 with adaptive family diversification. Direct NDCG@5 optimization — the same metric used for evaluation. Training logic in [src/models/train_model.py](../src/models/train_model.py); inference in [src/models/predict_model.py](../src/models/predict_model.py).

## Why LambdaRank over pointwise classification

Early versions used binary classification (logistic regression / binary LightGBM) with cart=1 / not-cart=0 labels. LambdaRank beat it by ~0.05 NDCG@5 because:

- It optimizes a ranking loss directly; classification optimizes log-loss regardless of rank.
- Pairwise comparisons within a session match how NDCG@5 is actually computed.
- Popularity-weighted negatives matter more than positive class weighting — LambdaRank handles this via group-aware pair sampling.

## 20 features

Defined in `FEATURE_NAMES` ([train_model.py:485-492](../src/models/train_model.py#L485-L492)), computed in `compute_features()` ([train_model.py:456-478](../src/models/train_model.py#L456-L478)):

| # | Feature | Source |
|---|---------|--------|
| 0 | `covisit_score` (normalized) | view→cart co-visitation |
| 1 | `item2vec_score` | Item2Vec cosine similarity |
| 2 | `cart_addition_rate` | product's global cart rate |
| 3 | `family_match` | 1 if in session family, else 0 |
| 4 | `cv_embedding_sim` | 1280-dim CV cosine |
| 5 | `popularity` (log-normalized) | product interactions |
| 6 | `family_popularity_rank_inv` | inverse rank within family |
| 7 | `trend_bonus` | 3-day trend ratio |
| 8 | `has_discount` | binary |
| 9 | `cart_rate_vs_family_avg` | product rate − family average |
| 10 | `device_type` | from session (categorical int) |
| 11 | `is_returning_user` | binary |
| 12 | `user_history_match` | 1 if in user's past carts |
| 13 | `family_top_score` | family-top-products signal strength |
| 14 | `global_fallback_score` | global-popularity signal strength |
| 15 | `is_viewed_in_session` | binary — crucial, per EDA |
| 16 | `cart2cart_score` (normalized) | cart→cart co-visitation |
| 17 | `discount_view_ratio` | session-level |
| 18 | `session_depth` | # unique products viewed |
| 19 | `section_top_score` | section-top-products signal strength |

## Training setup

From `train_ranker()` at [train_model.py:593-622](../src/models/train_model.py#L593-L622):

```python
LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    n_estimators=300,
    num_leaves=31,
    learning_rate=0.05,
    min_child_samples=20,
    eval_at=[5],
    random_state=42,
    n_jobs=4,
)
```

### Training data construction

From `build_training_data()` at [train_model.py:499-590](../src/models/train_model.py#L499-L590):

- **Sessions used**: training sessions with **≥ 1 cart addition** (line 519). Earlier versions filtered to ≥ 5 carts to match the test guarantee, but this skewed the distribution — test sessions average 4 interactions vs 53 in 5+ cart sessions. See [Lessons Learned](4-lessons-learned.md).
- **Viewed-products truncation**: last 10 products viewed per session, matching test-time distribution.
- **Labels** (line 576): `1 if pid in carted else 0` — positives are products the user added to cart; negatives are all other candidates (viewed-but-not-carted + non-viewed recommendations from other signals).
- **Groups**: one group per session; `group_sizes` list of `len(candidate_pids)` per session is passed directly to `LGBMRanker.fit(group=...)`.

## Adaptive family diversification

Applied at inference time in `diversified_top_k()` ([predict_model.py:629-668](../src/models/predict_model.py#L629-L668)). Selects the top 5 with a per-family cap that flexes based on candidate quality:

1. **Baseline cap**: max 3 products per family.
2. **Adaptive bump** ([predict_model.py:632-645](../src/models/predict_model.py#L632-L645)): if the dominant family has ≥ 3 candidates scoring above the session median, raise its cap to 4. Rationale — when a family legitimately dominates (e.g., a user clearly shopping for one category), hard caps hurt more than they help.
3. **Selection loop** ([predict_model.py:647-657](../src/models/predict_model.py#L647-L657)): iterate candidates sorted by score; skip if family already at cap; stop at k=5.
4. **Fallback fill** ([predict_model.py:660-666](../src/models/predict_model.py#L660-L666)): if fewer than 5 selected after the cap-respecting pass, top up from the remaining pool to guarantee exactly 5.

## Results

Offline on 1,000 held-out training sessions with 5+ cart additions:

| Version | NDCG@5 | Hit Rate@5 |
|---------|--------|------------|
| Hardcoded baseline (top-5 popularity) | ~0.01 | ~5% |
| v1 (initial pipeline, binary classifier) | 0.214 | 45.5% |
| v2 (LambdaRank + adaptive diversification + viewed-products + retrained on 1+ cart sessions) | **0.377** | **76.0%** |

Top features by LightGBM importance (logged at [train_model.py:615-620](../src/models/train_model.py#L615-L620)): `covisit_score`, `popularity`, `item2vec_score`, `session_depth`, `cart_rate_vs_family_avg`, `cart2cart_score`.

> **Note**: validation uses training data that overlaps with the co-visitation matrix, so these scores are optimistic. Real test performance may differ.
