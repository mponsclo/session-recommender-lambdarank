# Candidate Generation

The first stage of the two-stage pipeline. For each test session, produce ~50–150 candidate products from 8 diverse signals. Recall is the objective here — the reranker ([docs/3](3-ranking-lambdarank.md)) handles precision. All logic lives in [src/models/predict_model.py](../src/models/predict_model.py), function `generate_candidates()` at line 270.

## The 8 signals

| # | Signal | Type | Built by | Candidates / source |
|---|--------|------|----------|---------------------|
| 1 | View-to-cart co-visitation | Collaborative | `build_covisitation_matrix()` ([predict_model.py:37](../src/models/predict_model.py#L37)) | top 50 per viewed product |
| 2 | Item2Vec (Word2Vec on sessions) | Sequential | `train_item2vec()` ([predict_model.py:125](../src/models/predict_model.py#L125)) | top 15 neighbors per viewed product |
| 3 | Family top products | Metadata | inline ([predict_model.py:309-327](../src/models/predict_model.py#L309-L327)) | 15 per session family |
| 4 | CV embedding similarity | Content | `load_cv_embeddings()` ([predict_model.py:235](../src/models/predict_model.py#L235)) | top 30 by cosine similarity |
| 5 | Global popularity fallback | Popularity | inline ([predict_model.py:344-347](../src/models/predict_model.py#L344-L347)) | top 30 (max 3/family) |
| 6 | User history (returning users) | Personalized | `load_user_history()` ([predict_model.py:252](../src/models/predict_model.py#L252)) | all prior-carted products |
| 7 | Cart-to-cart co-visitation | Collaborative | `build_cart2cart_matrix()` ([predict_model.py:86](../src/models/predict_model.py#L86)) | top 30 per carted product |
| 8 | Section top products | Metadata | inline ([predict_model.py:363-373](../src/models/predict_model.py#L363-L373)) | 10 per session section |

## Why these eight

- **Co-visitation (1, 7)**: strongest collaborative signal on this data. EDA showed top 20 view-to-cart pairs are all within-family — exactly what co-visitation captures implicitly.
- **Item2Vec (2)**: sequential signal from 3.1M session sequences. Captures substitutability (products often viewed in the same session).
- **CV embeddings (4)**: 1280-dim pre-computed visual embeddings. Adds content similarity for the 3% of test products missing co-visitation data.
- **Family/section top (3, 8)**: hedges against sparse sessions (avg 3.5 products viewed). If co-visitation yields < 20 candidates, family/section fillers keep recall high.
- **User history (6)**: only 6.8% of sessions are returning users, but when they are, prior cart history is highly predictive.
- **Global popularity (5)**: fallback for cold-start sessions with almost no in-session signal. Capped at 3 per family to avoid over-concentration.

## Pool assembly and deduplication

All sources write into a single `defaultdict(lambda: {...})` keyed by `product_id` ([predict_model.py:276](../src/models/predict_model.py#L276)). Each candidate ends up with 8 score fields — one per source — defaulted to 0 for sources that didn't contribute. This is critical for the reranker: zero-valued features carry information ("this candidate wasn't recommended by co-visitation") rather than being missing.

Typical pool size per session: 50–150 unique products after dedup, pulled from ~43,692 total in the catalog. The reranker then scores all of them in one LightGBM inference pass.

## Including viewed products as candidates

Products the user already viewed in the session are **not filtered out**. EDA found that 24.2% of cart additions in training go to products viewed earlier in the same session. Excluding them was the single biggest regression in early experiments — including them jumped NDCG@5 from ~0.21 to 0.35+. See [Lessons Learned](4-lessons-learned.md) for the full story.

## Entry points

- [`generate_candidates()`](../src/models/predict_model.py#L270) — per-session candidate pool
- [`main()`](../src/models/predict_model.py#L692) — pipeline orchestration
- Session loop at [predict_model.py:743-748](../src/models/predict_model.py#L743-L748)
