# Inditex E-Commerce Recommender System

A session-based recommender system for Inditex/Zara e-commerce, built for a NUWE hackathon challenge. The system recommends 5 products per session, handling extreme cold-start: 93% of test sessions have no user history and 81% are fully anonymous.

## Challenge Overview

| Task | Description | Points | Status |
|------|-------------|--------|--------|
| Task 1 | SQL/analytical queries on user, product and interaction data | 100 | Done |
| Task 2 | Session metrics function (`get_session_metrics`) | 100 | Done (8/8 tests) |
| Task 3 | Product recommender system (evaluated by NDCG@5) | 900 | Done |

## Recommender System (Task 3)

### The Problem

Given a user's in-session browsing activity (avg 3.5 products viewed), predict which 5 products they will add to cart. Scored by NDCG@5 (Normalized Discounted Cumulative Gain).

Key constraints:
- **43,692 products** in the catalog
- **93.2% cold-start** sessions (no user history in training data)
- **80.7% fully anonymous** (no `user_id` at all)
- Only in-session signals + product-level features available for most users

### Architecture

Two-stage pipeline inspired by the [OTTO Kaggle competition](https://www.kaggle.com/competitions/otto-recommender-system) winning approaches and session-based recommendation research ([Ludewig & Jannach, 2018](https://arxiv.org/abs/1803.09587)):

```
                                                          ┌─────────────────┐
  Co-visitation matrix (view-to-cart) ──┐                 │                 │
  Cart-to-cart co-visitation ───────────┤                 │   LightGBM      │
  Item2Vec embeddings (Word2Vec) ───────┼─> Candidates ──>│   LambdaRank    │──> Diversify ──> Top 5
  CV embedding similarity (1280-dim) ───┤   (~50-100)     │   (NDCG@5)      │
  Family/section top products ──────────┤                 │   20 features   │
  Global popularity fallback ───────────┘                 └─────────────────┘
```

**Stage 1 — Candidate Generation**: For each test session, generate 50-100 candidate products from 8 sources, combining collaborative filtering signals (what products co-occur in sessions) with content-based signals (visual embeddings, product metadata).

**Stage 2 — Ranking**: Score all candidates using a LightGBM LambdaRank model trained directly on the NDCG objective with 20 features. Apply family-diversity constraints (adaptive max per family) and select the top 5.

### Signal Sources

| Source | Method | Signal Type |
|--------|--------|-------------|
| View-to-cart co-visitation | Products carted in sessions where source product was viewed | Collaborative |
| Cart-to-cart co-visitation | Products frequently carted together in the same session | Collaborative |
| Item2Vec (Word2Vec) | Skip-gram embeddings trained on 3.1M session sequences | Sequential |
| CV embeddings | Cosine similarity using 1280-dim product image embeddings | Content |
| Family top products | Highest-converting products within the session's product families | Metadata |
| Section top products | Highest-converting products within the session's sections | Metadata |
| User history | Previously carted products for returning users (6.8% of sessions) | Personalized |
| Global popularity | Top products by cart additions (family-diverse fallback) | Popularity |

### Key Design Decisions

**Allowing viewed products as candidates**: 24.2% of cart additions in training data go to products the user already viewed in the same session. Including these as candidates was the single biggest improvement.

**Training data matching test distribution**: Initially trained only on sessions with 5+ cart adds (avg 53 interactions). Switching to sessions with 1+ carts and truncating viewed products to the last 10 better matches test conditions (avg 4 interactions).

**LambdaRank over pointwise classification**: Directly optimizes NDCG rather than binary cart/no-cart prediction, which better aligns with the evaluation metric.

**Adaptive diversification**: Instead of a rigid cap of 3 products per family, the system allows up to 4 from a dominant family when it has many high-scoring candidates.

### Results

Offline validation on 1,000 held-out training sessions with 5+ cart additions:

| Metric | Baseline (hardcoded) | v1 (initial pipeline) | v2 (improved) |
|--------|---------------------|----------------------|---------------|
| NDCG@5 | ~0.01 | 0.214 | **0.377** |
| Hit Rate@5 | ~5% | 45.5% | **76.0%** |
| Sessions with hit | ~50 | 455 | **760** |

Top features by LightGBM importance: co-visitation score, popularity, Item2Vec similarity, session depth, cart-rate-vs-family-avg, cart-to-cart score.

> **Note**: Validation uses training data that overlaps with the co-visitation matrix construction, so these scores are optimistic. Real test performance may differ.

## Data Pipeline (dbt + DuckDB)

All data transformations are managed with dbt targeting DuckDB:

```
Raw CSVs ──> Staging (views) ──> Intermediate (tables) ──> Marts (tables) ──> Features (tables)
```

- **Staging**: Type casting, column renaming, data split tagging
- **Intermediate**: Session aggregation, product statistics, user profiles, user-product interactions
- **Marts**: `dim_products`, `dim_users`, `dim_sessions`, `fct_interactions`
- **Features**: `feat_recommendation_input` (final wide table), `feat_product_popularity`, `feat_user_behavior`, `feat_session_context`

## EDA Insights

Full analysis in [`docs/eda_findings.md`](docs/eda_findings.md). Key findings:

- **Product `cart_addition_rate`** is the strongest product-level predictor (varies 0-100%)
- **Co-view to cart is strongly same-family**: top 20 view-to-cart pairs are all within the same family
- **Cross-family co-carts represent 43%** of pairs — diversification matters
- **Device 3** has 9.0% cart rate vs device 1 at 5.7%
- **Mid-popularity products** (500-999 interactions) have the highest conversion rate (7.0%)
- **97%** of test products also appear in training data — product-level features are reliable

## Project Structure

```
├── data/raw/                   # Raw CSVs and product embeddings (not included)
├── predictions/                # Task outputs (predictions_1.json, predictions_3.json)
├── src/
│   ├── data/
│   │   └── session_metrics.py  # Task 2: get_session_metrics()
│   ├── explore/
│   │   └── queries.py          # Task 1: SQL queries
│   └── models/
│       └── predict_model.py    # Task 3: Recommender pipeline
├── transform/                  # dbt project (DuckDB)
│   └── models/                 # Staging, intermediate, marts, features
├── tests/                      # Unit tests for Task 2
├── docs/                       # EDA findings
└── notebooks/                  # Exploratory analysis
```

## Tech Stack

- **Python 3.12** — pandas, numpy, scikit-learn, scipy
- **LightGBM** — LambdaRank for NDCG-optimized ranking
- **Gensim** — Word2Vec for Item2Vec product embeddings
- **dbt + DuckDB** — data transformation pipeline
- **Jupyter** — exploratory data analysis

## Running

```bash
# Activate environment
source .venv/bin/activate

# Install dependencies
pip install lightgbm gensim

# Run dbt pipeline (requires data in data/raw/)
cd transform && dbt run && dbt test && cd ..

# Run Task 2 tests
python -m pytest tests/function_tests.py -v

# Generate predictions
python src/explore/queries.py        # Task 1
python src/models/predict_model.py   # Task 3 (~15 min)
```

## Data Notice

The datasets were provided by Inditex through the NUWE hackathon platform and are not included in this repository. The data includes user interactions (~46.5M rows), product catalogs (43,692 products), user RFM profiles (577K users), and product CV embeddings — all proprietary to Inditex.
