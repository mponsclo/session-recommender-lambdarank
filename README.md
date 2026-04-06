# Inditex E-Commerce Recommender System

A recommender system built for the Inditex/Zara e-commerce platform, developed as part of a NUWE hackathon challenge. The system suggests 5 products per user session, handling cold-start scenarios where 93% of test sessions have no prior user history.

## Challenge Overview

The hackathon consisted of three tasks weighted by complexity:

| Task | Description | Points |
|------|-------------|--------|
| Task 1 | SQL/analytical queries on user, product and interaction data | 100 |
| Task 2 | Session metrics function (`get_session_metrics`) | 100 |
| Task 3 | Product recommender system (evaluated by NDCG) | 900 |

## Approach

### Data Pipeline (dbt + DuckDB)
Built a full data transformation pipeline using dbt with DuckDB:
- **Staging**: Raw data ingestion and type casting
- **Intermediate**: Session aggregation, product statistics, user profiles
- **Marts**: Dimension and fact tables
- **Features**: Engineered features for the recommender model

### Key Insights from EDA
- 93.2% of test sessions are cold-start (no user history in training data)
- 80.7% are fully anonymous (no `user_id`)
- Co-view to cart behavior is strongly within the same product family
- Product `cart_addition_rate` is the strongest predictive signal
- Device type influences cart rate (device 3: 9.0% vs device 1: 5.7%)

### Recommender Strategy
1. **Session-based signals**: Products viewed in session infer product family preferences
2. **Ranking**: Score by cart addition rate, embedding similarity, family match, popularity
3. **Diversification**: Cross-family co-carts represent 43% of pairs
4. **Fallback**: Top products by absolute cart additions for minimal-signal sessions
5. **Embeddings**: Product CV embeddings for within-family similarity

## Project Structure

```
├── data/                       # Not included (see note below)
├── predictions/                # Task outputs
├── models/                     # Trained models (not included)
├── src/
│   ├── data/                   # Data processing and API extraction
│   │   └── session_metrics.py  # Task 2 implementation
│   ├── explore/                # EDA and Task 1 queries
│   └── models/                 # Model training and prediction
├── transform/                  # dbt project (DuckDB)
├── tests/                      # Unit tests for Task 2
└── docs/                       # EDA findings and analysis
```

## Tech Stack

- **Python 3.12** — pandas, scikit-learn, numpy, matplotlib, seaborn
- **dbt + DuckDB** — data transformation pipeline
- **Jupyter** — exploratory data analysis

## Running

```bash
# Activate environment
source .venv/bin/activate

# Run dbt pipeline (requires data in data/raw/)
cd transform && dbt run && dbt test

# Run Task 2 tests
python -m pytest tests/function_tests.py -v

# Generate predictions
python src/explore/queries.py        # Task 1
python src/models/predict_model.py   # Task 3
```

## Data Notice

The datasets used in this project were provided by Inditex through the NUWE hackathon platform and are not included in this repository. The data includes user interactions, product catalogs, and product image embeddings which are proprietary to Inditex.
