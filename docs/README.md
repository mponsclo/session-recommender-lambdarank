# Documentation

Technical deep-dives on each layer of the recommender. Start with the project [README](../README.md) for overview and results.

| Guide | Description |
|-------|-------------|
| [1. Data Pipeline](1-data-pipeline.md) | dbt + DuckDB: staging → intermediate → marts → features |
| [2. Candidate Generation](2-candidate-generation.md) | 8 signal sources producing ~100 candidates per session |
| [3. LambdaRank Ranking](3-ranking-lambdarank.md) | LightGBM LambdaRank with 20 features, direct NDCG@5 optimization |
| [4. Lessons Learned](4-lessons-learned.md) | What failed, what changed, what we'd do differently |
| [EDA Findings](eda_findings.md) | Exploratory analysis: cold-start, family structure, co-visitation signal |

**Notebooks:** [exploratory analysis](../notebooks/eda.ipynb)
