# EDA Findings — Inditex Recommender System

## 1. Test Set Composition (7,349 sessions)

| Segment | Sessions | % | Available Signals |
|---------|----------|---|-------------------|
| Anonymous cold-start | 5,932 | **80.7%** | Only in-session products + product stats |
| Known cold-start (not in train) | 920 | 12.5% | In-session + RFM profile |
| Returning (in train) | 497 | 6.8% | In-session + RFM + full behavior history |

**93.2% of test sessions are cold-start.** The recommender must work primarily from in-session signals and product-level features.

---

## 2. Available Signals per Session

- **Every session** has `products_viewed` (avg 3.5 products), device type, country, and product context (avg cart rate of viewed products = 6.8%).
- **Only 497 sessions** (6.8%) have user history (preferred family, past cart behavior).
- User preferred family matches the current session family only **17.5% of the time** — past category preferences are weak predictors of the current session.

---

## 3. Product Insights

### Co-occurrence Patterns
- **Co-view to cart is strongly same-family**: The top 20 view-to-cart product pairs are ALL within the same family.
- Same-family pairs generate **57% more co-occurrence sessions** than cross-family pairs (8.67M vs 6.51M total co-sessions).

### Catalog & Popularity
- **Section 1** dominates: 73% of test sessions, 72% of cart additions in training data.
- **Section 3** is second: 24% of test sessions, higher avg cart rate (8.8% vs 4.6%).
- Products viewed in test are **well-known** in training: avg 3,047 interactions, median 1,545.
- Top 1% of products account for a disproportionate share of interactions (long-tail distribution).
- **1,697 products** (3.9% of catalog) have never been interacted with in training data.

### Discount Effect
- Discounted products have higher cart rate: **7.2% vs 5.4%** for non-discounted.
- Discounted products also receive more interactions on average (2,235 vs 1,070).
- Only 10% of test sessions include views of discounted products.

### Device Type
- **Device 3** has significantly higher cart rate: **9.0%** vs device 1 at 5.7%.
- Device 1 is dominant (90% of sessions), device 3 is second (9.5%).

---

## 4. Session Behavior

### Interaction Depth & Conversion
| Session depth | % of sessions | % with cart add |
|---------------|--------------|-----------------|
| 1 interaction | 31.7% | 6.2% |
| 2-3 | 18.5% | 13.4% |
| 4-5 | 10.8% | 19.4% |
| 6-10 | 14.8% | 27.6% |
| 11-20 | 12.1% | 40.5% |
| 20+ | 12.2% | 59.1% |

- **24.2%** of cart-adds go to products the user viewed earlier in the same session.
- High-cart sessions (>= 5 carts, similar to test users) average **37.9 products viewed** and **53 interactions** — much deeper than the 3.5 avg products in test sessions.

### Cart Rate by Popularity Bucket
| Product popularity | Avg cart rate |
|-------------------|---------------|
| < 100 interactions | 4.3% |
| 100-499 | 6.1% |
| 500-999 | **7.0%** |
| 1K-4.9K | 6.7% |
| 5K+ | 5.4% |

Mid-popularity products (500-999 interactions) have the highest conversion rate.

---

## 5. Family-Level Analysis

### Top Families by Cart Volume (Training Data)
| Family | Avg Cart Rate | Total Carts | Products |
|--------|---------------|-------------|----------|
| 53 | 3.3% | 421K | 2,495 |
| 73 | 4.9% | 341K | 4,584 |
| 156 | 7.1% | 320K | 3,404 |
| 51 | 4.5% | 274K | 3,321 |
| 153 | 6.8% | 164K | 1,414 |
| 99 | **11.0%** | 74K | 802 |
| 126 | **13.5%** | 46K | 387 |

Families 99 and 126 have very high cart rates — products from these families are strong recommendation candidates when the session context points to them.

### Top 20 Products by Cart Additions
All top 20 products belong to **section 1** and families 53, 73, 156, 51, 153. None are discounted. These are the baseline "safe bets" for any recommendation.

---

## 6. Cold-Start Considerations

### Product Overlap
- **97%** of test products (8,983 / 9,253) also appear in training data.
- Only 270 test products are unseen — product-level features from training are reliable.

### User Overlap
- 380K known users in training, but only 497 test sessions come from returning users.
- For the 920 known cold-start users (in users.csv but not in train), we have RFM data but no interaction history.

### Anonymous Users (80.7% of test)
- No user_id at all — only session-level and product-level signals available.
- Must rely on: what products they are viewing now, product popularity, family/section affinity, device type, and country.

---

## 7. NDCG Scoring Implications

- **Every test session has >= 5 products added to cart** (per README), so there are always relevant items to find.
- NDCG rewards **placing relevant items at the top of the ranked list** — position 1 matters more than position 5.
- With only 5 recommendation slots, precision at the top is critical.
- The test sessions have much fewer interactions (avg 4.0) than the high-cart training sessions (avg 53.3), so the recommender receives limited signal and must make the most of it.

---

## 8. Strategy Implications for the Recommender

### Primary Signals (ranked by reliability)
1. **Products viewed in the current session** — infer family/section preference, find similar high-cart products.
2. **Product cart_addition_rate** — the strongest product-level predictor. Varies from 0% to 100% across products.
3. **Family-level affinity** — if a user views products from family 156, recommend top products from family 156.
4. **Product popularity within family** — `family_popularity_rank` from dim_products provides within-family ranking.
5. **Product embeddings** — CV embeddings enable content-based similarity within and across families.
6. **Device type** — device 3 users convert at 9% (vs 5.7%), adjust confidence accordingly.

### Weak Signals (use with caution)
- **User historical preferences** — available for only 6.8% of sessions, and match current session only 17.5% of the time.
- **RFM data** — available for 19.3% of sessions (returning + known cold-start), but recency/frequency/monetary don't directly predict which products to recommend.
- **Country** — some countries have higher avg cart ratios, but the signal is noisy.

### Recommended Architecture
1. **Candidate generation**: For each session, generate candidates from the same families as viewed products, plus globally popular products as fallback.
2. **Ranking**: Score candidates by a weighted combination of product cart rate, family match, embedding similarity to viewed products, and popularity.
3. **Diversification**: Ensure the 5 recommendations aren't all from the same family (cross-family co-carts represent 43% of pairs).
4. **Fallback**: For sessions with minimal signal (1 interaction, anonymous), fall back to the top 5 products by cart additions overall.
