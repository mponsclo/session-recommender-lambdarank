"""
Task 3: Session-based product recommender system.

Two-stage pipeline:
1. Candidate generation (co-visitation, Item2Vec, family top products, CV embeddings, global fallback)
2. LightGBM LambdaRank reranker optimizing NDCG@5

Outputs predictions/predictions_3.json with 5 product recommendations per test session.
"""

import os
import json
import pickle
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
import duckdb
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import lightgbm as lgb

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

BASE_DIR = os.path.join(os.path.dirname(__file__), '../..')
DB_PATH = os.path.join(BASE_DIR, 'transform/target/inditex_recommender.duckdb')
EMBEDDINGS_PATH = os.path.join(BASE_DIR, 'data/raw/products.pkl')
OUTPUT_PATH = os.path.join(BASE_DIR, 'predictions/predictions_3.json')


# ---------------------------------------------------------------------------
# Step 1: Build co-visitation matrix
# ---------------------------------------------------------------------------

def build_covisitation_matrix(con, top_k=50):
    """Build view-to-cart co-visitation dict from training sessions."""
    log.info("Building co-visitation matrix...")
    df = con.sql(f"""
        WITH cart_events AS (
            SELECT session_id, product_id AS carted_product
            FROM marts.fct_interactions
            WHERE data_split = 'train' AND is_added_to_cart = 1
        ),
        view_events AS (
            SELECT session_id, product_id AS viewed_product
            FROM marts.fct_interactions
            WHERE data_split = 'train'
        ),
        covisit AS (
            SELECT
                v.viewed_product,
                c.carted_product,
                COUNT(DISTINCT v.session_id) AS co_sessions
            FROM view_events v
            JOIN cart_events c
                ON v.session_id = c.session_id
                AND v.viewed_product != c.carted_product
            GROUP BY v.viewed_product, c.carted_product
            HAVING COUNT(DISTINCT v.session_id) >= 2
        ),
        ranked AS (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY viewed_product
                    ORDER BY co_sessions DESC
                ) AS rn
            FROM covisit
        )
        SELECT viewed_product, carted_product, co_sessions
        FROM ranked
        WHERE rn <= {top_k}
    """).fetchdf()

    covisit_dict = {}
    for viewed, group in df.groupby('viewed_product'):
        covisit_dict[int(viewed)] = list(zip(
            group['carted_product'].astype(int),
            group['co_sessions'].astype(int)
        ))
    log.info(f"Co-visitation matrix: {len(covisit_dict)} source products")
    return covisit_dict


def build_cart2cart_matrix(con, top_k=30):
    """Build cart-to-cart co-visitation dict from training sessions."""
    log.info("Building cart-to-cart co-visitation matrix...")
    df = con.sql(f"""
        WITH cart_events AS (
            SELECT session_id, product_id
            FROM marts.fct_interactions
            WHERE data_split = 'train' AND is_added_to_cart = 1
        ),
        pairs AS (
            SELECT a.product_id AS product_a, b.product_id AS product_b,
                   COUNT(DISTINCT a.session_id) AS co_sessions
            FROM cart_events a
            JOIN cart_events b
                ON a.session_id = b.session_id AND a.product_id < b.product_id
            GROUP BY a.product_id, b.product_id
            HAVING co_sessions >= 2
        ),
        ranked AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY product_a ORDER BY co_sessions DESC) AS rn
            FROM pairs
        )
        SELECT product_a, product_b, co_sessions FROM ranked WHERE rn <= {top_k}
    """).fetchdf()

    # Store symmetrically
    cart2cart_dict = defaultdict(list)
    for _, row in df.iterrows():
        a, b, count = int(row['product_a']), int(row['product_b']), int(row['co_sessions'])
        cart2cart_dict[a].append((b, count))
        cart2cart_dict[b].append((a, count))
    log.info(f"Cart-to-cart matrix: {len(cart2cart_dict)} source products")
    return dict(cart2cart_dict)


# ---------------------------------------------------------------------------
# Step 2: Train Item2Vec (Word2Vec on product sequences)
# ---------------------------------------------------------------------------

def train_item2vec(con, vector_size=64, window=5, min_count=3, epochs=10):
    """Train Word2Vec on session product sequences from training data."""
    log.info("Training Item2Vec...")
    df = con.sql("""
        SELECT session_id, product_id, interaction_timestamp
        FROM marts.fct_interactions
        WHERE data_split = 'train'
        ORDER BY session_id, interaction_timestamp
    """).fetchdf()

    # Build sequences: list of product_id strings per session
    sentences = []
    for _, group in df.groupby('session_id'):
        products = group['product_id'].astype(str).tolist()
        # Deduplicate consecutive repeats
        deduped = [products[0]]
        for p in products[1:]:
            if p != deduped[-1]:
                deduped.append(p)
        if len(deduped) >= 2:
            sentences.append(deduped)

    log.info(f"Item2Vec: {len(sentences)} sessions with >= 2 unique products")
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=1,  # skip-gram
        workers=4,
        epochs=epochs,
        seed=42,
    )
    log.info(f"Item2Vec: vocabulary size = {len(model.wv)}")
    return model


# ---------------------------------------------------------------------------
# Step 3: Load data and build lookup structures
# ---------------------------------------------------------------------------

def load_test_sessions(con):
    """Load test sessions from feat_recommendation_input."""
    df = con.sql("""
        SELECT session_id, user_id, is_anonymous, is_returning_user, country_id,
               products_viewed, device_type, dominant_family_in_session,
               dominant_section_in_session, unique_products_in_session,
               session_interaction_count, avg_viewed_product_cart_rate,
               discount_view_ratio
        FROM features.feat_recommendation_input
    """).fetchdf()
    log.info(f"Test sessions: {len(df)}")
    return df


def load_product_catalog(con):
    """Load product catalog into a dict for O(1) lookup."""
    df = con.sql("""
        SELECT product_id, family_id, section_id, color_id, has_discount,
               total_cart_additions, cart_addition_rate, total_interactions,
               family_avg_cart_rate, family_popularity_rank, trend_ratio
        FROM marts.dim_products
    """).fetchdf()

    product_info = {}
    for _, row in df.iterrows():
        product_info[int(row['product_id'])] = {
            'family_id': int(row['family_id']) if pd.notna(row['family_id']) else -1,
            'section_id': int(row['section_id']) if pd.notna(row['section_id']) else -1,
            'color_id': int(row['color_id']) if pd.notna(row['color_id']) else -1,
            'has_discount': int(row['has_discount']),
            'total_cart_additions': float(row['total_cart_additions']),
            'cart_addition_rate': float(row['cart_addition_rate']),
            'total_interactions': int(row['total_interactions']),
            'family_avg_cart_rate': float(row['family_avg_cart_rate']) if pd.notna(row['family_avg_cart_rate']) else 0.0,
            'family_popularity_rank': int(row['family_popularity_rank']) if pd.notna(row['family_popularity_rank']) else 999,
            'trend_ratio': float(row['trend_ratio']) if pd.notna(row['trend_ratio']) else 0.0,
        }

    # Family -> products sorted by cart score
    family_products = defaultdict(list)
    for pid, info in product_info.items():
        score = info['total_cart_additions'] * (info['cart_addition_rate'] + 0.001)
        family_products[info['family_id']].append((pid, score))
    for fam in family_products:
        family_products[fam].sort(key=lambda x: -x[1])

    # Section -> products sorted by cart score
    section_products = defaultdict(list)
    for pid, info in product_info.items():
        score = info['total_cart_additions'] * (info['cart_addition_rate'] + 0.001)
        section_products[info['section_id']].append((pid, score))
    for sec in section_products:
        section_products[sec].sort(key=lambda x: -x[1])

    # Global top products (family-diverse: max 3 per family)
    all_products_sorted = sorted(product_info.items(), key=lambda x: -x[1]['total_cart_additions'])
    global_top = []
    fam_count = defaultdict(int)
    for pid, info in all_products_sorted:
        if fam_count[info['family_id']] < 3:
            global_top.append(pid)
            fam_count[info['family_id']] += 1
        if len(global_top) >= 30:
            break

    log.info(f"Product catalog: {len(product_info)} products, {len(family_products)} families, {len(section_products)} sections")
    return product_info, dict(family_products), global_top, dict(section_products)


def load_cv_embeddings():
    """Load CV embeddings from products.pkl."""
    log.info("Loading CV embeddings...")
    with open(EMBEDDINGS_PATH, 'rb') as f:
        df = pickle.load(f)
    valid = df[df['embedding'].notna()].copy()
    embeddings = np.stack(valid['embedding'].values).astype(np.float32)
    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms
    product_ids = valid['partnumber'].values.astype(int)
    pid_to_idx = {int(pid): i for i, pid in enumerate(product_ids)}
    log.info(f"CV embeddings: {len(pid_to_idx)} products, {embeddings.shape[1]} dims")
    return embeddings, pid_to_idx


def load_user_history(con):
    """Load returning user cart history."""
    df = con.sql("""
        SELECT user_id, product_id, add_to_cart_count
        FROM intermediate.int_user_product_interactions
        WHERE add_to_cart_count > 0
    """).fetchdf()
    user_history = defaultdict(list)
    for _, row in df.iterrows():
        user_history[int(row['user_id'])].append(int(row['product_id']))
    log.info(f"User history: {len(user_history)} users with cart history")
    return dict(user_history)


# ---------------------------------------------------------------------------
# Step 4: Candidate generation
# ---------------------------------------------------------------------------

def generate_candidates(products_viewed, covisit_dict, item2vec_model,
                        product_info, family_products, global_top,
                        cv_embeddings, cv_pid_to_idx, user_history_products,
                        cart2cart_dict=None, section_products=None):
    """Generate candidate products for a single session."""
    viewed_set = set(int(p) for p in products_viewed)
    candidates = defaultdict(lambda: {
        'covisit_score': 0.0,
        'item2vec_score': 0.0,
        'family_top_score': 0.0,
        'cv_sim_score': 0.0,
        'global_score': 0.0,
        'user_history_score': 0.0,
        'cart2cart_score': 0.0,
        'section_top_score': 0.0,
    })

    n_viewed = len(products_viewed)

    # Source 1: Co-visitation (allow viewed products as candidates)
    for pos, pid in enumerate(products_viewed):
        pid = int(pid)
        position_weight = 1.0 + 0.2 * (pos / max(n_viewed - 1, 1))
        if pid in covisit_dict:
            for carted_pid, co_count in covisit_dict[pid]:
                candidates[carted_pid]['covisit_score'] += co_count * position_weight

    # Source 2: Item2Vec neighbors (allow viewed products as candidates)
    for pid in products_viewed:
        pid_str = str(int(pid))
        if pid_str in item2vec_model.wv:
            try:
                neighbors = item2vec_model.wv.most_similar(pid_str, topn=15)
                for neighbor_str, sim in neighbors:
                    neighbor_pid = int(neighbor_str)
                    candidates[neighbor_pid]['item2vec_score'] += sim
            except KeyError:
                pass

    # Source 3: Family top products (allow viewed products as candidates)
    session_families = set()
    dominant_family = None
    family_count = defaultdict(int)
    for pid in products_viewed:
        pid = int(pid)
        if pid in product_info:
            fam = product_info[pid]['family_id']
            session_families.add(fam)
            family_count[fam] += 1
    if family_count:
        dominant_family = max(family_count, key=family_count.get)

    for fam in session_families:
        if fam in family_products:
            is_dominant = (fam == dominant_family)
            for rank, (fpid, fscore) in enumerate(family_products[fam][:15]):
                weight = 1.0 if is_dominant else 0.5
                candidates[fpid]['family_top_score'] += weight / (rank + 1)

    # Source 4: CV embedding similarity (allow viewed products as candidates)
    viewed_with_emb = [int(p) for p in products_viewed if int(p) in cv_pid_to_idx]
    if viewed_with_emb:
        viewed_indices = [cv_pid_to_idx[p] for p in viewed_with_emb]
        session_emb = cv_embeddings[viewed_indices].mean(axis=0, keepdims=True)
        sims = (cv_embeddings @ session_emb.T).flatten()
        top_indices = np.argpartition(sims, -30)[-30:]
        top_indices = top_indices[np.argsort(-sims[top_indices])]
        idx_to_pid = {v: k for k, v in cv_pid_to_idx.items()}
        for idx in top_indices:
            cpid = idx_to_pid[idx]
            candidates[cpid]['cv_sim_score'] = max(
                candidates[cpid]['cv_sim_score'], float(sims[idx])
            )

    # Source 5: Global fallback
    for rank, gpid in enumerate(global_top):
        if gpid not in viewed_set:
            candidates[gpid]['global_score'] = 1.0 / (rank + 1)

    # Source 6: User history
    if user_history_products:
        for hpid in user_history_products:
            if hpid not in viewed_set:
                candidates[hpid]['user_history_score'] = 1.0

    # Source 7: Cart-to-cart co-visitation
    if cart2cart_dict:
        for pid in products_viewed:
            pid = int(pid)
            if pid in cart2cart_dict:
                for co_pid, co_count in cart2cart_dict[pid]:
                    candidates[co_pid]['cart2cart_score'] += co_count

    # Source 8: Section top products
    if section_products:
        session_sections = set()
        for pid in products_viewed:
            pid = int(pid)
            if pid in product_info:
                session_sections.add(product_info[pid]['section_id'])
        for sec in session_sections:
            if sec in section_products:
                for rank, (spid, _) in enumerate(section_products[sec][:10]):
                    candidates[spid]['section_top_score'] += 1.0 / (rank + 1)

    return dict(candidates)


# ---------------------------------------------------------------------------
# Step 5: Feature engineering for reranker
# ---------------------------------------------------------------------------

def compute_features(candidates, products_viewed, product_info, session_row,
                     cv_embeddings, cv_pid_to_idx, user_history_products):
    """Compute feature vectors for all candidates in a session."""
    viewed_set = set(int(p) for p in products_viewed)
    session_families = set()
    dominant_family = None
    family_count = defaultdict(int)
    for pid in products_viewed:
        pid = int(pid)
        if pid in product_info:
            fam = product_info[pid]['family_id']
            session_families.add(fam)
            family_count[fam] += 1
    if family_count:
        dominant_family = max(family_count, key=family_count.get)

    # Normalize covisit scores within session
    max_covisit = max((c['covisit_score'] for c in candidates.values()), default=1.0)
    if max_covisit == 0:
        max_covisit = 1.0

    # Normalize cart2cart scores within session
    max_cart2cart = max((c.get('cart2cart_score', 0) for c in candidates.values()), default=1.0)
    if max_cart2cart == 0:
        max_cart2cart = 1.0

    # Max cart additions for log normalization
    max_cart_adds = max(
        (product_info.get(pid, {}).get('total_cart_additions', 0) for pid in candidates),
        default=1.0
    )
    if max_cart_adds == 0:
        max_cart_adds = 1.0

    user_history_set = set(user_history_products) if user_history_products else set()
    user_history_families = set()
    if user_history_products:
        for hpid in user_history_products:
            if hpid in product_info:
                user_history_families.add(product_info[hpid]['family_id'])

    features = []
    candidate_pids = []
    for pid, scores in candidates.items():
        info = product_info.get(pid, {})
        fam = info.get('family_id', -1)

        # Family match
        if fam == dominant_family:
            family_match = 1.0
        elif fam in session_families:
            family_match = 0.5
        else:
            family_match = 0.0

        # Popularity (log-normalized)
        popularity = np.log1p(info.get('total_cart_additions', 0)) / np.log1p(max_cart_adds)

        # User history match
        user_hist_match = 0.0
        if pid in user_history_set:
            user_hist_match = 1.0
        elif fam in user_history_families:
            user_hist_match = 0.3

        # Trend bonus
        trend = min(info.get('trend_ratio', 0.0) / 5.0, 1.0)

        # Cart rate relative to family
        cart_rate_vs_fam = 0.0
        fam_avg = info.get('family_avg_cart_rate', 0.0)
        if fam_avg > 0:
            cart_rate_vs_fam = info.get('cart_addition_rate', 0.0) / fam_avg

        feat = [
            scores['covisit_score'] / max_covisit,       # 0: covisit_score (normalized)
            scores['item2vec_score'],                      # 1: item2vec_score
            info.get('cart_addition_rate', 0.0),           # 2: cart_addition_rate
            family_match,                                  # 3: family_match
            scores['cv_sim_score'],                        # 4: cv_embedding_similarity
            popularity,                                    # 5: popularity (log-normalized)
            1.0 / info.get('family_popularity_rank', 999), # 6: family_popularity_rank (inverted)
            trend,                                         # 7: trend_bonus
            float(info.get('has_discount', 0)),            # 8: has_discount
            cart_rate_vs_fam,                               # 9: cart_rate_vs_family_avg
            float(session_row.get('device_type', 1)),      # 10: device_type
            float(session_row.get('is_returning_user', False)), # 11: is_returning_user
            user_hist_match,                               # 12: user_history_match
            scores['family_top_score'],                    # 13: family_top_score
            scores['global_score'],                        # 14: global_fallback_score
            # New features (improvements 1-6)
            1.0 if pid in viewed_set else 0.0,             # 15: is_viewed_in_session
            scores.get('cart2cart_score', 0.0) / max_cart2cart, # 16: cart2cart_score (normalized)
            float(session_row.get('discount_view_ratio', 0.0)), # 17: discount_view_ratio
            min(float(session_row.get('session_interaction_count', 1)) / 20.0, 1.0), # 18: session_depth
            scores.get('section_top_score', 0.0),          # 19: section_top_score
        ]
        features.append(feat)
        candidate_pids.append(pid)

    return np.array(features, dtype=np.float32), candidate_pids


FEATURE_NAMES = [
    'covisit_score', 'item2vec_score', 'cart_addition_rate', 'family_match',
    'cv_embedding_sim', 'popularity', 'family_popularity_rank_inv', 'trend_bonus',
    'has_discount', 'cart_rate_vs_family_avg', 'device_type', 'is_returning_user',
    'user_history_match', 'family_top_score', 'global_fallback_score',
    'is_viewed_in_session', 'cart2cart_score', 'discount_view_ratio',
    'session_depth', 'section_top_score',
]


# ---------------------------------------------------------------------------
# Step 6: Train LightGBM reranker
# ---------------------------------------------------------------------------

def build_training_data(con, covisit_dict, item2vec_model, product_info,
                        family_products, global_top, cv_embeddings, cv_pid_to_idx,
                        user_history, cart2cart_dict, section_products,
                        max_sessions=15000):
    """Build training data from training sessions with >= 1 cart addition."""
    log.info("Building training data for LightGBM reranker...")

    # Include sessions with >= 1 cart add (not just >= 5) to match test distribution
    df = con.sql(f"""
        SELECT f.session_id, f.user_id,
               mode(f.device_type_id) AS device_type,
               count(*) AS interaction_count,
               list(f.product_id) FILTER (WHERE f.is_added_to_cart = 0 OR f.is_added_to_cart IS NULL)
                   AS products_viewed,
               list(DISTINCT f.product_id) FILTER (WHERE f.is_added_to_cart = 1)
                   AS products_carted
        FROM marts.fct_interactions f
        WHERE f.data_split = 'train'
          AND f.session_id IN (
              SELECT session_id FROM intermediate.int_sessions
              WHERE products_added_to_cart >= 1
          )
        GROUP BY f.session_id, f.user_id
        HAVING len(products_viewed) >= 1 AND len(products_carted) >= 1
        ORDER BY random()
        LIMIT {max_sessions}
    """).fetchdf()

    log.info(f"Training sessions: {len(df)}")

    all_features = []
    all_labels = []
    group_sizes = []

    for _, row in df.iterrows():
        viewed_raw = row['products_viewed']
        carted_raw = row['products_carted']
        viewed = [int(p) for p in viewed_raw] if viewed_raw is not None and len(viewed_raw) > 0 else []
        carted = set(int(p) for p in carted_raw) if carted_raw is not None and len(carted_raw) > 0 else set()

        if not viewed:
            continue

        # Deduplicate viewed
        seen = set()
        viewed_dedup = []
        for p in viewed:
            if p not in seen:
                seen.add(p)
                viewed_dedup.append(p)
        # Truncate to last 10 products to simulate short test sessions
        viewed = viewed_dedup[-10:]

        user_id = row['user_id']
        user_hist = user_history.get(int(user_id), []) if pd.notna(user_id) else []

        candidates = generate_candidates(
            viewed, covisit_dict, item2vec_model,
            product_info, family_products, global_top,
            cv_embeddings, cv_pid_to_idx, user_hist,
            cart2cart_dict=cart2cart_dict, section_products=section_products
        )

        if not candidates:
            continue

        session_info = {
            'device_type': int(row['device_type']) if pd.notna(row['device_type']) else 1,
            'is_returning_user': bool(user_hist),
            'discount_view_ratio': 0.0,
            'session_interaction_count': int(row['interaction_count']) if pd.notna(row['interaction_count']) else 1,
        }
        features, candidate_pids = compute_features(
            candidates, viewed, product_info, session_info,
            cv_embeddings, cv_pid_to_idx, user_hist
        )

        labels = np.array([1 if pid in carted else 0 for pid in candidate_pids])

        # Only include sessions where at least 1 candidate is relevant
        if labels.sum() == 0:
            continue

        all_features.append(features)
        all_labels.append(labels)
        group_sizes.append(len(candidate_pids))

    X = np.vstack(all_features)
    y = np.concatenate(all_labels)
    log.info(f"Training data: {X.shape[0]} samples, {len(group_sizes)} groups, "
             f"{y.sum():.0f} positives ({y.mean()*100:.1f}%)")
    return X, y, group_sizes


def train_ranker(X, y, group_sizes):
    """Train LightGBM LambdaRank model."""
    log.info("Training LightGBM LambdaRank...")
    ranker = lgb.LGBMRanker(
        objective='lambdarank',
        metric='ndcg',
        n_estimators=300,
        num_leaves=31,
        learning_rate=0.05,
        min_child_samples=20,
        eval_at=[5],
        verbosity=-1,
        random_state=42,
        n_jobs=4,
    )
    ranker.fit(
        X, y,
        group=group_sizes,
        feature_name=FEATURE_NAMES,
    )
    log.info("LightGBM training complete")

    # Log feature importance
    importance = ranker.feature_importances_
    sorted_idx = np.argsort(-importance)
    log.info("Feature importance (top 10):")
    for i in sorted_idx[:10]:
        log.info(f"  {FEATURE_NAMES[i]}: {importance[i]}")

    return ranker


# ---------------------------------------------------------------------------
# Step 7: Predict and diversify
# ---------------------------------------------------------------------------

def diversified_top_k(candidate_pids, scores, product_info, k=5, max_per_family=3):
    """Select top-k products with adaptive family diversity constraint."""
    sorted_indices = np.argsort(-scores)

    # Adaptive: allow up to 4 from the dominant high-scoring family
    adaptive_max = {}
    if len(scores) > 0:
        median_score = np.median(scores)
        fam_counts_above_median = defaultdict(int)
        for idx in sorted_indices:
            fam = product_info.get(candidate_pids[idx], {}).get('family_id', -1)
            if scores[idx] > median_score:
                fam_counts_above_median[fam] += 1
        if fam_counts_above_median:
            top_fam = max(fam_counts_above_median, key=fam_counts_above_median.get)
            if fam_counts_above_median[top_fam] >= 3:
                adaptive_max[top_fam] = 4

    selected = []
    family_counts = defaultdict(int)
    for idx in sorted_indices:
        pid = candidate_pids[idx]
        fam = product_info.get(pid, {}).get('family_id', -1)
        limit = adaptive_max.get(fam, max_per_family)
        if family_counts[fam] < limit:
            selected.append(pid)
            family_counts[fam] += 1
        if len(selected) == k:
            break

    # Fill remaining if needed
    if len(selected) < k:
        for idx in sorted_indices:
            pid = candidate_pids[idx]
            if pid not in selected:
                selected.append(pid)
            if len(selected) == k:
                break

    return selected


def weighted_fallback_score(candidates, product_info):
    """Simple weighted scoring when LightGBM isn't available."""
    scores = {}
    max_covisit = max((c['covisit_score'] for c in candidates.values()), default=1.0) or 1.0
    for pid, c in candidates.items():
        info = product_info.get(pid, {})
        scores[pid] = (
            0.35 * (c['covisit_score'] / max_covisit) +
            0.25 * info.get('cart_addition_rate', 0.0) +
            0.15 * c.get('family_top_score', 0.0) +
            0.10 * c.get('item2vec_score', 0.0) +
            0.10 * c.get('cv_sim_score', 0.0) +
            0.05 * c.get('global_score', 0.0)
        )
    return scores


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    log.info("=" * 60)
    log.info("Starting recommender pipeline")
    log.info("=" * 60)

    con = duckdb.connect(DB_PATH, read_only=True)

    # Step 1: Co-visitation matrices
    covisit_dict = build_covisitation_matrix(con)
    cart2cart_dict = build_cart2cart_matrix(con)

    # Step 2: Item2Vec
    item2vec_model = train_item2vec(con)

    # Step 3: Load data
    test_sessions = load_test_sessions(con)
    product_info, family_products, global_top, section_products = load_product_catalog(con)
    cv_embeddings, cv_pid_to_idx = load_cv_embeddings()
    user_history = load_user_history(con)

    # Step 6: Train LightGBM reranker
    X_train, y_train, group_sizes = build_training_data(
        con, covisit_dict, item2vec_model, product_info,
        family_products, global_top, cv_embeddings, cv_pid_to_idx,
        user_history, cart2cart_dict, section_products,
        max_sessions=15000
    )
    ranker = train_ranker(X_train, y_train, group_sizes)

    con.close()

    # Step 7: Generate predictions for all test sessions
    log.info("Generating predictions for all test sessions...")
    predictions = {}

    for idx, row in test_sessions.iterrows():
        session_id = str(int(row['session_id']))
        products_viewed = row['products_viewed']

        if products_viewed is None or len(products_viewed) == 0:
            predictions[session_id] = global_top[:5]
            continue

        products_viewed = [int(p) for p in products_viewed]

        user_id = row['user_id']
        user_hist = []
        if pd.notna(user_id) and int(user_id) in user_history:
            user_hist = user_history[int(user_id)]

        # Generate candidates
        candidates = generate_candidates(
            products_viewed, covisit_dict, item2vec_model,
            product_info, family_products, global_top,
            cv_embeddings, cv_pid_to_idx, user_hist,
            cart2cart_dict=cart2cart_dict, section_products=section_products
        )

        if not candidates:
            predictions[session_id] = global_top[:5]
            continue

        # Compute features and score with LightGBM
        session_info = {
            'device_type': row.get('device_type', 1),
            'is_returning_user': bool(row.get('is_returning_user', False)),
            'discount_view_ratio': float(row.get('discount_view_ratio', 0.0)) if pd.notna(row.get('discount_view_ratio')) else 0.0,
            'session_interaction_count': int(row.get('session_interaction_count', 1)) if pd.notna(row.get('session_interaction_count')) else 1,
        }
        features, candidate_pids = compute_features(
            candidates, products_viewed, product_info, session_info,
            cv_embeddings, cv_pid_to_idx, user_hist
        )

        scores = ranker.predict(features)

        # Diversified top-5
        top5 = diversified_top_k(candidate_pids, scores, product_info)

        # Ensure we have exactly 5
        if len(top5) < 5:
            for gpid in global_top:
                if gpid not in top5:
                    top5.append(gpid)
                if len(top5) == 5:
                    break

        predictions[session_id] = [int(p) for p in top5[:5]]

        if (idx + 1) % 1000 == 0:
            log.info(f"  Processed {idx + 1}/{len(test_sessions)} sessions")

    # Step 8: Save predictions
    log.info(f"Total predictions: {len(predictions)} sessions")
    assert len(predictions) == 7349, f"Expected 7349 sessions, got {len(predictions)}"
    for sid, prods in predictions.items():
        assert len(prods) == 5, f"Session {sid} has {len(prods)} products"
        assert len(set(prods)) == 5, f"Session {sid} has duplicate products"

    output = {"target": predictions}
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    log.info(f"Predictions saved to {OUTPUT_PATH}")
    log.info("Done!")


if __name__ == '__main__':
    main()
