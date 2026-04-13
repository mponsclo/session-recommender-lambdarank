"""LightGBM LambdaRank training for the session recommender.

Builds the training matrix from `marts.fct_interactions` (sessions with >=1 cart
addition, viewed products truncated to the last 10 to match test-time session
length) and fits an `LGBMRanker` with `objective='lambdarank'` to optimize NDCG@5
directly. Invoked from `predict_model.main()`; inference lives in predict_model.
"""

import logging

import lightgbm as lgb
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


FEATURE_NAMES = [
    "covisit_score",
    "item2vec_score",
    "cart_addition_rate",
    "family_match",
    "cv_embedding_sim",
    "popularity",
    "family_popularity_rank_inv",
    "trend_bonus",
    "has_discount",
    "cart_rate_vs_family_avg",
    "device_type",
    "is_returning_user",
    "user_history_match",
    "family_top_score",
    "global_fallback_score",
    "is_viewed_in_session",
    "cart2cart_score",
    "discount_view_ratio",
    "session_depth",
    "section_top_score",
]


def build_training_data(
    con,
    covisit_dict,
    item2vec_model,
    product_info,
    family_products,
    global_top,
    cv_embeddings,
    cv_pid_to_idx,
    user_history,
    cart2cart_dict,
    section_products,
    max_sessions=15000,
):
    """Build training data from training sessions with >= 1 cart addition."""
    # Deferred import breaks the train_model <-> predict_model cycle:
    # predict_model.main() imports from this module, and we need its
    # candidate/feature helpers at call time (both modules are loaded by then).
    from predict_model import compute_features, generate_candidates

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
        viewed_raw = row["products_viewed"]
        carted_raw = row["products_carted"]
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

        user_id = row["user_id"]
        user_hist = user_history.get(int(user_id), []) if pd.notna(user_id) else []

        candidates = generate_candidates(
            viewed,
            covisit_dict,
            item2vec_model,
            product_info,
            family_products,
            global_top,
            cv_embeddings,
            cv_pid_to_idx,
            user_hist,
            cart2cart_dict=cart2cart_dict,
            section_products=section_products,
        )

        if not candidates:
            continue

        session_info = {
            "device_type": int(row["device_type"]) if pd.notna(row["device_type"]) else 1,
            "is_returning_user": bool(user_hist),
            "discount_view_ratio": 0.0,
            "session_interaction_count": int(row["interaction_count"]) if pd.notna(row["interaction_count"]) else 1,
        }
        features, candidate_pids = compute_features(
            candidates, viewed, product_info, session_info, cv_embeddings, cv_pid_to_idx, user_hist
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
    log.info(
        f"Training data: {X.shape[0]} samples, {len(group_sizes)} groups, "
        f"{y.sum():.0f} positives ({y.mean() * 100:.1f}%)"
    )
    return X, y, group_sizes


def train_ranker(X, y, group_sizes):
    """Train LightGBM LambdaRank model."""
    log.info("Training LightGBM LambdaRank...")
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
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
        X,
        y,
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
