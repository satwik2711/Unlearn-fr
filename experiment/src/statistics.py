"""Author-clustered uncertainty summaries with fixed resampling seeds."""

import numpy as np


def clustered_bootstrap(values, authors, seed=42, n=5000):
    values = np.asarray(values)
    authors = np.asarray(authors)
    if values.shape[0] != authors.shape[0] or not values.size:
        raise ValueError("values and authors must be non-empty and equally sized")
    rng = np.random.default_rng(seed)
    unique_authors = np.unique(authors)
    means = []
    for _ in range(n):
        pick = rng.choice(unique_authors, len(unique_authors), replace=True)
        means.append(np.mean([np.mean(values[authors == x]) for x in pick]))
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "n_authors": len(unique_authors),
        "positive_authors": int(
            sum(np.mean(values[authors == author]) > 0 for author in unique_authors)
        ),
        "bootstrap_seed": seed,
        "bootstrap_samples": n,
    }
