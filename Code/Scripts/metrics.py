import numpy as np

def pinball_loss(y_true, y_pred, quantile):
    error = y_true - y_pred
    return np.mean(np.maximum(quantile * error, (quantile - 1) * error))

def crps_score(y_true, y_pred_dist):
    """Approximates CRPS by averaging Pinball Loss across 99 quantiles."""
    quantiles = np.linspace(0.01, 0.99, 99)
    losses = [pinball_loss(y_true, y_pred_dist[:, i], q) for i, q in enumerate(quantiles)]
    return np.mean(losses)