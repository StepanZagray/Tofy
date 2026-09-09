"""C14 fixed F64 ridge readout; no files, generators, or model calls.

The objective averages over rows and sums over the four output columns. Only
coefficients are penalized; the intercept is the training one-hot label mean.
"""
import os

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np

LAMBDA = 0.01
MIN_STD = 1e-6
QUANTILES = (0., 0.01, 0.1, 0.5, 0.9, 0.99, 1.)
OUTPUTS = 4
MAX_RESIDUAL = 1e-9


def _array(value, name, shape=None):
    array = np.asarray(value)
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name}: real numeric values required")
    array = np.asarray(array, dtype=np.float64)
    if (shape is not None and array.shape != shape) or not np.isfinite(array).all():
        raise ValueError(f"{name}: invalid shape or nonfinite value")
    return array


def _features(value):
    array = _array(value, "features")
    if array.ndim != 2 or min(array.shape) < 1:
        raise ValueError("features: nonempty [rows, dimensions] required")
    return array


def fit(X_train, y_train):
    """Fit the single registered ridge recipe, with general n/d for fixtures."""
    X = _features(X_train)
    n, d = X.shape
    labels = np.asarray(y_train)
    if (labels.shape != (n,) or labels.dtype.kind not in "iu"
            or np.any(labels < 0) or np.any(labels >= OUTPUTS)):
        raise ValueError("labels: integer [rows] in 0..3 required")
    Y = np.eye(OUTPUTS, dtype=np.float64)[labels.astype(np.intp)]
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            mean = X.mean(axis=0, dtype=np.float64)
            std = X.std(axis=0, ddof=0, dtype=np.float64)
            scale = np.maximum(std, MIN_STD)
            Z = (X - mean) / scale
            intercept = Y.mean(axis=0, dtype=np.float64)
            centered = Y - intercept
            normal = Z.T @ Z / n + LAMBDA * np.eye(d, dtype=np.float64)
            rhs = Z.T @ centered / n
            # No near-zero threshold: a genuinely zero RHS has exactly zero B.
            coefficients = (np.zeros((d, OUTPUTS), dtype=np.float64)
                            if not np.any(rhs) else np.linalg.solve(normal, rhs))
            residual = float(np.linalg.norm(normal @ coefficients - rhs)
                             / max(float(np.linalg.norm(rhs)), np.finfo(np.float64).tiny))
            error = Z @ coefficients + intercept - Y
            objective = float(np.sum(error * error) / n
                              + LAMBDA * np.sum(coefficients * coefficients))
            condition = float(np.linalg.cond(normal))
            quantiles = {"probabilities": np.asarray(QUANTILES, dtype=np.float64),
                         "std": np.quantile(std, QUANTILES, method="linear"),
                         "scale": np.quantile(scale, QUANTILES, method="linear")}
    except (FloatingPointError, np.linalg.LinAlgError) as error:
        raise ValueError("ridge computation failed numerically") from error
    for name, value in (("mean", mean), ("std", std), ("scale", scale),
                        ("coefficients", coefficients), ("intercept", intercept),
                        ("diagnostics", [residual, objective, condition])):
        _array(value, name)
    if residual > MAX_RESIDUAL or objective < 0 or condition < 1:
        raise ValueError("ridge normal equation or scalar diagnostic failed")
    if not np.any(rhs) and np.any(coefficients):
        raise ValueError("zero RHS must have an exactly zero solution")
    return dict(mean=mean, std=std, scale=scale, coefficients=coefficients, intercept=intercept,
                relative_normal_equation_residual=residual, objective=objective,
                condition_number=condition, clamped_dimensions=int(np.count_nonzero(std < MIN_STD)),
                quantiles=quantiles)


def predict(X, fitted):
    """Return uncalibrated F64 scores, using training preprocessing unchanged."""
    X = _features(X)
    d = X.shape[1]
    mean = _array(fitted["mean"], "mean", (d,))
    scale = _array(fitted["scale"], "scale", (d,))
    coefficients = _array(fitted["coefficients"], "coefficients", (d, OUTPUTS))
    intercept = _array(fitted["intercept"], "intercept", (OUTPUTS,))
    if np.any(scale < MIN_STD):
        raise ValueError("scale must respect the registered positive floor")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            scores = ((X - mean) / scale) @ coefficients + intercept
    except FloatingPointError as error:
        raise ValueError("nonfinite ridge prediction") from error
    return _array(scores, "scores", (len(X), OUTPUTS))
