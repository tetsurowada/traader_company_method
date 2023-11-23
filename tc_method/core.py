"""Core logic for the trader company method v2."""
from abc import ABC, abstractmethod

import numpy as np
from numba import jit

DEFAULT_ACTVS = ["eq", "tanh", "sinh", "relu"]
DEFAULT_BINS = ["sum", "prod", "sub", "proj", "max", "min"]


class Sampler(ABC):
    """
    (Public)
    Abstract base class for samplers.
    """

    @abstractmethod
    def init(self):
        pass

    @abstractmethod
    def create_params(
        self,
        stock_combs: np.ndarray,
        max_lags: int,
        max_feats: int,
        actvs: list[str],
        bins: list[str],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        (Public)
        Create a tuple of parameters for generating factors.
        """
        return

    def is_stateful(self) -> bool:
        """
        (Public)
        Return whether the sampler is stateful.
        """
        return

    @abstractmethod
    def update_recorder(self):
        """
        (Public)
        Update the recorder.
        """
        pass


def _spawn_traders(
    sampler: Sampler,
    stocks: int,
    max_lags: int,
    max_feats: int,
    n: int,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    (Private)
    Spawn a number of traders.
    """
    stock_combs, actvs, bins = _prepare(stocks, **kwargs)
    sampler.set_size(n)
    int_params, str_params, _feats = sampler.create_params(
        stock_combs, max_lags, max_feats, actvs, bins
    )
    return int_params, str_params, _feats


def _prepare(stocks, actvs=None, bins=None, default=True):
    """
    (Private)
    Return util arrays for generating factors.
    """
    stock_combs = np.array(np.meshgrid(range(stocks), range(stocks)))
    stock_combs = stock_combs.T.reshape(-1, 2)
    stock_combs = stock_combs[stock_combs[:, 0] != stock_combs[:, 1]]
    if default:
        return stock_combs, DEFAULT_ACTVS, DEFAULT_BINS
    return stock_combs, actvs, bins


def _generate_factor_params(
    rng, stock_combs, max_lags, max_feats, actvs, bins, n
) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    (Private)
    Create a list of parameters for generating factors.
    """
    feats = rng.integers(1, max_feats + 1, size=n)
    _size = np.sum(feats)
    _actvs = rng.choice(actvs, size=_size, replace=True)
    _bins = rng.choice(bins, size=_size, replace=True)
    _idxs = rng.choice(stock_combs.shape[0], size=_size, replace=True)
    _stock_combs = stock_combs[_idxs]

    _d_ps = rng.integers(0, max_lags, size=_size)
    _d_qs = rng.integers(0, max_lags, size=_size)

    int_params = np.column_stack((_stock_combs, _d_ps, _d_qs))
    str_params = np.column_stack(
        (
            _actvs,
            _bins,
        )
    )

    return int_params, str_params, feats


# =============================================================================
@jit(nopython=True)
def generate_group_factors(
    stock_returns,
    max_lags,
    max_feats,
    int_params,
    str_params,
    feats,
):
    """
    (Public)
    Generate factors for a group of traders.
    """
    interval = stock_returns.shape[0] - max_lags
    trader_factors = np.zeros((feats.shape[0], interval, max_feats))
    cursor = 0
    for i, feat in enumerate(feats):
        new_cursor = cursor + feat
        trader_factors[i] = _generate_factors(
            stock_returns,
            max_lags,
            max_feats,
            cursor,
            new_cursor,
            int_params,
            str_params,
        )
        cursor = new_cursor
    return trader_factors


@jit(nopython=True)
def _generate_factors(
    stock_returns,
    max_lags,
    max_feats,
    idx_from,
    idx_to,
    int_params,
    str_params,
):
    """
    (Private)
    Generate factors for a trader.
    """
    interval = stock_returns.shape[0] - max_lags
    factors = np.zeros((interval, max_feats))
    cursor = 0
    for i in range(idx_from, idx_to):
        p, q, d_p, d_q = int_params[i]
        actv, bo = str_params[i]

        for t in range(interval):
            et = t + max_lags + 1
            factors[t, cursor] = _factorize(
                stock_returns[t:et], p, q, d_p, d_q, actv, bo
            )
        cursor += 1

    return factors


@jit(nopython=True)
def _factorize(arr, p, q, d_p, d_q, actv, bo):
    """
    (Private)
    Create a factor from a time series array.
    """
    x1, x2 = arr[-(d_p + 1), p], arr[-(d_q + 1), q]
    match bo:
        case "sum":
            result = x1 + x2
        case "prod":
            result = x1 * x2
        case "sub":
            result = x1 - x2
        case "proj":
            result = x1
        case "max":
            result = max(x1, x2)
        case "min":
            result = min(x1, x2)
        case _:
            raise ValueError("Unknown binary operator")

    match actv:
        case "eq":
            return result
        case "tanh":
            return np.tanh(result)
        case "sinh":
            return np.sinh(result)
        case "relu":
            return max(0, result)
        case _:
            raise ValueError("Unknown activation function")


def init_traders(
    sampler: Sampler,
    stocks: int,
    max_lags: int,
    max_feats: int,
    traders: int,
    feat_stock_returns: np.ndarray,
    tgt_stock_returns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    (Public)
    Initialize a group of traders.
    """
    int_params, str_params, feats = _spawn_traders(
        sampler, stocks, max_lags, max_feats, traders
    )
    group_feats = generate_group_factors(
        feat_stock_returns,
        max_lags,
        max_feats,
        int_params,
        str_params,
        feats,
    )
    weights = _create_weights(group_feats, tgt_stock_returns)
    return int_params, str_params, feats, weights


def educate_traders(
    scores,
    tgt_returns,
    group_factors,
    weights,
    q,
) -> np.ndarray:
    educated, _ = np.nonzero(scores < np.quantile(scores, q))
    educated_weights = _create_weights(group_factors[educated], tgt_returns)
    new_weights = weights.copy()
    new_weights[educated] = educated_weights
    return new_weights


def replace_traders(
    sampler: Sampler,
    stocks: int,
    max_lags: int,
    max_feats: int,
    feat_returns: np.ndarray,
    tgt_returns: np.ndarray,
    scores: np.ndarray,
    int_params: np.ndarray,
    str_params: np.ndarray,
    feats: np.ndarray,
    weights: np.ndarray,
    q: float,
):
    pruned, _ = np.nonzero(scores < np.quantile(scores, q))
    if sampler.is_stateful():
        survived, _ = np.nonzero(scores >= np.quantile(scores, q))
        sampler.update_recorder(weights.shape[0], survived)
    pruned_idxs = _find_param_idxs(feats, pruned)
    new_int_params = np.delete(int_params, pruned_idxs, axis=0)
    new_str_params = np.delete(str_params, pruned_idxs, axis=0)
    new_feats = np.delete(feats, pruned, axis=0)
    new_weights = np.delete(weights, pruned, axis=0)

    spawn_int_params, spawn_str_params, spawn_feats = _spawn_traders(
        sampler, stocks, max_lags, max_feats, pruned.shape[0]
    )
    spawn_group_feats = generate_group_factors(
        feat_returns,
        max_lags,
        max_feats,
        spawn_int_params,
        spawn_str_params,
        spawn_feats,
    )
    spawn_weights = _create_weights(spawn_group_feats, tgt_returns)

    new_int_params = np.concatenate((new_int_params, spawn_int_params), axis=0)
    new_str_params = np.concatenate((new_str_params, spawn_str_params), axis=0)
    new_feats = np.concatenate((new_feats, spawn_feats), axis=0)
    new_weights = np.concatenate((new_weights, spawn_weights), axis=0)

    return new_int_params, new_str_params, new_feats, new_weights


def _create_weights(group_factors, targets):
    """
    (Private)
    Create weights from group factors and targets.
    """
    result = np.zeros((group_factors.shape[0], group_factors.shape[2]))
    for i, factors in enumerate(group_factors):
        weights = np.linalg.lstsq(factors, targets, rcond=None)[0]
        result[i] = weights.T
    return result


def create_predictions(group_factors, weights):
    """
    Create predictions from group factors and weights.
    """
    reshaped = weights[:, np.newaxis, :]
    return np.sum(group_factors * reshaped, axis=2)


def evaluate_predictions(predicted, actual):
    """
    Evaluate predictions.
    """
    signs_match = np.equal(np.sign(predicted), np.sign(actual.T))
    evaluation = np.where(signs_match, np.abs(actual.T), -np.abs(actual.T))
    return np.sum(evaluation, axis=1, keepdims=True)


@jit(nopython=True)
def _find_param_idxs(feats, pruned):
    pruned_idxs = []
    cursor = 0
    for i, feat in enumerate(feats):
        new_cursor = cursor + feat
        if i in pruned:
            for idx in range(cursor, new_cursor):
                pruned_idxs.append(idx)
        cursor = new_cursor
    return np.array(pruned_idxs)
