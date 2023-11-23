"""
Interface for Trader Company Method.
"""
from logging import Logger

import numpy as np
from tqdm import tqdm

from tc_method.core import (
    Sampler,
    create_predictions,
    educate_traders,
    evaluate_predictions,
    generate_group_factors,
    init_traders,
    replace_traders,
)
from tc_method.sampler import NumpyRandomGenerator


class TraderCompanyModel:
    """
    Trader Company Method Model.
    """

    def __init__(
        self,
        stocks: int,
        duration: int,
        max_lags: int,
        max_feats: int,
        traders: int,
        cut_off: float,
    ):
        self.stocks = stocks
        self.duration = duration
        self.max_lags = max_lags
        self.max_feats = max_feats
        self.traders = traders
        self.cut_off = cut_off
        self.core = {
            "int_params": None,
            "str_params": None,
            "feats": None,
            "weights": None,
        }

    def train(
        self,
        train_x: np.array,
        train_y: np.array,
        sampler: Sampler,
        logger: Logger = None,
    ):
        if sampler is None:
            sampler = NumpyRandomGenerator()
        sampler.init()

        logger.info("Training Trader Company Model...")
        logger.info("Train X shape: %s ", str(train_x.shape))
        logger.info("Train Y shape: %s ", str(train_y.shape))

        interval = self.max_lags + self.duration

        _feat_returns = train_x[:interval]
        _tgt_returns = train_y[: self.duration]

        int_params, str_params, feats, weights = init_traders(
            sampler,
            self.stocks,
            self.max_lags,
            self.max_feats,
            self.traders,
            _feat_returns,
            _tgt_returns,
        )
        logger.info("Initial traders created. Looping...")
        _loops = range(train_x.shape[0] - interval + 1)
        if logger is not None:
            _loops = tqdm(_loops)
        for i in _loops:
            _feat_span = i + interval
            _tgt_span = i + self.duration
            _feat_returns = train_x[i:_feat_span]
            _tgt_returns = train_y[i:_tgt_span]

            _group_feats = generate_group_factors(
                _feat_returns,
                self.max_lags,
                self.max_feats,
                int_params,
                str_params,
                feats,
            )
            _preds1 = create_predictions(_group_feats, weights)
            _scores1 = evaluate_predictions(_preds1, _tgt_returns)
            weights = educate_traders(
                _scores1, _tgt_returns, _group_feats, weights, self.cut_off
            )

            _preds2 = create_predictions(_group_feats, weights)
            _scores2 = evaluate_predictions(_preds2, _tgt_returns)
            int_params, str_params, feats, weights = replace_traders(
                sampler,
                self.stocks,
                self.max_lags,
                self.max_feats,
                _feat_returns,
                _tgt_returns,
                _scores2,
                int_params,
                str_params,
                feats,
                weights,
                self.cut_off,
            )

        # Save the core parameters
        self.core["int_params"] = int_params
        self.core["str_params"] = str_params
        self.core["feats"] = feats
        self.core["weights"] = weights
        logger.info("Training completed.")

    def predict(self, test_x: np.array):
        _group_feats = generate_group_factors(
            test_x,
            self.max_lags,
            self.max_feats,
            self.core["int_params"],
            self.core["str_params"],
            self.core["feats"],
        )
        _preds = create_predictions(_group_feats, self.core["weights"])
        return _preds

    def get_latest_prediction(self, test_x: np.array):
        return self.predict(test_x)[:, -1].mean()
