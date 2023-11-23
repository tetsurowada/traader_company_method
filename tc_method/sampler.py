import numpy as np
import optuna

from tc_method.core import Sampler


class NumpyRandomGenerator(Sampler):
    """
    Numpy Random Generator.
    """

    def __init__(self, seed: int = 42, logger=None):
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.size = None
        self.logger = logger

    def init(self):
        if self.logger is not None:
            self.logger.info("Numpy Random Generator initialized.")

    def is_stateful(self) -> bool:
        """
        (Public)
        Return whether the sampler is stateful.
        """
        return False

    def update_recorder(self):
        raise NotImplementedError

    def set_size(self, size: int):
        self.size = size

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
        int_params, str_params, feats = self._generate_factor_params(
            stock_combs, max_lags, max_feats, actvs, bins
        )
        return int_params, str_params, feats

    def _generate_factor_params(
        self, stock_combs, max_lags, max_feats, actvs, bins
    ) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        (Private)
        Create a list of parameters for generating factors.
        """
        feats = self.rng.integers(1, max_feats + 1, size=self.size)
        _size = np.sum(feats)
        _actvs = self.rng.choice(actvs, size=_size, replace=True)
        _bins = self.rng.choice(bins, size=_size, replace=True)
        _idxs = self.rng.choice(stock_combs.shape[0], size=_size, replace=True)
        _stock_combs = stock_combs[_idxs]

        _d_ps = self.rng.integers(0, max_lags, size=_size)
        _d_qs = self.rng.integers(0, max_lags, size=_size)

        int_params = np.column_stack((_stock_combs, _d_ps, _d_qs))
        str_params = np.column_stack(
            (
                _actvs,
                _bins,
            )
        )

        return int_params, str_params, feats


_NUM_FACTOR_NAME = "num_factors"


# FIXME: Takes too long to run
class OptunaSampler(Sampler):
    def __init__(self, logger=None):
        self.study = None
        self.current_trials = None
        self.size = 0
        self.logger = logger

    def is_stateful(self):
        return True

    def init(self, verbose=False):
        """
        Init a study.
        """
        optuna.logging.disable_default_handler()
        if verbose:
            optuna.logging.enable_default_handler()

        self.study = optuna.create_study(direction="maximize")
        if self.logger is not None:
            self.logger.info("Optuna Sampler initialized.")

    def _check_set_study(self):
        if self.study is None:
            raise ValueError("Study is not set. Please set study first")

    def set_size(self, size: int):
        self.size = size

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
        self._check_set_study()
        _trials = self.study.get_trials()
        _trials = _trials[-500:]
        self.study = optuna.create_study(direction="maximize")
        self.study.add_trials(_trials)
        current_study = optuna.create_study(direction="maximize")
        current_study.add_trials(_trials)

        int_params, str_params, feats = self._generate_factor_params(
            current_study, stock_combs, max_lags, max_feats, actvs, bins
        )
        return int_params, str_params, feats

    def _generate_factor_params(
        self,
        study: optuna.study.Study,
        stock_combs: np.ndarray,
        max_lags: int,
        max_feats: int,
        actvs: list[str],
        bins: list[str],
    ):
        """
        (Private)
        Create a list of parameters for generating factors.
        """
        stock_comb_idxs = range(len(stock_combs))
        study.optimize(
            lambda trial: _tcm_sample(
                trial,
                max_lags,
                max_feats,
                actvs,
                bins,
                stock_comb_idxs,
            ),
            n_trials=self.size,
            n_jobs=-1,
        )
        n = self.size
        current_trials = study.get_trials()[-n:]
        int_params, str_params, feats = _extract_params(
            current_trials,
            stock_combs,
        )
        self.current_trials = current_trials
        return int_params, str_params, feats

    def update_recorder(
        self,
        num: int = 0,
        good_trader_idxs: np.ndarray = None,
    ):
        if good_trader_idxs is None:
            raise ValueError("No good_trader_idxs provided")

        _trials = self.study.get_trials()
        prev_trials = [i for i in _trials[-num:] if i.value > 0]
        current_trials = prev_trials + self.current_trials
        current_trials = self._choose_good_traders(
            good_trader_idxs,
            current_trials,
        )
        # Add Current Trials to Study
        self.study.add_trials(current_trials)

    def _choose_good_traders(
        self,
        good_trader_idxs: np.ndarray,
        trials: list[optuna.trial.FrozenTrial],
    ):
        good_trader_idxs = set(good_trader_idxs)
        for i, trial in enumerate(trials):
            if i in good_trader_idxs:
                trial.value = 1.0
            else:
                trial.value = 0.0
        return trials


def _tcm_sample(
    trial,
    max_lags: int,
    max_feats: int,
    actvs: list[str],
    bins: list[str],
    stock_comb_idxs: np.ndarray,
):
    num_factors = trial.suggest_int(_NUM_FACTOR_NAME, 1, max_feats)
    factors = []
    for i in range(num_factors):
        _actv = trial.suggest_categorical(f"actv_{i}", actvs)
        _bin = trial.suggest_categorical(f"bins_{i}", bins)
        _comb_idx = trial.suggest_categorical(f"comb_idx_{i}", stock_comb_idxs)

        _d_p = trial.suggest_int(f"d_p_{i}", 1, max_lags)
        _d_s = trial.suggest_int(f"d_s_{i}", 1, max_lags)
        factors.append(
            (
                _actv,
                _bin,
                _comb_idx,
                _d_p,
                _d_s,
            )
        )
    return _dummy_eval_func(num_factors, factors)


def _dummy_eval_func(num_factors, factors):
    """
    Dummy evaluation function.
    """
    _, _ = num_factors, factors
    return 0


def _extract_params(
    trials: list,
    stock_combs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray,]:
    feats = []
    params = []

    for trial in trials:
        feats.append(trial.params[_NUM_FACTOR_NAME])
        params.extend(
            [v for k, v in trial.params.items() if k != _NUM_FACTOR_NAME],
        )
    feats = np.array(feats)
    params = np.array(params).reshape(-1, 5)
    idx_params = params[:, 2].astype(int)
    int_params1 = stock_combs[idx_params]
    int_params2 = params[:, 3:].astype(int)
    str_params = params[:, :2]
    return (
        np.concatenate([int_params1, int_params2], axis=1),
        str_params,
        feats,
    )
