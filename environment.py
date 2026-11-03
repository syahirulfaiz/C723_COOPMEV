"""COSC723 CoopMEV-Lite: frozen v2.09 CPMM arbitrage environment and PPO policy symbols.

Concise port of the PPR01 DRL pipeline: the three-swap CPMM simulator, its wallet/reserve/
gas/block-time (C1-C6) constraints, observation normalisation, action decoding, route-policy
binding, and the two-head residual PPO policy. Training, HPO, callbacks and their CLI are
intentionally omitted. The module also registers itself as ``train`` so the frozen
``best_model.zip`` (serialised against the legacy module name) can be unpickled by
``stable_baselines3`` without the original training module.
"""

from __future__ import annotations
import argparse
import copy
from collections import Counter
from dataclasses import dataclass
from functools import partial
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch as th
from torch import nn
from torch.distributions import Categorical, Normal
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
import gymnasium as gym
from gymnasium import spaces

th.set_num_threads(1)
HYBRID_PPO_AVAILABLE = True
GYM_API = "gymnasium"
class CategoricalContinuousActionSpace(spaces.Box):
    """Two-value SB3 carrier whose first coordinate samples categorically."""

    def __init__(self, path_count: int, amount_min: float, amount_max: float):
        self.path_count = int(path_count)
        if self.path_count <= 0:
            raise ValueError("CategoricalContinuousActionSpace requires at least one path.")
        super().__init__(
            low=np.asarray([0.0, float(amount_min)], dtype=np.float32),
            high=np.asarray([float(self.path_count - 1), float(amount_max)], dtype=np.float32),
            dtype=np.float32,
        )

    def set_path_count(self, path_count: int) -> None:
        self.path_count = int(path_count)
        if self.path_count <= 0:
            raise ValueError("CategoricalContinuousActionSpace requires at least one path.")
        self.high[0] = float(self.path_count - 1)

    def sample(self, mask: Any = None, probability: Any = None) -> np.ndarray:
        del mask, probability
        rng = getattr(self, "np_random", np.random)
        integer_sampler = getattr(rng, "integers", getattr(rng, "randint", None))
        if integer_sampler is None:
            raise RuntimeError("Action-space random generator cannot sample integers.")
        path_index = int(integer_sampler(0, self.path_count))
        amount = float(rng.uniform(float(self.low[1]), float(self.high[1])))
        return np.asarray([float(path_index), amount], dtype=self.dtype)

    def contains(self, value: Any) -> bool:
        array = np.asarray(value, dtype=self.dtype)
        if (
            array.shape != self.shape
            or not bool(np.isfinite(array).all())
            or (not bool((array >= self.low).all()))
            or (not bool((array <= self.high).all()))
        ):
            return False
        return bool(abs(float(array[0]) - round(float(array[0]))) <= 0.0001)


SEED_DEFAULT = 20
DATASET_DATE = "2026-06-27"
POOL_DATA_FILENAME = f"uniswap_v2_and_sushiswap_v2_pools_data.{DATASET_DATE}.json"
TOKEN_PRICE_FILENAME = f"token_prices_{DATASET_DATE}.json"
WALLET_FILE_GLOB = f"wallet_balance_{DATASET_DATE}.*.json"
TOKEN_PRICE_GENERATOR_FILENAME = "token_price_generator.py"
EXPECTED_TOKEN_COUNT = 154
EXPECTED_POOL_COUNT = 174
EXPECTED_FUNDED_TOKEN_COUNT = 5
REFERENCE_PATH_COUNT = 114
EXPECTED_PARTICIPATING_POOL_COUNT = 24
EXPECTED_UNISWAP_POOL_COUNT = 89
EXPECTED_SUSHISWAP_POOL_COUNT = 85
EXPECTED_WALLET_BUDGETS = tuple(range(1000, 10001, 1000))
DATASET_IDENTITY = "legacy_symbol_collapsed_2026_06_27"
FUNDED_TOKEN_SYMBOL_BY_ID = {
    "ethereum": "ETH",
    "dai": "DAI",
    "usd-coin": "USDC",
    "tether": "USDT",
    "wrapped-bitcoin": "WBTC",
}
ETH_SYMBOL = FUNDED_TOKEN_SYMBOL_BY_ID["ethereum"]
FEE_DEFAULT = 0.003
AMOUNT_MIN_DEFAULT = 0.0
AMOUNT_MAX_DEFAULT = 1.0
EPS = 1e-12
OBSERVATION_DIM = EXPECTED_FUNDED_TOKEN_COUNT + 2 * EXPECTED_POOL_COUNT + 2
FIXED_FEATURE_MODE = "residual_features"
CORE_PATH_FEATURES = (
    "funded_start_wallet",
    "leg1_reserve_in",
    "leg1_reserve_out",
    "leg2_reserve_in",
    "leg2_reserve_out",
    "leg3_reserve_in",
    "leg3_reserve_out",
    "leg1_fee_multiplier",
    "leg2_fee_multiplier",
    "leg3_fee_multiplier",
    "gas_cost",
    "remaining_wall_clock",
)
PATH_FEATURE_SPEC_VERSION = "residual-features-v1"
PROBE_PATH_FEATURES = CORE_PATH_FEATURES + ("fixed_1pct_after_gas_profit_score",)
PROBE_FRACTION = 0.01
PROBE_PROFIT_SCALE_USD = 10.0
RESIDUAL_PATH_FEATURES = PROBE_PATH_FEATURES + (
    "relative_liquidity_capacity_score",
    "myopic_optimal_fraction",
)
FSTAR_FEATURE_INDEX = RESIDUAL_PATH_FEATURES.index("myopic_optimal_fraction")
FIXED_AMOUNT_TREATMENT = "residual_fstar"
PPO_WALLET_CURRICULA = ("fixed10k", "cycle1k10k")
PATH_POLICY_SPEC_VERSION = "path-policy-spec-v2-simple"
REWARD_REFERENCE_USD_DEFAULT = 10000.0
WALLET_NORMALIZATION_SCALE_USD = 10000.0
NORMALIZATION_SCHEMA_VERSION = "bounded-wallet-gas-v2"
C1_REJECTION_REWARD = -0.1
PHYSICAL_REJECTION_REWARD = -0.1
C6_EXPIRY_REWARD = 0.0
C1_REWARD_MODE_FIXED = "fixed"
C1_REWARD_MODE_COUNTERFACTUAL = "selected_action_counterfactual_after_gas"
C1_COUNTERFACTUAL_REWARD_FORMULA = (
    "clip(counterfactual_net_profit_usd_if_submitted / 10000, -0.1, 0)"
)
POLICY_AMOUNT_INTERIOR_FRACTION = 1e-06
AMOUNT_LOG_STD_MIN = -5.0
AMOUNT_LOG_STD_MAX = 2.0
PPO_AMOUNT_MU_ABS_MAX = 8.0
PPO_ENTROPY_QUADRATURE_ORDER = 8
_PPO_ENTROPY_NODES, _PPO_ENTROPY_WEIGHTS = np.polynomial.hermite.hermgauss(
    PPO_ENTROPY_QUADRATURE_ORDER
)
PPO_ENTROPY_QUADRATURE_NODES = tuple((float(value) for value in _PPO_ENTROPY_NODES))
PPO_ENTROPY_QUADRATURE_WEIGHTS = tuple((float(value) for value in _PPO_ENTROPY_WEIGHTS))
PPO_INITIAL_AMOUNT_LOG_STD = 0.0
PROTOCOL_ID = "v2.09-legacy-symbol-jun27-174p-114r-c6-r3"
ACTION_MODE_AUTO = "auto"
ACTION_MODE_HYBRID_PPO = "hybrid_ppo"
ACTION_MODE_CONTINUOUS_RELAXED = "continuous_relaxed"
PPO_ACTION_HEAD_ARCHITECTURE = "simple_shared_path_mlp_categorical_path_logistic_normal_amount"
ACTION_SPACE_NOTES = {
    ACTION_MODE_HYBRID_PPO: "PPO uses separate action heads over a shared actor representation: one categorical head samples a structural path and one continuous head parameterizes a path-conditioned logistic-normal amount fraction. The learned parameter shapes do not depend on the number of paths. The environment stores [path_id, u] in a two-value Box carrier solely for Stable-Baselines3 rollout-buffer compatibility; path_id is sampled categorically, not continuously.",
    ACTION_MODE_CONTINUOUS_RELAXED: "SAC/TD3 diagnostic Box action [path_selector, amount_fraction]. path_selector is clipped to [0, 1) and mapped deterministically to a structural path; the amount fraction remains continuous.",
}
DEFAULT_AMOUNT_GRID_TEXT = "0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1.0"


def canonical_ppo_path_feature_mode(value):
    return "residual_features"


def canonical_ppo_amount_treatment(value):
    return "residual_fstar"


def ppo_path_feature_dimension(value):
    return 15


def ppo_path_feature_requires_economic_static_inputs(value):
    return True


def parse_amount_grid_text(
    text: str, amount_min: float = AMOUNT_MIN_DEFAULT, amount_max: float = AMOUNT_MAX_DEFAULT
) -> List[float]:
    raw_parts = re.split("[,\\s]+", str(text or ""))
    values: List[float] = []
    for part in raw_parts:
        if not part:
            continue
        try:
            value = float(part)
        except ValueError as exc:
            raise ValueError(
                f"Invalid amount-grid entry {part!r}; expected numeric fractions such as 0.01"
            ) from exc
        if value <= 0:
            continue
        if value < amount_min - EPS or value > amount_max + EPS:
            continue
        values.append(float(np.clip(value, amount_min, amount_max)))
    values.extend([float(amount_min), float(amount_max)])
    return sorted(set((round(v, 12) for v in values)))


def action_space_note_for_mode(action_mode: str) -> str:
    return ACTION_SPACE_NOTES.get(action_mode, "Unknown action-space mode")


def resolve_action_mode(args: argparse.Namespace, algo: Optional[str] = None) -> str:
    requested = str(getattr(args, "action_mode", ACTION_MODE_AUTO)).strip().lower()
    algo_name = str(algo or "").upper()
    if requested == ACTION_MODE_AUTO:
        return ACTION_MODE_HYBRID_PPO if algo_name == "PPO" else ACTION_MODE_CONTINUOUS_RELAXED
    if requested not in {ACTION_MODE_HYBRID_PPO, ACTION_MODE_CONTINUOUS_RELAXED}:
        raise ValueError(f"Unsupported --action-mode {requested!r}")
    if requested == ACTION_MODE_HYBRID_PPO and algo_name in {"SAC", "TD3"}:
        raise ValueError(
            f"{algo_name} cannot use --action-mode hybrid_ppo. Use --action-mode auto or continuous_relaxed."
        )
    if requested == ACTION_MODE_CONTINUOUS_RELAXED and algo_name == "PPO":
        raise ValueError("PPO must use --action-mode hybrid_ppo in the v2.09 protocol.")
    return requested


def canonical_c1_rejection_reward(value: Any) -> float:
    reward = float(value)
    if math.isclose(reward, 0.0, rel_tol=0.0, abs_tol=1e-12):
        return 0.0
    if math.isclose(reward, C1_REJECTION_REWARD, rel_tol=0.0, abs_tol=1e-12):
        return float(C1_REJECTION_REWARD)
    raise ValueError(
        f"C1 rejection reward must be exactly 0 or -0.1 for the predeclared Stage-1 A/B screen; received {value!r}."
    )


def resolved_c1_rejection_reward(args):
    return float(getattr(args, "c1_rejection_reward", 0.0))


def resolved_c1_counterfactual_reward(args):
    return False


def c1_reward_rules(
    fixed_c1_reward: float, counterfactual_enabled: bool, *, c6_key: str = "C6_expiry"
) -> Dict[str, Any]:
    return {
        "accepted": "after_gas_profit_usd / 10000",
        "C1_rejection_mode": (
            C1_REWARD_MODE_COUNTERFACTUAL if counterfactual_enabled else C1_REWARD_MODE_FIXED
        ),
        "C1_rejection": (
            C1_COUNTERFACTUAL_REWARD_FORMULA
            if counterfactual_enabled
            else canonical_c1_rejection_reward(fixed_c1_reward)
        ),
        "C2_C3_rejection": PHYSICAL_REJECTION_REWARD,
        c6_key: C6_EXPIRY_REWARD,
    }


def c1_execution_semantics_note(fixed_c1_reward: float, counterfactual_enabled: bool) -> str:
    if counterfactual_enabled:
        return "Selected-action C1 rejections receive clipped counterfactual after-gas learning reward as if submitted; this is not realized profit. C2/C3 rejections receive -0.1. All rejected actions record zero realized profit, change no economic state, and incur no gas charge."
    return f"Selected-action C1 rejections receive reward {canonical_c1_rejection_reward(fixed_c1_reward):g}; C2/C3 rejections receive -0.1. All rejected actions record zero realized profit, change no economic state, and incur no gas charge."


@dataclass(frozen=True)
class PathPolicySpec:
    """Serializable path-to-observation mapping for the simple shared path MLP."""

    schema_version: str
    observation_dim: int
    wallet_indices: Tuple[int, ...]
    reserve_indices: Tuple[Tuple[int, ...], ...]
    fee_multipliers: Tuple[Tuple[float, ...], ...]
    gas_index: int
    remaining_time_index: int
    path_labels: Tuple[str, ...]
    reserve_amount_references: Tuple[Tuple[float, ...], ...] = ()
    start_token_prices_usd: Tuple[float, ...] = ()
    wallet_value_reference_usd: float = WALLET_NORMALIZATION_SCALE_USD
    gas_cost_reference_usd: float = 1.0
    economic_feature_spec_version: str = PATH_FEATURE_SPEC_VERSION

    def __post_init__(self) -> None:
        n_paths = len(self.wallet_indices)
        if self.schema_version != PATH_POLICY_SPEC_VERSION:
            raise ValueError(
                f"Unsupported path-policy schema {self.schema_version!r}; expected {PATH_POLICY_SPEC_VERSION!r}."
            )
        if self.observation_dim <= 0 or n_paths <= 0:
            raise ValueError(
                "PathPolicySpec requires a positive observation dimension and at least one path."
            )
        same_length = (len(self.reserve_indices), len(self.fee_multipliers), len(self.path_labels))
        if any((length != n_paths for length in same_length)):
            raise ValueError("PathPolicySpec fields do not contain the same number of paths.")
        if any((len(row) != 6 for row in self.reserve_indices)):
            raise ValueError("Each path must map six direction-specific reserve features.")
        if any((len(row) != 3 for row in self.fee_multipliers)):
            raise ValueError("Each path must provide three CPMM fee multipliers.")
        if any(
            (
                not math.isfinite(value) or value <= 0.0
                for row in self.fee_multipliers
                for value in row
            )
        ):
            raise ValueError("Every path fee multiplier must be finite and positive.")
        economic_lengths = (len(self.reserve_amount_references), len(self.start_token_prices_usd))
        if any(economic_lengths) and any((length != n_paths for length in economic_lengths)):
            raise ValueError("Economic path metadata must be present for every structural path.")
        if self.reserve_amount_references:
            if any((len(row) != 6 for row in self.reserve_amount_references)):
                raise ValueError(
                    "Each path must provide six directed reserve normalization references."
                )
            if any(
                (
                    not math.isfinite(value) or value <= 0.0
                    for row in self.reserve_amount_references
                    for value in row
                )
            ):
                raise ValueError(
                    "Every directed reserve normalization reference must be finite and positive."
                )
            if any(
                (not math.isfinite(value) or value <= 0.0 for value in self.start_token_prices_usd)
            ):
                raise ValueError(
                    "Every funded starting-token USD price must be finite and positive."
                )
        for name, value in (
            ("wallet_value_reference_usd", self.wallet_value_reference_usd),
            ("gas_cost_reference_usd", self.gas_cost_reference_usd),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"PathPolicySpec {name} must be finite and positive.")
        if self.economic_feature_spec_version != PATH_FEATURE_SPEC_VERSION:
            raise ValueError(
                f"Unsupported economic feature specification {self.economic_feature_spec_version!r}."
            )
        flat_indices = (
            list(self.wallet_indices)
            + [index for row in self.reserve_indices for index in row]
            + [self.gas_index, self.remaining_time_index]
        )
        if min(flat_indices) < 0 or max(flat_indices) >= self.observation_dim:
            raise ValueError(
                "PathPolicySpec references an observation feature outside the declared dimension."
            )

    @property
    def path_count(self) -> int:
        return len(self.wallet_indices)

    @property
    def has_economic_static_inputs(self) -> bool:
        return (
            len(self.reserve_amount_references) == self.path_count
            and len(self.start_token_prices_usd) == self.path_count
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "observation_dim": int(self.observation_dim),
            "wallet_indices": list(self.wallet_indices),
            "reserve_indices": [list(row) for row in self.reserve_indices],
            "fee_multipliers": [list(row) for row in self.fee_multipliers],
            "gas_index": int(self.gas_index),
            "remaining_time_index": int(self.remaining_time_index),
            "path_labels": list(self.path_labels),
            "reserve_amount_references": [list(row) for row in self.reserve_amount_references],
            "start_token_prices_usd": list(self.start_token_prices_usd),
            "wallet_value_reference_usd": float(self.wallet_value_reference_usd),
            "gas_cost_reference_usd": float(self.gas_cost_reference_usd),
            "economic_feature_spec_version": self.economic_feature_spec_version,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PathPolicySpec":
        return cls(
            schema_version=str(value.get("schema_version", "")),
            observation_dim=int(value.get("observation_dim", 0)),
            wallet_indices=tuple((int(x) for x in value.get("wallet_indices", ()))),
            reserve_indices=tuple(
                (tuple((int(x) for x in row)) for row in value.get("reserve_indices", ()))
            ),
            fee_multipliers=tuple(
                (tuple((float(x) for x in row)) for row in value.get("fee_multipliers", ()))
            ),
            gas_index=int(value.get("gas_index", -1)),
            remaining_time_index=int(value.get("remaining_time_index", -1)),
            path_labels=tuple((str(x) for x in value.get("path_labels", ()))),
            reserve_amount_references=tuple(
                (
                    tuple((float(x) for x in row))
                    for row in value.get("reserve_amount_references", ())
                )
            ),
            start_token_prices_usd=tuple(
                (float(x) for x in value.get("start_token_prices_usd", ()))
            ),
            wallet_value_reference_usd=float(
                value.get("wallet_value_reference_usd", WALLET_NORMALIZATION_SCALE_USD)
            ),
            gas_cost_reference_usd=float(value.get("gas_cost_reference_usd", 1.0)),
            economic_feature_spec_version=str(
                value.get("economic_feature_spec_version", PATH_FEATURE_SPEC_VERSION)
            ),
        )


def _mlp(input_dim: int, hidden_dims: Sequence[int], activation_fn: Any) -> Any:
    layers: List[Any] = []
    previous = int(input_dim)
    for hidden in hidden_dims:
        layers.extend([nn.Linear(previous, int(hidden)), activation_fn()])
        previous = int(hidden)
    return nn.Sequential(*layers)


if HYBRID_PPO_AVAILABLE:

    class VariablePathLogisticNormalDistribution(Distribution):
        """Categorical path plus path-conditioned logistic-normal amount."""

        def __init__(self, amount_min: float, amount_max: float):
            super().__init__()
            self.set_bounds(amount_min, amount_max)
            self.path_distribution: Any = None
            self.amount_mu: Any = None
            self.amount_log_std: Any = None
            self.distribution: Any = None

        @staticmethod
        def _require_finite(name: str, value: Any) -> None:
            if not bool(th.isfinite(value).all()):
                raise FloatingPointError(f"PPO {name} contains NaN or infinity.")

        def set_bounds(self, amount_min: float, amount_max: float) -> None:
            self.amount_min = float(amount_min)
            self.amount_max = float(amount_max)
            full_span = self.amount_max - self.amount_min
            if full_span <= 0.0:
                raise ValueError("Continuous amount bounds must define a positive interval.")
            boundary = POLICY_AMOUNT_INTERIOR_FRACTION * full_span
            self.sampling_min = self.amount_min + boundary
            self.sampling_max = self.amount_max - boundary
            self.amount_span = self.sampling_max - self.sampling_min
            if self.amount_span <= 0.0:
                raise ValueError(
                    "Continuous amount bounds contain no representable interior values."
                )

        def proba_distribution_net(self, *_args: Any, **_kwargs: Any) -> Any:
            raise NotImplementedError("The variable-path policy builds its shared heads directly.")

        def proba_distribution(
            self, path_logits: Any, amount_mu: Any, amount_log_std: Any
        ) -> "VariablePathLogisticNormalDistribution":
            if (
                path_logits.ndim != 2
                or amount_mu.shape != path_logits.shape
                or amount_log_std.shape != path_logits.shape
            ):
                raise ValueError(
                    "Path logits, amount means, and amount log standard deviations must all have shape [batch, path_count]."
                )
            self._require_finite("path logits", path_logits)
            self._require_finite("amount mean", amount_mu)
            self._require_finite("amount log standard deviation", amount_log_std)
            self.path_distribution = Categorical(logits=path_logits)
            self.amount_mu = amount_mu
            self.amount_log_std = amount_log_std
            self.distribution = [self.path_distribution]
            return self

        def _selected_amount_distribution(self, path_index: Any) -> Any:
            n_paths = int(self.amount_mu.shape[1])
            index = path_index.long().clamp(0, n_paths - 1)
            mu = self.amount_mu.gather(1, index.unsqueeze(1)).squeeze(1)
            log_std = self.amount_log_std.gather(1, index.unsqueeze(1)).squeeze(1)
            return Normal(mu, th.exp(log_std))

        def _fraction_from_logit(self, logit: Any) -> Any:
            unit = th.sigmoid(logit)
            return self.sampling_min + self.amount_span * unit

        def sample(self) -> Any:
            path_index = self.path_distribution.sample()
            amount_logit = self._selected_amount_distribution(path_index).sample()
            amount = self._fraction_from_logit(amount_logit)
            return th.stack((path_index.to(dtype=amount.dtype), amount), dim=1)

        def mode(self) -> Any:
            path_index = th.argmax(self.path_distribution.logits, dim=1)
            selected = self._selected_amount_distribution(path_index)
            amount = self._fraction_from_logit(selected.mean)
            return th.stack((path_index.to(dtype=amount.dtype), amount), dim=1)

        def log_prob(self, actions: Any) -> Any:
            n_paths = int(self.amount_mu.shape[1])
            path_index = actions[:, 0].round().long().clamp(0, n_paths - 1)
            amount = actions[:, 1]
            unit = (amount - self.sampling_min) / self.amount_span
            machine_epsilon = th.finfo(unit.dtype).eps
            unit = unit.clamp(machine_epsilon, 1.0 - machine_epsilon)
            amount_logit = th.log(unit) - th.log1p(-unit)
            amount_distribution = self._selected_amount_distribution(path_index)
            log_jacobian = (
                math.log(self.amount_span)
                + th.nn.functional.logsigmoid(amount_logit)
                + th.nn.functional.logsigmoid(-amount_logit)
            )
            joint_log_probability = (
                self.path_distribution.log_prob(path_index)
                + amount_distribution.log_prob(amount_logit)
                - log_jacobian
            )
            self._require_finite("joint log probability", joint_log_probability)
            return joint_log_probability

        def entropy(self) -> Any:
            amount_std = th.exp(self.amount_log_std)
            nodes = th.as_tensor(
                PPO_ENTROPY_QUADRATURE_NODES,
                dtype=self.amount_mu.dtype,
                device=self.amount_mu.device,
            )
            weights = th.as_tensor(
                PPO_ENTROPY_QUADRATURE_WEIGHTS,
                dtype=self.amount_mu.dtype,
                device=self.amount_mu.device,
            )
            quadrature_logits = (
                self.amount_mu.unsqueeze(-1) + math.sqrt(2.0) * amount_std.unsqueeze(-1) * nodes
            )
            log_jacobian = (
                math.log(self.amount_span)
                + th.nn.functional.logsigmoid(quadrature_logits)
                + th.nn.functional.logsigmoid(-quadrature_logits)
            )
            expected_log_jacobian = (log_jacobian * weights).sum(dim=-1) / math.sqrt(math.pi)
            conditional_amount_entropy = (
                Normal(self.amount_mu, amount_std).entropy() + expected_log_jacobian
            )
            joint_entropy = self.path_distribution.entropy() + (
                self.path_distribution.probs * conditional_amount_entropy
            ).sum(dim=1)
            self._require_finite("joint entropy", joint_entropy)
            return joint_entropy

        def actions_from_params(
            self, path_logits: Any, amount_mu: Any, amount_log_std: Any, deterministic: bool = False
        ) -> Any:
            return self.proba_distribution(path_logits, amount_mu, amount_log_std).get_actions(
                deterministic=deterministic
            )

        def log_prob_from_params(
            self, path_logits: Any, amount_mu: Any, amount_log_std: Any
        ) -> Tuple[Any, Any]:
            actions = self.actions_from_params(path_logits, amount_mu, amount_log_std)
            return (actions, self.log_prob(actions))

    class VariablePathHybridPPOPolicy(ActorCriticPolicy):
        """Simple shared MLP for a categorical path and continuous amount."""

        def __init__(
            self,
            *args: Any,
            path_policy_spec: Mapping[str, Any],
            path_feature_mode: str = FIXED_FEATURE_MODE,
            amount_min: float = AMOUNT_MIN_DEFAULT,
            amount_max: float = AMOUNT_MAX_DEFAULT,
            amount_log_std_min: float = AMOUNT_LOG_STD_MIN,
            initial_amount_log_std: float = PPO_INITIAL_AMOUNT_LOG_STD,
            amount_treatment: str = FIXED_AMOUNT_TREATMENT,
            wallet_curriculum: str = "fixed10k",
            **kwargs: Any,
        ):
            self._pending_path_policy_spec = PathPolicySpec.from_mapping(path_policy_spec)
            self._path_policy_spec_serialized = self._pending_path_policy_spec.to_dict()
            self.path_feature_mode = canonical_ppo_path_feature_mode(path_feature_mode)
            self.path_feature_dim = ppo_path_feature_dimension(self.path_feature_mode)
            self.amount_treatment = canonical_ppo_amount_treatment(amount_treatment)
            self.wallet_curriculum = str(wallet_curriculum).strip().lower()
            if ppo_path_feature_requires_economic_static_inputs(self.path_feature_mode) and (
                not self._pending_path_policy_spec.has_economic_static_inputs
            ):
                raise ValueError(
                    f"{self.path_feature_mode} requires reserve-reference and funded-token price metadata for every structural path."
                )
            self.amount_min = float(amount_min)
            self.amount_max = float(amount_max)
            self.amount_log_std_min = float(amount_log_std_min)
            self.initial_amount_log_std = float(initial_amount_log_std)
            if (
                not math.isfinite(self.amount_log_std_min)
                or not self.amount_log_std_min < AMOUNT_LOG_STD_MAX
            ):
                raise ValueError("PPO amount log-standard-deviation floor is invalid.")
            if (
                not math.isfinite(self.initial_amount_log_std)
                or not self.amount_log_std_min < self.initial_amount_log_std < AMOUNT_LOG_STD_MAX
            ):
                raise ValueError("PPO initial amount log standard deviation is outside its bounds.")
            super().__init__(*args, **kwargs)

        @staticmethod
        def _hidden_dims(net_arch: Any) -> List[int]:
            if isinstance(net_arch, Mapping):
                values = net_arch.get("pi", net_arch.get("vf", ()))
            else:
                values = net_arch
            dims = [int(value) for value in values or ()]
            if not dims:
                dims = [128, 128]
            if any((value <= 0 for value in dims)):
                raise ValueError("PPO shared-path MLP dimensions must be positive.")
            return dims

        def _build(self, lr_schedule: Any) -> None:
            hidden_dims = self._hidden_dims(self.net_arch)
            latent_dim = hidden_dims[-1]
            self.path_encoder = _mlp(self.path_feature_dim, hidden_dims, self.activation_fn)
            self.path_action_head = nn.Linear(latent_dim, 1)
            self.amount_action_head = nn.Linear(latent_dim, 2)
            critic_width = int(hidden_dims[-1])
            self.value_head = nn.Sequential(
                nn.Linear(2 * latent_dim, critic_width),
                self.activation_fn(),
                nn.Linear(critic_width, critic_width),
                self.activation_fn(),
                nn.Linear(critic_width, 1),
            )
            self.action_dist = VariablePathLogisticNormalDistribution(
                self.amount_min, self.amount_max
            )
            self._install_path_policy_spec(self._pending_path_policy_spec, register=True)
            if self.ortho_init:
                self.path_encoder.apply(partial(self.init_weights, gain=np.sqrt(2)))
                self.value_head.apply(partial(self.init_weights, gain=np.sqrt(2)))
                self.path_action_head.apply(partial(self.init_weights, gain=0.01))
                self.amount_action_head.apply(partial(self.init_weights, gain=0.01))
                self.value_head[-1].apply(partial(self.init_weights, gain=1.0))
            initial_scale_position = (self.initial_amount_log_std - self.amount_log_std_min) / (
                AMOUNT_LOG_STD_MAX - self.amount_log_std_min
            )
            if not 0.0 < initial_scale_position < 1.0:
                raise ValueError("PPO initial amount log standard deviation is outside its bounds.")
            initial_scale_bias = math.log(initial_scale_position / (1.0 - initial_scale_position))
            with th.no_grad():
                self.amount_action_head.weight[0].zero_()
                self.amount_action_head.bias[0].zero_()
                self.amount_action_head.weight[1].zero_()
                self.amount_action_head.bias[1].fill_(initial_scale_bias)
            self.optimizer = self.optimizer_class(
                self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
            )

        def _install_path_policy_spec(self, spec: PathPolicySpec, register: bool) -> None:
            device = next(self.parameters()).device
            tensors = {
                "path_wallet_indices": th.as_tensor(
                    spec.wallet_indices, dtype=th.long, device=device
                ),
                "path_reserve_indices": th.as_tensor(
                    spec.reserve_indices, dtype=th.long, device=device
                ),
                "path_fee_multipliers": th.as_tensor(
                    spec.fee_multipliers, dtype=th.float32, device=device
                ),
                "path_reserve_amount_references": th.as_tensor(
                    spec.reserve_amount_references, dtype=th.float64, device=device
                ),
                "path_start_token_prices_usd": th.as_tensor(
                    spec.start_token_prices_usd, dtype=th.float64, device=device
                ),
                "path_wallet_value_reference_usd": th.as_tensor(
                    spec.wallet_value_reference_usd, dtype=th.float64, device=device
                ),
                "path_gas_cost_reference_usd": th.as_tensor(
                    spec.gas_cost_reference_usd, dtype=th.float64, device=device
                ),
                "path_gas_index": th.as_tensor(spec.gas_index, dtype=th.long, device=device),
                "path_remaining_time_index": th.as_tensor(
                    spec.remaining_time_index, dtype=th.long, device=device
                ),
            }
            if register:
                for name, tensor in tensors.items():
                    self.register_buffer(name, tensor, persistent=False)
            else:
                for name, tensor in tensors.items():
                    setattr(self, name, tensor)
            self.path_count = spec.path_count
            self._path_policy_spec_serialized = spec.to_dict()

        def configure_path_policy_spec(self, value: Mapping[str, Any]) -> None:
            spec = PathPolicySpec.from_mapping(value)
            if spec.observation_dim != self.features_dim:
                raise ValueError(
                    f"Path policy expects observation dimension {self.features_dim}, but the supplied catalogue maps dimension {spec.observation_dim}."
                )
            if ppo_path_feature_requires_economic_static_inputs(self.path_feature_mode) and (
                not spec.has_economic_static_inputs
            ):
                raise ValueError(
                    f"Cannot bind a {self.path_feature_mode} policy to a path catalogue without the required causal static inputs."
                )
            self._install_path_policy_spec(spec, register=False)
            set_path_count = getattr(self.action_space, "set_path_count", None)
            if callable(set_path_count):
                set_path_count(spec.path_count)
            elif (
                hasattr(self.action_space, "high") and np.asarray(self.action_space.high).size >= 1
            ):
                self.action_space.high[0] = float(spec.path_count - 1)

        def _path_features(self, observations: Any) -> Any:
            wallet = observations.index_select(1, self.path_wallet_indices)
            flat_reserve_indices = self.path_reserve_indices.reshape(-1)
            reserves = observations.index_select(1, flat_reserve_indices).reshape(
                observations.shape[0], self.path_count, 6
            )
            static_fees = self.path_fee_multipliers.unsqueeze(0).expand(
                observations.shape[0], -1, -1
            )
            gas = (
                observations.index_select(1, self.path_gas_index.view(1))
                .view(observations.shape[0], 1, 1)
                .expand(observations.shape[0], self.path_count, 1)
            )
            remaining_time = (
                observations.index_select(1, self.path_remaining_time_index.view(1))
                .view(observations.shape[0], 1, 1)
                .expand(observations.shape[0], self.path_count, 1)
            )
            base_features = th.cat(
                (wallet.unsqueeze(-1), reserves, static_fees, gas, remaining_time), dim=-1
            )
            if self.path_reserve_amount_references.shape != self.path_reserve_indices.shape:
                raise RuntimeError(
                    "Residual feature reserve-reference shape does not match the directed reserve-index shape."
                )
            work_reserves = reserves.to(dtype=th.float64)
            raw_reserves = work_reserves * self.path_reserve_amount_references.unsqueeze(0)
            tiny = th.finfo(th.float64).tiny
            normalized_upper = 1.0 - th.finfo(observations.dtype).eps
            wallet_bounded = th.clamp(wallet.to(dtype=th.float64), min=0.0, max=normalized_upper)
            wallet_usd = (
                self.path_wallet_value_reference_usd
                * wallet_bounded
                / th.clamp(1.0 - wallet_bounded, min=tiny)
            )
            start_prices = self.path_start_token_prices_usd.unsqueeze(0)
            wallet_amount = wallet_usd / th.clamp(start_prices, min=tiny)
            start_amount = PROBE_FRACTION * wallet_amount
            amount = start_amount
            mechanically_valid = start_amount > 0.0
            fee_multipliers = self.path_fee_multipliers.to(dtype=th.float64).unsqueeze(0)
            for leg in range(3):
                reserve_in = raw_reserves[..., 2 * leg]
                reserve_out = raw_reserves[..., 2 * leg + 1]
                mechanically_valid = mechanically_valid & (reserve_in > 0.0) & (reserve_out > 0.0)
                amount_after_fee = amount * fee_multipliers[..., leg]
                amount = (
                    amount_after_fee
                    * reserve_out
                    / th.clamp(reserve_in + amount_after_fee, min=tiny)
                )
            gas_bounded = th.clamp(gas[..., 0].to(dtype=th.float64), min=0.0, max=normalized_upper)
            gas_usd = (
                self.path_gas_cost_reference_usd
                * gas_bounded
                / th.clamp(1.0 - gas_bounded, min=tiny)
            )
            gross_profit_usd = (amount - start_amount) * start_prices
            net_profit_usd = th.where(mechanically_valid, gross_profit_usd - gas_usd, -gas_usd)
            fixed_profit_score = (
                th.tanh(net_profit_usd / PROBE_PROFIT_SCALE_USD)
                .unsqueeze(-1)
                .to(dtype=observations.dtype)
            )
            cycle_r = th.ones_like(wallet_amount, dtype=th.float64)
            cycle_d = th.zeros_like(wallet_amount, dtype=th.float64)
            qstar_valid = wallet_amount > 0.0
            for leg in range(3):
                reserve_in = raw_reserves[..., 2 * leg]
                reserve_out = raw_reserves[..., 2 * leg + 1]
                fee = fee_multipliers[..., leg]
                leg_valid = (
                    th.isfinite(reserve_in)
                    & th.isfinite(reserve_out)
                    & th.isfinite(fee)
                    & (reserve_in > 0.0)
                    & (reserve_out > 0.0)
                    & (fee > 0.0)
                )
                qstar_valid = qstar_valid & leg_valid
                safe_reserve_in = th.clamp(reserve_in, min=tiny)
                leg_m = fee * reserve_out / safe_reserve_in
                leg_h = fee / safe_reserve_in
                previous_r = cycle_r
                cycle_r = leg_m * previous_r
                cycle_d = cycle_d + leg_h * previous_r
            qstar_valid = (
                qstar_valid
                & th.isfinite(cycle_r)
                & th.isfinite(cycle_d)
                & (cycle_r > 1.0)
                & (cycle_d >= 0.0)
            )
            qstar_numerator = th.expm1(0.5 * th.log(th.clamp(cycle_r, min=tiny)))
            upper_amount = wallet_amount * float(self.amount_max)
            lower_amount = wallet_amount * float(self.amount_min)
            stationary = th.where(
                cycle_d > tiny, qstar_numerator / th.clamp(cycle_d, min=tiny), upper_amount
            )
            bounded_stationary = th.maximum(lower_amount, th.minimum(stationary, upper_amount))
            qstar_raw = th.where(
                qstar_valid & th.isfinite(bounded_stationary),
                bounded_stationary,
                th.zeros_like(bounded_stationary),
            )
            myopic_fraction = th.where(
                wallet_amount > tiny,
                qstar_raw / th.clamp(wallet_amount, min=tiny),
                th.zeros_like(qstar_raw),
            )
            myopic_fraction = th.clamp(myopic_fraction, min=0.0, max=1.0)
            log_raw = th.log(th.clamp(raw_reserves, min=tiny))
            log_reserve_in = log_raw[..., 0::2]
            log_reserve_out = log_raw[..., 1::2]
            log_spot = log_reserve_out - log_reserve_in
            log_capacity_start_units = (
                th.stack(
                    (
                        log_reserve_in[..., 0],
                        log_reserve_in[..., 1] - log_spot[..., 0],
                        log_reserve_in[..., 2] - log_spot[..., 0] - log_spot[..., 1],
                    ),
                    dim=-1,
                )
                .min(dim=-1)
                .values
            )
            log_capacity_usd = log_capacity_start_units + th.log(th.clamp(start_prices, min=tiny))
            log_wallet_usd = th.log(th.clamp(wallet_usd, min=tiny))
            liquidity_capacity_score = th.sigmoid(log_capacity_usd - log_wallet_usd)
            liquidity_capacity_score = th.where(
                qstar_valid | mechanically_valid,
                liquidity_capacity_score,
                th.zeros_like(liquidity_capacity_score),
            )
            extra_features = th.stack((liquidity_capacity_score, myopic_fraction), dim=-1)
            derived_features = th.cat(
                (fixed_profit_score, extra_features.to(dtype=observations.dtype)), dim=-1
            )
            VariablePathLogisticNormalDistribution._require_finite(
                f"{self.path_feature_mode} derived path features", derived_features
            )
            features = th.cat((base_features, derived_features), dim=-1)
            if int(features.shape[-1]) != self.path_feature_dim:
                raise RuntimeError(
                    f"Expected {self.path_feature_dim} {self.path_feature_mode} path features, received {features.shape[-1]}."
                )
            return features

        def _fstar_amount_mu_target(self, path_features: Any) -> Tuple[Any, Any]:
            if int(path_features.shape[-1]) <= FSTAR_FEATURE_INDEX:
                raise ValueError("Residual amount target feature is unavailable.")
            dtype = path_features.dtype
            fstar = path_features[..., FSTAR_FEATURE_INDEX].to(dtype=th.float64)
            unit = (fstar - float(self.action_dist.sampling_min)) / float(
                self.action_dist.amount_span
            )
            unit_min = float(1.0 / (1.0 + math.exp(PPO_AMOUNT_MU_ABS_MAX)))
            unit = unit.clamp(unit_min, 1.0 - unit_min)
            target_mu = (th.log(unit) - th.log1p(-unit)).clamp(
                -PPO_AMOUNT_MU_ABS_MAX, PPO_AMOUNT_MU_ABS_MAX
            )
            projected = float(self.action_dist.sampling_min) + float(
                self.action_dist.amount_span
            ) * th.sigmoid(target_mu)
            return (target_mu.to(dtype=dtype), projected.to(dtype=dtype))

        def _policy_outputs(self, obs: Any) -> Tuple[Any, Any, Any, Any]:
            observations = self.extract_features(obs)
            path_features = self._path_features(observations)
            path_latent = self.path_encoder(path_features)
            path_logits = self.path_action_head(path_latent).squeeze(-1)
            amount_params = self.amount_action_head(path_latent)
            base_mu, _projected = self._fstar_amount_mu_target(path_features)
            amount_mu = (base_mu + amount_params[..., 0]).clamp(
                -PPO_AMOUNT_MU_ABS_MAX, PPO_AMOUNT_MU_ABS_MAX
            )
            amount_log_std = self.amount_log_std_min + (
                AMOUNT_LOG_STD_MAX - self.amount_log_std_min
            ) * th.sigmoid(amount_params[..., 1])
            pooled_mean = path_latent.mean(dim=1)
            pooled_max = path_latent.max(dim=1).values
            values = self.value_head(th.cat((pooled_mean, pooled_max), dim=1))
            for name, value in (
                ("path logits", path_logits),
                ("amount mean", amount_mu),
                ("amount log standard deviation", amount_log_std),
                ("value estimate", values),
            ):
                VariablePathLogisticNormalDistribution._require_finite(name, value)
            return (path_logits, amount_mu, amount_log_std, values)

        def _distribution(self, obs: Any) -> Tuple[VariablePathLogisticNormalDistribution, Any]:
            path_logits, amount_mu, amount_log_std, values = self._policy_outputs(obs)
            distribution = self.action_dist.proba_distribution(
                path_logits, amount_mu, amount_log_std
            )
            return (distribution, values)

        def _get_action_dist_from_latent(self, _latent_pi: Any) -> Any:
            raise NotImplementedError(
                "VariablePathHybridPPOPolicy builds its distribution from observations."
            )

        def forward(self, obs: Any, deterministic: bool = False) -> Tuple[Any, Any, Any]:
            distribution, values = self._distribution(obs)
            actions = distribution.get_actions(deterministic=deterministic)
            return (
                actions.reshape((-1, *self.action_space.shape)),
                values,
                distribution.log_prob(actions),
            )

        def evaluate_actions(self, obs: Any, actions: Any) -> Tuple[Any, Any, Any]:
            distribution, values = self._distribution(obs)
            return (values, distribution.log_prob(actions), distribution.entropy())

        def get_distribution(self, obs: Any) -> VariablePathLogisticNormalDistribution:
            distribution, _values = self._distribution(obs)
            return distribution

        def predict_values(self, obs: Any) -> Any:
            _distribution, values = self._distribution(obs)
            return values

        def _predict(self, observation: Any, deterministic: bool = False) -> Any:
            return self.get_distribution(observation).get_actions(deterministic=deterministic)

        def _get_constructor_parameters(self) -> Dict[str, Any]:
            data = super()._get_constructor_parameters()
            data.update(
                {
                    "path_policy_spec": self._path_policy_spec_serialized,
                    "path_feature_mode": self.path_feature_mode,
                    "amount_min": self.amount_min,
                    "amount_max": self.amount_max,
                    "amount_log_std_min": self.amount_log_std_min,
                    "initial_amount_log_std": self.initial_amount_log_std,
                    "amount_treatment": self.amount_treatment,
                    "wallet_curriculum": self.wallet_curriculum,
                }
            )
            return data


@dataclass(frozen=True)
class Pool:
    pool_id: str
    label: str
    token0: str
    token1: str
    reserve0: float
    reserve1: float
    token0_symbol: str = ""
    token1_symbol: str = ""
    fee: float = FEE_DEFAULT
    dex_type: str = "V2"


@dataclass
class GasSample:
    gas_price_gwei: float
    gas_used: float
    eth_usd: float
    gas_cost_usd: float
    source_row_index: int = -1


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float, np.integer, np.floating)):
        out = float(value)
        return out if math.isfinite(out) else default
    if isinstance(value, Mapping):
        for key in ("amount", "usd", "value", "reserve", "value_usd"):
            if key in value:
                return safe_float(value[key], default)
        return default
    text = str(value).strip().replace(",", "").replace("$", "")
    if not text or text.lower() in {"nan", "none", "null", "n/a", "na"}:
        return default
    try:
        out = float(text)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def canonical_symbol(value: Any) -> str:
    symbol = re.sub("\\s+", " ", str(value).strip()).upper()
    return "ETH" if symbol in {"WETH", "ETHEREUM"} else symbol


def parse_coingecko_id_order(dataset_dir: Path) -> List[str]:
    generator = Path(dataset_dir) / TOKEN_PRICE_GENERATOR_FILENAME
    if not generator.is_file():
        raise FileNotFoundError(f"Missing required price-order input {generator}.")
    for line in generator.read_text(encoding="utf-8", errors="strict").splitlines():
        stripped = line.strip()
        if stripped.startswith(("url=", "url =")):
            match = re.search("ids=([^&\\\"']+)", stripped)
            if match:
                return [item.strip() for item in match.group(1).split(",") if item.strip()]
    raise ValueError(f"Could not parse CoinGecko id order from {generator}.")


def symbol_order_from_pools(raw_pools: Sequence[Mapping[str, Any]]) -> List[str]:
    order: List[str] = []
    for pool in raw_pools:
        for key in ("token0", "token1"):
            symbol = canonical_symbol(pool.get(key, ""))
            if symbol and symbol not in order:
                order.append(symbol)
    return order


def build_symbol_to_coingecko_id(
    dataset_dir: Path, raw_pools: Sequence[Mapping[str, Any]]
) -> Dict[str, str]:
    symbols = symbol_order_from_pools(raw_pools)
    coin_ids = parse_coingecko_id_order(dataset_dir)
    if len(symbols) != EXPECTED_TOKEN_COUNT or len(coin_ids) != len(symbols):
        raise ValueError(
            f"Legacy symbol/price-generator ordering mismatch: {len(symbols)} symbols versus {len(coin_ids)} CoinGecko ids."
        )
    return dict(zip(symbols, coin_ids))


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def participating_pool_ids(
    paths: Sequence[Tuple[str, str, str, Tuple[str, str, str]]],
) -> List[str]:
    return sorted({pool_id for path in paths for pool_id in path[3]})


def legacy_symbol_collision_audit(
    raw_pools: Sequence[Mapping[str, Any]],
    paths: Sequence[Tuple[str, str, str, Tuple[str, str, str]]],
) -> Dict[str, Any]:
    symbol_addresses: Dict[str, set] = {}
    pool_tokens: Dict[str, Dict[str, str]] = {}
    for index, raw_pool in enumerate(raw_pools):
        dex_type = str(raw_pool.get("type", "")).upper()
        label = str(raw_pool.get("pool", ""))
        pool_id = f"{index:04d}_{dex_type}_{label}".replace(" ", "_")
        endpoints: Dict[str, str] = {}
        for side in (0, 1):
            symbol = canonical_symbol(raw_pool.get(f"token{side}"))
            address = str(raw_pool.get(f"token{side}_address", "")).strip().lower()
            if re.fullmatch("0x[0-9a-f]{40}", address) is None:
                raise ValueError(f"Pool row {index} has an invalid token{side}_address.")
            endpoints[symbol] = address
            symbol_addresses.setdefault(symbol, set()).add(address)
        pool_tokens[pool_id] = endpoints
    discontinuous = 0
    for token_a, token_b, token_c, pool_ids in paths:
        cycle_tokens = (token_a, token_b, token_c)
        if any(
            (
                pool_tokens[pool_ids[leg]][cycle_tokens[(leg + 1) % 3]]
                != pool_tokens[pool_ids[(leg + 1) % 3]][cycle_tokens[(leg + 1) % 3]]
                for leg in range(3)
            )
        ):
            discontinuous += 1
    colliding = sorted(
        (symbol for symbol, addresses in symbol_addresses.items() if len(addresses) > 1)
    )
    return {
        "colliding_symbols": colliding,
        "colliding_symbol_count": len(colliding),
        "contract_discontinuous_path_count": discontinuous,
        "contract_continuous_path_count": len(paths) - discontinuous,
    }


def discover_dataset_dir(path: Path) -> Path:
    root = Path(path).expanduser().resolve()
    candidates = [root, root / "DATASET"]
    for candidate in candidates:
        if (candidate / POOL_DATA_FILENAME).is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find {POOL_DATA_FILENAME} under {root}. Point --dataset-dir to the DATASET folder."
    )


def wallet_budget_sort_key(path: Path) -> Tuple[int, str]:
    match = re.search("wallet_balance_[^.]+(?:-[^.]+)*\\.(\\d+(?:\\.\\d+)?)\\.json$", path.name)
    if match:
        return (int(float(match.group(1))), path.name)
    return (10**18, path.name)


def load_prices_by_symbol(
    dataset_dir: Path, raw_pools: Sequence[Mapping[str, Any]]
) -> Tuple[Dict[str, float], Dict[str, str]]:
    price_path = dataset_dir / TOKEN_PRICE_FILENAME
    if not price_path.is_file():
        raise FileNotFoundError(f"Missing required token-price file {price_path}.")
    payload = load_json(price_path)
    if payload.get("date") != DATASET_DATE:
        raise ValueError(f"{price_path} must record date {DATASET_DATE}.")
    raw_prices = payload.get("prices", payload)
    if not isinstance(raw_prices, Mapping):
        raise ValueError(f"{price_path} does not contain a prices mapping.")
    symbol_to_id = build_symbol_to_coingecko_id(dataset_dir, raw_pools)
    prices: Dict[str, float] = {}
    for symbol, coin_id in symbol_to_id.items():
        raw_value = raw_prices.get(coin_id)
        price = safe_float(raw_value.get("usd") if isinstance(raw_value, Mapping) else raw_value)
        if price <= 0.0:
            raise ValueError(f"{price_path} lacks a positive {coin_id!r} price for {symbol!r}.")
        prices[symbol] = price
    missing = sorted(set(FUNDED_TOKEN_SYMBOL_BY_ID.values()) - set(prices))
    if missing:
        raise ValueError(f"{price_path} lacks positive prices for funded symbols {missing}.")
    return (prices, symbol_to_id)


def load_wallets_by_symbol(
    dataset_dir: Path, prices: Mapping[str, float], symbol_to_id: Mapping[str, str]
) -> List[Tuple[str, Dict[str, float]]]:
    wallet_paths = sorted(dataset_dir.glob(WALLET_FILE_GLOB), key=wallet_budget_sort_key)
    if len(wallet_paths) != len(EXPECTED_WALLET_BUDGETS):
        raise ValueError(
            f"Expected {len(EXPECTED_WALLET_BUDGETS)} wallet files, found {len(wallet_paths)} in {dataset_dir}."
        )
    wallets: List[Tuple[str, Dict[str, float]]] = []
    observed_budgets: List[int] = []
    inverse = {coin_id: symbol for symbol, coin_id in symbol_to_id.items()}
    expected_symbols = set(FUNDED_TOKEN_SYMBOL_BY_ID.values())
    for wallet_path in wallet_paths:
        payload = load_json(wallet_path)
        if payload.get("date") != DATASET_DATE:
            raise ValueError(f"{wallet_path} must record date {DATASET_DATE}.")
        metadata = payload.get("wallet_metadata", {})
        budget = int(round(safe_float(metadata.get("total_budget_usd"), -1.0)))
        observed_budgets.append(budget)
        wallet_raw = payload.get("wallet", payload)
        if not isinstance(wallet_raw, Mapping):
            raise ValueError(f"{wallet_path} does not contain a wallet mapping.")
        if len(wallet_raw) != EXPECTED_FUNDED_TOKEN_COUNT:
            raise ValueError(f"{wallet_path} must contain exactly five funded-token allocations.")
        wallet: Dict[str, float] = {}
        wallet_total = 0.0
        expected_value_usd = float(budget) / float(EXPECTED_FUNDED_TOKEN_COUNT)
        for raw_key, raw_value in wallet_raw.items():
            symbol = inverse.get(str(raw_key))
            if symbol not in expected_symbols:
                raise ValueError(f"{wallet_path} contains unsupported wallet token {raw_key!r}.")
            amount = safe_float(
                raw_value.get("amount") if isinstance(raw_value, Mapping) else raw_value
            )
            value_usd = safe_float(
                raw_value.get("value_usd") if isinstance(raw_value, Mapping) else 0.0
            )
            allocation = safe_float(
                raw_value.get("allocation_weight") if isinstance(raw_value, Mapping) else 0.0
            )
            recorded_price = safe_float(
                raw_value.get("price_usd") if isinstance(raw_value, Mapping) else 0.0
            )
            canonical_price = float(prices.get(symbol, 0.0))
            if amount <= 0.0 or not math.isclose(allocation, 0.2, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"{wallet_path} has an invalid funded-token allocation.")
            if canonical_price <= 0.0 or not math.isclose(
                recorded_price, canonical_price, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(
                    f"{wallet_path} has a price inconsistent with the canonical token-price file for {raw_key!r}."
                )
            if not math.isclose(value_usd, expected_value_usd, rel_tol=0.0, abs_tol=1e-06):
                raise ValueError(
                    f"{wallet_path} must allocate 20% of USD {budget:,} to each funded token."
                )
            if not math.isclose(amount * canonical_price, value_usd, rel_tol=1e-12, abs_tol=1e-06):
                raise ValueError(
                    f"{wallet_path} has amount x price inconsistent with value_usd for {raw_key!r}."
                )
            wallet[symbol] = amount
            wallet_total += value_usd
        if set(wallet) != expected_symbols:
            raise ValueError(f"{wallet_path} must fund exactly the five configured symbols.")
        if not math.isclose(wallet_total, float(budget), rel_tol=0.0, abs_tol=1e-06):
            raise ValueError(f"{wallet_path} values do not sum to USD {budget:,}.")
        wallets.append((wallet_path.name, wallet))
    if tuple(observed_budgets) != EXPECTED_WALLET_BUDGETS:
        raise ValueError(
            f"Wallet budgets must be {EXPECTED_WALLET_BUDGETS}; loaded {tuple(observed_budgets)}."
        )
    return wallets


def extract_reserve(pool: Mapping[str, Any], token: str, side: int) -> float:
    adjusted = pool.get("reserves_adjusted") or {}
    if isinstance(adjusted, Mapping):
        candidates = (
            token,
            canonical_symbol(token),
            token.upper(),
            token.lower(),
            "WETH" if canonical_symbol(token) == "ETH" else token,
        )
        for key in candidates:
            value = safe_float(adjusted.get(key))
            if value > 0.0:
                return value
    for key in (f"reserve{side}_adjusted", f"reserve{side}", f"reserves{side}"):
        value = safe_float(pool.get(key))
        if value > 0.0:
            return value
    return 0.0


def load_pools(
    dataset_dir: Path, prices: Mapping[str, float], fee_default: float = FEE_DEFAULT
) -> List[Pool]:
    pool_path = dataset_dir / POOL_DATA_FILENAME
    raw_pools = load_json(pool_path)
    if not isinstance(raw_pools, list):
        raise ValueError(f"{pool_path} must contain a JSON list of pool records.")
    if len(raw_pools) != EXPECTED_POOL_COUNT:
        raise ValueError(f"{pool_path} must contain exactly {EXPECTED_POOL_COUNT} source rows.")
    pools: List[Pool] = []
    for index, raw_pool in enumerate(raw_pools):
        token0 = canonical_symbol(raw_pool.get("token0"))
        token1 = canonical_symbol(raw_pool.get("token1"))
        if token0 == token1:
            raise ValueError(f"Pool row {index} has identical canonical symbols.")
        dex_type = str(raw_pool.get("type", "")).upper()
        if dex_type not in {"UNISWAP_V2", "SUSHISWAP_V2"}:
            raise ValueError(f"Pool row {index} has unsupported DEX type {dex_type!r}.")
        if token0 not in prices or token1 not in prices:
            raise ValueError(f"Pool row {index} contains an unpriced symbol.")
        reserve0 = extract_reserve(raw_pool, str(raw_pool.get("token0", "")), 0)
        reserve1 = extract_reserve(raw_pool, str(raw_pool.get("token1", "")), 1)
        if reserve0 <= 0.0 or reserve1 <= 0.0:
            raise ValueError(f"Pool row {index} has nonpositive adjusted reserves.")
        fee = safe_float(raw_pool.get("fee"), fee_default)
        if not 0.0 < fee < 1.0:
            fee = fee_default
        label = str(raw_pool.get("pool", f"{token0}/{token1}"))
        pool_id = f"{index:04d}_{dex_type}_{label}".replace(" ", "_")
        pools.append(
            Pool(
                pool_id=pool_id,
                label=label,
                token0=token0,
                token1=token1,
                reserve0=reserve0,
                reserve1=reserve1,
                token0_symbol=token0,
                token1_symbol=token1,
                fee=fee,
                dex_type=dex_type,
            )
        )
    dex_counts = Counter((pool.dex_type for pool in pools))
    expected_dex_counts = Counter(
        {"UNISWAP_V2": EXPECTED_UNISWAP_POOL_COUNT, "SUSHISWAP_V2": EXPECTED_SUSHISWAP_POOL_COUNT}
    )
    if len(pools) != EXPECTED_POOL_COUNT or dex_counts != expected_dex_counts:
        raise ValueError(
            f"Pool audit loaded {len(pools)} pools with {dict(dex_counts)}; expected {EXPECTED_POOL_COUNT} with {dict(expected_dex_counts)}."
        )
    token_count = len({token for pool in pools for token in (pool.token0, pool.token1)})
    if token_count != EXPECTED_TOKEN_COUNT:
        raise ValueError(
            f"Pool audit found {token_count} token addresses; expected {EXPECTED_TOKEN_COUNT}."
        )
    return pools


class GasCostModel:
    """Seeded empirical bootstrap of paired gas price and gas usage rows."""

    def __init__(
        self,
        dataset_dir: Path,
        eth_usd: float,
        seed: int = SEED_DEFAULT,
        gas_multiplier: float = 1.0,
    ):
        self.dataset_dir = dataset_dir
        self.eth_usd = float(eth_usd)
        self.rng = np.random.default_rng(seed)
        self.sample_counter = 0
        self.gas_multiplier = float(gas_multiplier)
        self.gas_prices: np.ndarray = np.empty(0, dtype=float)
        self.gas_used: np.ndarray = np.empty(0, dtype=float)
        self.source_row_indices: np.ndarray = np.empty(0, dtype=int)
        self.empirical_gas_costs_usd: np.ndarray = np.empty(0, dtype=float)
        self._load_book1()

    def _load_book1(self) -> None:
        book = self.dataset_dir / "Book1.xlsx"
        if not book.is_file():
            raise FileNotFoundError(f"Required paired gas sample is missing: {book}")
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("pandas and openpyxl are required for the gas model.") from exc
        try:
            try:
                frame = pd.read_excel(book, sheet_name="ARBITRAGE_RECORD", engine="openpyxl")
            except Exception:
                frame = pd.read_excel(book, engine="openpyxl")
            columns = {str(column).strip().lower(): column for column in frame.columns}
            price_column = columns.get("gas price") or columns.get("gasprice")
            used_column = (
                columns.get("gas usage") or columns.get("gas used") or columns.get("gasusage")
            )
            if price_column is None or used_column is None:
                raise ValueError("Book1.xlsx must contain Gas Price and Gas Usage columns.")
            prices = pd.to_numeric(frame[price_column], errors="coerce")
            usage = pd.to_numeric(frame[used_column], errors="coerce")
            valid = prices.notna() & usage.notna() & (prices > 0) & (usage > 0)
            if int(valid.sum()) < 2:
                raise ValueError("Book1.xlsx contains fewer than two valid paired gas rows.")
            self.gas_prices = prices.loc[valid].to_numpy(dtype=float)
            self.gas_used = usage.loc[valid].to_numpy(dtype=float)
            self.source_row_indices = frame.index[valid].to_numpy(dtype=int)
            self.empirical_gas_costs_usd = self.gas_prices * self.gas_used * self.eth_usd * 1e-09
        except Exception as exc:
            raise RuntimeError(f"Could not load paired gas rows from {book}: {exc}") from exc

    def reseed(self, seed: int) -> None:
        self.rng = np.random.default_rng(int(seed))
        self.sample_counter = 0

    def sample(self) -> GasSample:
        position = int(self.rng.integers(0, len(self.gas_prices)))
        self.sample_counter += 1
        gas_price_gwei = float(self.gas_prices[position]) * self.gas_multiplier
        gas_used = float(self.gas_used[position])
        gas_cost_usd = gas_price_gwei * gas_used * self.eth_usd * 1e-09
        return GasSample(
            gas_price_gwei=gas_price_gwei,
            gas_used=gas_used,
            eth_usd=self.eth_usd,
            gas_cost_usd=gas_cost_usd,
            source_row_index=int(self.source_row_indices[position]),
        )


class PPR01TriangularArbitrageEnv(gym.Env if gym is not None else object):
    """Sequential three-swap arbitrage environment shared by all DRL methods."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        dataset_dir: Path,
        seed: int = SEED_DEFAULT,
        amount_min: float = AMOUNT_MIN_DEFAULT,
        amount_max: float = AMOUNT_MAX_DEFAULT,
        gas_multiplier: float = 1.0,
        max_paths: int = 5000,
        wallet_mode: str = "cycle",
        block_time_seconds: float = 12.0,
        action_mode: str = ACTION_MODE_CONTINUOUS_RELAXED,
        amount_grid: Optional[Sequence[float]] = None,
        reward_reference_usd: float = REWARD_REFERENCE_USD_DEFAULT,
        clock: Optional[Callable[[], float]] = None,
        protocol_id: str = PROTOCOL_ID,
        fixed_wallet_usd: Optional[int] = None,
        c1_rejection_reward: float = C1_REJECTION_REWARD,
        path_feature_mode: str = FIXED_FEATURE_MODE,
        c1_counterfactual_reward: bool = False,
    ):
        if gym is not None:
            super().__init__()
        if str(protocol_id) != PROTOCOL_ID:
            raise ValueError(
                f"Protocol mismatch: expected {PROTOCOL_ID!r}, received {protocol_id!r}."
            )
        self.protocol_id = PROTOCOL_ID
        self.dataset_dir = discover_dataset_dir(Path(dataset_dir))
        self.pool_dataset_file = self.dataset_dir / POOL_DATA_FILENAME
        self.seed_value = int(seed)
        self.rng = np.random.default_rng(seed)
        self.amount_min = float(max(amount_min, 0.0))
        self.amount_max = float(np.clip(amount_max, self.amount_min, 1.0))
        if self.amount_max <= self.amount_min:
            raise ValueError("amount_max must be greater than amount_min.")
        self.min_profit_usd = 0.0
        self.gas_multiplier = float(gas_multiplier)
        self.max_paths = int(max_paths)
        if self.max_paths <= 0:
            raise ValueError("max_paths must be a positive topology validation limit.")
        self.wallet_mode = str(wallet_mode)
        self.gas_resample_mode = "episode"
        self.include_path_profitability = False
        self.charge_gas_on_failure = False
        self.block_time_seconds = float(block_time_seconds)
        if self.block_time_seconds <= 0.0:
            raise ValueError("block_time_seconds must be positive.")
        self.reward_reference_usd = float(reward_reference_usd)
        if self.reward_reference_usd <= 0.0:
            raise ValueError("reward_reference_usd must be positive.")
        self.c1_rejection_reward = canonical_c1_rejection_reward(c1_rejection_reward)
        self.c1_counterfactual_reward = bool(c1_counterfactual_reward)
        if self.c1_counterfactual_reward and self.c1_rejection_reward != 0.0:
            raise ValueError("Counterfactual C1 shaping requires the fixed C1 fallback to be 0.")
        self.physical_rejection_reward = PHYSICAL_REJECTION_REWARD
        self.c6_expiry_reward = C6_EXPIRY_REWARD
        self.path_feature_mode = canonical_ppo_path_feature_mode(path_feature_mode)
        self._clock: Callable[[], float] = clock or time.perf_counter
        self.action_mode = str(action_mode).strip().lower()
        if self.action_mode not in {ACTION_MODE_HYBRID_PPO, ACTION_MODE_CONTINUOUS_RELAXED}:
            raise ValueError(f"Unsupported v2.09 action_mode {action_mode!r}")
        self.amount_grid = list(amount_grid or [])
        self.action_space_form = ""
        self.action_space_note = ""
        raw_pools = load_json(self.pool_dataset_file)
        self.prices, self.symbol_to_coingecko_id = load_prices_by_symbol(
            self.dataset_dir, raw_pools
        )
        self.wallets = load_wallets_by_symbol(
            self.dataset_dir, self.prices, self.symbol_to_coingecko_id
        )
        self.fixed_wallet_usd = None if fixed_wallet_usd is None else int(fixed_wallet_usd)
        self.fixed_wallet_index: Optional[int] = None
        if self.fixed_wallet_usd is not None:
            matches = [
                index
                for index, (_name, wallet) in enumerate(self.wallets)
                if math.isclose(
                    self._portfolio_value_usd(wallet),
                    float(self.fixed_wallet_usd),
                    rel_tol=0.0,
                    abs_tol=1e-06,
                )
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one USD {self.fixed_wallet_usd:,} wallet; found {len(matches)}."
                )
            self.fixed_wallet_index = int(matches[0])
        self.wallet_start_tokens = sorted(
            {
                token
                for _, wallet in self.wallets
                for token, amount in wallet.items()
                if float(amount) > 0.0
            }
        )
        if len(self.wallet_start_tokens) != EXPECTED_FUNDED_TOKEN_COUNT:
            raise ValueError(
                f"v2.09 legacy114 expects five funded token symbols; loaded {len(self.wallet_start_tokens)}: {self.wallet_start_tokens}."
            )
        self.wallet_observation_tokens = tuple(self.wallet_start_tokens)
        self.pool_list = load_pools(self.dataset_dir, self.prices)
        self.dataset_mode = DATASET_IDENTITY
        self.pool_template: Dict[str, Pool] = {pool.pool_id: pool for pool in self.pool_list}
        self.tokens = sorted(
            {token for pool in self.pool_list for token in (pool.token0, pool.token1)}
            | {token for _, wallet in self.wallets for token in wallet}
        )
        self.token_symbols = {symbol: symbol for symbol in self.tokens}
        eth_price = self.prices.get(ETH_SYMBOL, 0.0)
        if eth_price <= 0.0:
            raise ValueError("ETH price is required for gas cost conversion.")
        self.gas_model = GasCostModel(
            self.dataset_dir, eth_usd=eth_price, seed=seed, gas_multiplier=self.gas_multiplier
        )
        self.triangular_paths = self._build_triangular_paths()
        if (len(self.tokens), len(self.pool_list), len(self.triangular_paths)) != (
            EXPECTED_TOKEN_COUNT,
            EXPECTED_POOL_COUNT,
            REFERENCE_PATH_COUNT,
        ):
            raise ValueError(
                f"The canonical v2.09 legacy-symbol catalogue expects {EXPECTED_TOKEN_COUNT} tokens, {EXPECTED_POOL_COUNT} pools, and {REFERENCE_PATH_COUNT} paths; loaded {len(self.tokens)}, {len(self.pool_list)}, and {len(self.triangular_paths)}."
            )
        if not self.triangular_paths:
            raise ValueError("The structural path catalogue must contain at least one cycle.")
        participating_count = len(participating_pool_ids(self.triangular_paths))
        if participating_count != EXPECTED_PARTICIPATING_POOL_COUNT:
            raise RuntimeError(
                f"Expected {EXPECTED_PARTICIPATING_POOL_COUNT} participating pools; loaded {participating_count}/{len(self.pool_list)}."
            )
        self.legacy_symbol_collision = legacy_symbol_collision_audit(
            raw_pools, self.triangular_paths
        )
        if self.legacy_symbol_collision != {
            "colliding_symbols": ["OHM", "WOJAK"],
            "colliding_symbol_count": 2,
            "contract_discontinuous_path_count": 8,
            "contract_continuous_path_count": 106,
        }:
            raise RuntimeError(
                f"Unexpected legacy ticker-collision audit: {self.legacy_symbol_collision}."
            )
        self.normalization_wallets = copy.deepcopy(self.wallets)
        self.normalization_pool_list = copy.deepcopy(self.pool_list)
        self.wallet_feature_start = 0
        self.reserve_feature_start = len(self.wallet_observation_tokens)
        self.gas_feature_index = self.reserve_feature_start + 2 * len(self.pool_list)
        self.remaining_time_feature_index = self.gas_feature_index + 1
        obs_dim = self.remaining_time_feature_index + 1
        if obs_dim != OBSERVATION_DIM:
            raise ValueError(
                f"v2.09 observation must contain {OBSERVATION_DIM} values; computed {obs_dim}."
            )
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self._configure_action_space()
        self.reset_counter = 0
        self.current_wallet_index = 0
        self.wallet_name = ""
        self.initial_balance: Dict[str, float] = {}
        self.current_balance: Dict[str, float] = {}
        self.current_pools: Dict[str, Dict[str, float]] = {}
        self.current_gas = self.gas_model.sample()
        self.initial_portfolio_usd = 0.0
        self.total_profit_usd = 0.0
        self.total_gross_usd = 0.0
        self.total_gas_usd = 0.0
        self.successful_arbitrages = 0
        self.failed_actions = 0
        self.episode_steps = 0
        self.last_profit_usd = 0.0
        self.cumulative_decision_time_seconds = 0.0
        self.decision_phase_seconds: Dict[str, float] = {}
        self.cumulative_common_stop_check_time_seconds = 0.0
        self.common_stop_check_phase_seconds: Dict[str, float] = {}
        self._decision_clock_active = False
        self._decision_clock_started_at = 0.0
        self.normalization_clipping_count = 0
        self.normalization_clipping_by_feature: Dict[str, int] = {}
        self.token_amount_references: Dict[str, float] = {}
        self.wallet_value_references_usd: Dict[str, float] = {}
        self.reserve_amount_references: Dict[str, float] = {}
        self.gas_cost_reference_usd = 1.0
        self._cached_common_stop_reason = ""
        self._cached_common_stop_best: Optional[Dict[str, Any]] = None
        self._cached_common_stop_state_key: Optional[Tuple[float, ...]] = None
        self.last_info: Dict[str, Any] = {}
        self.refresh_normalization_references()
        self.reset(seed=seed)
        self.reset_counter = 0

    def _configure_action_space(self) -> None:
        if self.action_mode == ACTION_MODE_HYBRID_PPO:
            self.action_space = CategoricalContinuousActionSpace(
                path_count=len(self.triangular_paths),
                amount_min=self.amount_min,
                amount_max=self.amount_max,
            )
            self.action_space_form = "categorical_path_continuous_logistic_normal_amount"
        else:
            self.action_space = spaces.Box(
                low=np.asarray([0.0, self.amount_min], dtype=np.float32),
                high=np.asarray([1.0, self.amount_max], dtype=np.float32),
                dtype=np.float32,
            )
            self.action_space_form = "continuous_relaxed_path_continuous_amount"
        self.action_space_note = action_space_note_for_mode(self.action_mode)

    def _action_from_path_amount(self, path_index: int, amount_fraction: float) -> np.ndarray:
        path_index = int(np.clip(path_index, 0, len(self.triangular_paths) - 1))
        amount_fraction = float(np.clip(amount_fraction, self.amount_min, self.amount_max))
        if self.action_mode == ACTION_MODE_HYBRID_PPO:
            return np.asarray([float(path_index), amount_fraction], dtype=np.float32)
        selector = (path_index + 0.5) / float(len(self.triangular_paths))
        return np.asarray([selector, amount_fraction], dtype=np.float32)

    def _decode_action(self, action: Any) -> Tuple[int, float, Dict[str, Any]]:
        arr = np.asarray(action, dtype=float).reshape(-1)
        if len(arr) != 2 or not np.isfinite(arr).all():
            raise ValueError(
                f"Actions require exactly two finite values; received shape {np.asarray(action).shape}."
            )
        if self.action_mode == ACTION_MODE_HYBRID_PPO:
            rounded = int(round(float(arr[0])))
            if abs(float(arr[0]) - rounded) > 0.0001:
                raise ValueError(
                    f"Hybrid PPO path value must be categorical/integer-like; received {arr[0]!r}."
                )
            if rounded < 0 or rounded >= len(self.triangular_paths):
                raise ValueError(
                    f"Hybrid PPO path index {rounded} is outside [0, {len(self.triangular_paths) - 1}]."
                )
            path_index = rounded
            raw_selector = float(arr[0])
        else:
            raw_selector = float(np.clip(arr[0], 0.0, np.nextafter(1.0, 0.0)))
            path_index = min(
                int(raw_selector * len(self.triangular_paths)), len(self.triangular_paths) - 1
            )
        amount_fraction = float(np.clip(arr[1], self.amount_min, self.amount_max))
        return (
            path_index,
            amount_fraction,
            {
                "raw_action_0_selector": raw_selector,
                "raw_action_1_amount": float(arr[1]),
                "path_sample_is_categorical": self.action_mode == ACTION_MODE_HYBRID_PPO,
                "amount_is_continuous": True,
                "amount_grid_size": 0,
                "action_head_architecture": (
                    PPO_ACTION_HEAD_ARCHITECTURE
                    if self.action_mode == ACTION_MODE_HYBRID_PPO
                    else "continuous_relaxed_actor"
                ),
            },
        )

    def _pool_state_from_template(self) -> Dict[str, Dict[str, float]]:
        return {
            pool.pool_id: {
                "token0": pool.token0,
                "token1": pool.token1,
                "reserve0": float(pool.reserve0),
                "reserve1": float(pool.reserve1),
                "fee": float(pool.fee),
            }
            for pool in self.pool_list
        }

    def _build_triangular_paths(self) -> List[Tuple[str, str, str, Tuple[str, str, str]]]:
        adjacency: Dict[str, Dict[str, List[str]]] = {}
        for pool in self.pool_list:
            adjacency.setdefault(pool.token0, {}).setdefault(pool.token1, []).append(pool.pool_id)
            adjacency.setdefault(pool.token1, {}).setdefault(pool.token0, []).append(pool.pool_id)
        paths: List[Tuple[str, str, str, Tuple[str, str, str]]] = []
        for token_a in sorted(adjacency):
            if token_a not in self.wallet_start_tokens:
                continue
            for token_b in sorted(adjacency.get(token_a, {})):
                if token_b == token_a:
                    continue
                for token_c in sorted(adjacency.get(token_b, {})):
                    if token_c in {token_a, token_b} or token_a not in adjacency.get(token_c, {}):
                        continue
                    for pool_ab in adjacency[token_a][token_b]:
                        for pool_bc in adjacency[token_b][token_c]:
                            for pool_ca in adjacency[token_c][token_a]:
                                if len({pool_ab, pool_bc, pool_ca}) == 3:
                                    paths.append(
                                        (token_a, token_b, token_c, (pool_ab, pool_bc, pool_ca))
                                    )
        paths = sorted(set(paths), key=lambda path: (path[0], path[1], path[2], path[3]))
        if len(paths) > self.max_paths:
            raise RuntimeError(
                f"The structural catalogue contains {len(paths)} paths, exceeding the declared safety limit {self.max_paths}. No path was silently discarded."
            )
        return paths

    def token_display_label(self, token: str) -> str:
        symbol = canonical_symbol(token)
        if symbol not in self.token_symbols:
            raise ValueError(f"Unknown legacy graph token {token!r}.")
        return symbol

    def format_path_tokens(self, path: Tuple[str, str, str, Tuple[str, str, str]]) -> str:
        labels = [self.token_display_label(token) for token in path[:3]]
        return "->".join(labels + [labels[0]])

    def refresh_normalization_references(self) -> None:
        max_wallet: Dict[str, float] = {token: 0.0 for token in self.tokens}
        for _name, wallet in self.normalization_wallets:
            for token in self.tokens:
                max_wallet[token] = max(max_wallet[token], float(wallet.get(token, 0.0)))
        pool_totals: Dict[str, float] = {token: 0.0 for token in self.tokens}
        for pool in self.normalization_pool_list:
            pool_totals[pool.token0] += float(pool.reserve0)
            pool_totals[pool.token1] += float(pool.reserve1)
        self.token_amount_references = {
            token: max(max_wallet[token] + pool_totals[token], EPS) for token in self.tokens
        }
        self.wallet_value_references_usd = {
            token: WALLET_NORMALIZATION_SCALE_USD for token in self.wallet_observation_tokens
        }
        self.reserve_amount_references = dict(self.token_amount_references)
        self.gas_cost_reference_usd = max(
            float(np.max(self.gas_model.empirical_gas_costs_usd)), EPS
        )

    def normalization_metadata(self) -> Dict[str, Any]:
        return {
            "schema_version": NORMALIZATION_SCHEMA_VERSION,
            "observation_dimension": OBSERVATION_DIM,
            "wallet_tokens": list(self.wallet_observation_tokens),
            "wallet_references_usd": dict(sorted(self.wallet_value_references_usd.items())),
            "wallet_transform": "wallet_usd / (wallet_usd + 10000)",
            "reserve_references_token_units": dict(sorted(self.reserve_amount_references.items())),
            "reserve_reference": "symbol-keyed token-unit conservation upper bound across configured wallets and loaded pools",
            "gas_reference_usd": self.gas_cost_reference_usd,
            "gas_transform": "gas_cost_usd / (gas_cost_usd + fixed_empirical_scale_usd)",
            "remaining_time_reference_seconds": self.block_time_seconds,
            "reward_reference_usd": self.reward_reference_usd,
            "clipping_count": self.normalization_clipping_count,
            "clipping_by_feature": dict(sorted(self.normalization_clipping_by_feature.items())),
        }

    def path_policy_spec(self) -> Dict[str, Any]:
        wallet_indices_by_token = {
            token: index for index, token in enumerate(self.wallet_observation_tokens)
        }
        pool_indices = {pool.pool_id: index for index, pool in enumerate(self.pool_list)}
        wallet_indices: List[int] = []
        reserve_indices: List[Tuple[int, ...]] = []
        fee_multipliers: List[Tuple[float, ...]] = []
        reserve_amount_references: List[Tuple[float, ...]] = []
        start_token_prices_usd: List[float] = []
        path_labels: List[str] = []
        for token_a, token_b, token_c, pool_ids in self.triangular_paths:
            expected_tokens = (token_a, token_b, token_c, token_a)
            path_reserve_indices: List[int] = []
            path_fees: List[float] = []
            path_reserve_references: List[float] = []
            for leg, pool_id in enumerate(pool_ids):
                pool = self.pool_template[pool_id]
                token_in = expected_tokens[leg]
                token_out = expected_tokens[leg + 1]
                if pool.token0 == token_in and pool.token1 == token_out:
                    in_side, out_side = (0, 1)
                elif pool.token1 == token_in and pool.token0 == token_out:
                    in_side, out_side = (1, 0)
                else:
                    raise ValueError(
                        f"Path {self.format_path_tokens((token_a, token_b, token_c, pool_ids))} is inconsistent with pool {pool_id}."
                    )
                pool_offset = self.reserve_feature_start + 2 * pool_indices[pool_id]
                path_reserve_indices.extend((pool_offset + in_side, pool_offset + out_side))
                path_reserve_references.extend(
                    (
                        float(self.reserve_amount_references[token_in]),
                        float(self.reserve_amount_references[token_out]),
                    )
                )
                path_fees.append(1.0 - float(pool.fee))
            if token_a not in wallet_indices_by_token:
                raise ValueError(f"Path starts from unfunded token {token_a!r}.")
            wallet_indices.append(self.wallet_feature_start + wallet_indices_by_token[token_a])
            reserve_indices.append(tuple(path_reserve_indices))
            fee_multipliers.append(tuple(path_fees))
            reserve_amount_references.append(tuple(path_reserve_references))
            start_token_prices_usd.append(float(self.prices[token_a]))
            path_labels.append(
                f"{self.format_path_tokens((token_a, token_b, token_c, pool_ids))}|{'|'.join(pool_ids)}"
            )
        spec = PathPolicySpec(
            schema_version=PATH_POLICY_SPEC_VERSION,
            observation_dim=int(self.observation_space.shape[0]),
            wallet_indices=tuple(wallet_indices),
            reserve_indices=tuple(reserve_indices),
            fee_multipliers=tuple(fee_multipliers),
            gas_index=int(self.gas_feature_index),
            remaining_time_index=int(self.remaining_time_feature_index),
            path_labels=tuple(path_labels),
            reserve_amount_references=tuple(reserve_amount_references),
            start_token_prices_usd=tuple(start_token_prices_usd),
            wallet_value_reference_usd=WALLET_NORMALIZATION_SCALE_USD,
            gas_cost_reference_usd=float(self.gas_cost_reference_usd),
            economic_feature_spec_version=PATH_FEATURE_SPEC_VERSION,
        )
        return spec.to_dict()

    def _select_wallet(
        self, seed: Optional[int] = None, options: Optional[Mapping[str, Any]] = None
    ) -> Tuple[str, Dict[str, float]]:
        if self.fixed_wallet_index is not None:
            self.current_wallet_index = self.fixed_wallet_index
            return self.wallets[self.fixed_wallet_index]
        requested_index = None if options is None else options.get("wallet_index")
        if requested_index is not None:
            idx = int(requested_index)
            if idx < 0 or idx >= len(self.wallets):
                raise ValueError(
                    f"wallet_index must be in [0, {len(self.wallets) - 1}]; received {idx}."
                )
        elif seed is not None:
            idx = int(seed) % len(self.wallets)
        elif self.wallet_mode == "random":
            idx = int(self.rng.integers(0, len(self.wallets)))
        else:
            idx = self.reset_counter % len(self.wallets)
        self.current_wallet_index = idx
        return self.wallets[idx]

    def set_clock(self, clock: Callable[[], float]) -> None:
        self._clock = clock
        self._decision_clock_active = False
        self._decision_clock_started_at = 0.0

    def _record_phase(self, phase: str, started_at: float) -> float:
        elapsed = max(0.0, float(self._clock()) - float(started_at))
        self.cumulative_decision_time_seconds += elapsed
        self.decision_phase_seconds[phase] = self.decision_phase_seconds.get(phase, 0.0) + elapsed
        return elapsed

    def _record_common_stop_phase(self, phase: str, started_at: float) -> float:
        elapsed = max(0.0, float(self._clock()) - float(started_at))
        self.cumulative_decision_time_seconds += elapsed
        decision_phase = f"common_stop_check_{phase}"
        self.decision_phase_seconds[decision_phase] = (
            self.decision_phase_seconds.get(decision_phase, 0.0) + elapsed
        )
        self.cumulative_common_stop_check_time_seconds += elapsed
        self.common_stop_check_phase_seconds[phase] = (
            self.common_stop_check_phase_seconds.get(phase, 0.0) + elapsed
        )
        return elapsed

    def resume_decision_clock(self) -> None:
        if not self._decision_clock_active:
            self._decision_clock_started_at = float(self._clock())
            self._decision_clock_active = True

    def pause_decision_clock(self, discard_pending: bool = False) -> None:
        if not self._decision_clock_active:
            return
        if not discard_pending:
            self._record_phase("selection", self._decision_clock_started_at)
        self._decision_clock_active = False
        self._decision_clock_started_at = 0.0

    def _finish_selection_timing(self) -> None:
        if self._decision_clock_active:
            self._record_phase("selection", self._decision_clock_started_at)
            self._decision_clock_active = False
            self._decision_clock_started_at = 0.0

    def _remaining_time_fraction(self) -> float:
        remaining = max(0.0, self.block_time_seconds - self.cumulative_decision_time_seconds)
        return remaining / self.block_time_seconds

    def _timed_common_stop_check(self) -> Tuple[str, Optional[Dict[str, Any]]]:
        started = float(self._clock())
        result = self._exact_common_stop_check()
        self._record_common_stop_phase("global_stop_check", started)
        return result

    def _common_stop_state_key(self) -> Tuple[float, ...]:
        values: List[float] = [float(self.current_gas.gas_cost_usd)]
        values.extend((float(self.current_balance.get(token, 0.0)) for token in self.tokens))
        for pool in self.pool_list:
            state = self.current_pools[pool.pool_id]
            values.extend((float(state["reserve0"]), float(state["reserve1"])))
        return tuple(values)

    def _timed_common_stop_state_key(self) -> Tuple[float, ...]:
        started = float(self._clock())
        key = self._common_stop_state_key()
        self._record_common_stop_phase("state_key", started)
        return key

    def _timed_observation(
        self,
        balance: Optional[Mapping[str, float]] = None,
        pools: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> np.ndarray:
        started = float(self._clock())
        observation = self._get_observation(balance=balance, pools=pools)
        self._record_phase("observation", started)
        observation[self.remaining_time_feature_index] = np.float32(self._remaining_time_fraction())
        return observation

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if gym is not None:
            super().reset(seed=seed)
        if seed is not None:
            self.seed_value = int(seed)
            self.rng = np.random.default_rng(int(seed))
            self.gas_model.reseed(int(seed) + 7919)
            seed_action_space = getattr(self.action_space, "seed", None)
            if callable(seed_action_space):
                seed_action_space(int(seed))
        self.refresh_normalization_references()
        wallet_name, wallet = self._select_wallet(seed=seed, options=options)
        self.reset_counter += 1
        self.wallet_name = wallet_name
        self.initial_balance = {token: float(wallet.get(token, 0.0)) for token in self.tokens}
        self.current_balance = copy.deepcopy(self.initial_balance)
        self.current_pools = self._pool_state_from_template()
        self.current_gas = self.gas_model.sample()
        self.initial_portfolio_usd = self._portfolio_value_usd(self.current_balance)
        self.total_profit_usd = 0.0
        self.total_gross_usd = 0.0
        self.total_gas_usd = 0.0
        self.successful_arbitrages = 0
        self.failed_actions = 0
        self.episode_steps = 0
        self.last_profit_usd = 0.0
        self.cumulative_decision_time_seconds = 0.0
        self.decision_phase_seconds = {}
        self.cumulative_common_stop_check_time_seconds = 0.0
        self.common_stop_check_phase_seconds = {}
        self._decision_clock_active = False
        self._decision_clock_started_at = 0.0
        self.normalization_clipping_count = 0
        self.normalization_clipping_by_feature = {}
        stop_reason, best = self._timed_common_stop_check()
        self._cached_common_stop_reason = str(stop_reason)
        self._cached_common_stop_best = copy.deepcopy(best)
        self._cached_common_stop_state_key = self._timed_common_stop_state_key()
        self.last_info = {
            "protocol_id": self.protocol_id,
            "dataset_file": str(self.pool_dataset_file),
            "dataset_type": self.dataset_mode,
            "wallet_file": self.wallet_name,
            "wallet_index": int(self.current_wallet_index),
            "wallet_start_tokens": list(self.wallet_start_tokens),
            "gas_cost_usd": self.current_gas.gas_cost_usd,
            "gas_price_gwei": self.current_gas.gas_price_gwei,
            "gas_used": self.current_gas.gas_used,
            "gas_source_row_index": self.current_gas.source_row_index,
            "gas_resample_mode": "episode",
            "gas_accounting_note": "Gas is included in after-gas profit and is paid by an account separate from trading-token balances.",
            "reward_rules": c1_reward_rules(
                self.c1_rejection_reward, self.c1_counterfactual_reward
            ),
            "block_time_seconds": self.block_time_seconds,
            "timing_semantics": "C6 includes observation construction, method selection, exact common stopping checks, screening, CPMM simulation, tentative state construction, and execution bookkeeping.",
            "action_mode": self.action_mode,
            "action_space_form": self.action_space_form,
            "action_space_note": self.action_space_note,
            "path_feature_mode": self.path_feature_mode,
            "action_head_architecture": (
                PPO_ACTION_HEAD_ARCHITECTURE
                if self.action_mode == ACTION_MODE_HYBRID_PPO
                else "continuous_relaxed_actor"
            ),
            "normalization": self.normalization_metadata(),
            "initial_stop_reason": stop_reason,
            "initial_best_net_profit_usd": float(best.get("net_profit_usd", 0.0)) if best else 0.0,
        }
        obs = self._timed_observation()
        self.last_info["normalization"] = self.normalization_metadata()
        self.last_info["cumulative_decision_time_seconds"] = self.cumulative_decision_time_seconds
        self.last_info["decision_phase_seconds"] = dict(self.decision_phase_seconds)
        self.last_info["cumulative_common_stop_check_time_seconds"] = (
            self.cumulative_common_stop_check_time_seconds
        )
        self.last_info["common_stop_check_phase_seconds"] = dict(
            self.common_stop_check_phase_seconds
        )
        info = dict(self.last_info)
        if GYM_API == "gymnasium":
            return (obs, info)
        return obs

    def _portfolio_value_usd(self, balance: Mapping[str, float]) -> float:
        return float(
            sum((float(amount) * self.prices.get(token, 0.0) for token, amount in balance.items()))
        )

    def _reserves_for_direction(
        self, pool_state: Mapping[str, Any], token_in: str
    ) -> Tuple[float, float, bool]:
        if pool_state["token0"] == token_in:
            return (float(pool_state["reserve0"]), float(pool_state["reserve1"]), True)
        if pool_state["token1"] == token_in:
            return (float(pool_state["reserve1"]), float(pool_state["reserve0"]), False)
        return (0.0, 0.0, True)

    @staticmethod
    def _cpmm_quote(amount_in: float, reserve_in: float, reserve_out: float, fee: float) -> float:
        if amount_in <= 0 or reserve_in <= 0 or reserve_out <= 0:
            return 0.0
        alpha_amount = amount_in * (1.0 - fee)
        amount_out = alpha_amount * reserve_out / (reserve_in + alpha_amount)
        if amount_out <= 0 or amount_out >= reserve_out:
            return 0.0
        return float(amount_out)

    def _execute_swap_on_state(
        self, pool_id: str, token_in: str, amount_in: float, state: Dict[str, Dict[str, float]]
    ) -> Tuple[float, str, bool]:
        if pool_id not in state:
            return (0.0, "", False)
        pool_state = state[pool_id]
        reserve_in, reserve_out, token_in_is_0 = self._reserves_for_direction(pool_state, token_in)
        amount_out = self._cpmm_quote(amount_in, reserve_in, reserve_out, float(pool_state["fee"]))
        if amount_out <= 0:
            return (0.0, "", False)
        if token_in_is_0:
            pool_state["reserve0"] += amount_in
            pool_state["reserve1"] -= amount_out
            token_out = str(pool_state["token1"])
        else:
            pool_state["reserve1"] += amount_in
            pool_state["reserve0"] -= amount_out
            token_out = str(pool_state["token0"])
        if pool_state["reserve0"] <= EPS or pool_state["reserve1"] <= EPS:
            return (0.0, "", False)
        return (amount_out, token_out, True)

    def _path_is_physically_feasible(self, path_index: int) -> bool:
        path = self.triangular_paths[int(path_index)]
        if float(self.current_balance.get(path[0], 0.0)) <= EPS:
            return False
        expected = [path[0], path[1], path[2], path[0]]
        for leg, pool_id in enumerate(path[3]):
            pool = self.current_pools.get(pool_id)
            if pool is None:
                return False
            reserve_in, reserve_out, _direction = self._reserves_for_direction(pool, expected[leg])
            if reserve_in <= EPS or reserve_out <= EPS:
                return False
            token_out = pool["token1"] if pool["token0"] == expected[leg] else pool["token0"]
            if token_out != expected[leg + 1]:
                return False
        return True

    def _simulate_path(
        self,
        path: Tuple[str, str, str, Tuple[str, str, str]],
        amount_in: float,
        mutate: bool = False,
    ) -> Dict[str, Any]:
        token_a, token_b, token_c, pool_ids = path
        expected = [token_a, token_b, token_c, token_a]
        affected_before = {
            pool_id: copy.deepcopy(self.current_pools[pool_id]) for pool_id in pool_ids
        }
        local_state = copy.deepcopy(affected_before)
        amount = float(amount_in)
        trace: List[Tuple[str, str, float, float, str]] = []
        for leg, pool_id in enumerate(pool_ids):
            amount_out, token_out, ok = self._execute_swap_on_state(
                pool_id, expected[leg], amount, local_state
            )
            if not ok or token_out != expected[leg + 1]:
                return {
                    "success": False,
                    "reason": "C3_reserve_or_path_continuity",
                    "constraint": "C3_C4_C5",
                    "gross_profit_usd": 0.0,
                    "net_profit_usd": -float(self.current_gas.gas_cost_usd),
                    "gas_cost_usd": float(self.current_gas.gas_cost_usd),
                    "trace": trace,
                    "affected_pools_before": affected_before,
                    "affected_pools_after": affected_before,
                }
            trace.append((expected[leg], token_out, amount, amount_out, pool_id))
            amount = amount_out
        gross_profit_usd = (amount - amount_in) * float(self.prices.get(token_a, 0.0))
        net_profit_usd = gross_profit_usd - float(self.current_gas.gas_cost_usd)
        success = bool(net_profit_usd > self.min_profit_usd)
        if mutate and success:
            for pool_id in pool_ids:
                self.current_pools[pool_id] = copy.deepcopy(local_state[pool_id])
        return {
            "success": success,
            "reason": "ok" if success else "C1_not_profitable_after_gas",
            "constraint": "" if success else "C1",
            "final_amount": float(amount),
            "token_gain": float(amount - amount_in),
            "gross_profit_usd": float(gross_profit_usd),
            "net_profit_usd": float(net_profit_usd),
            "gas_cost_usd": float(self.current_gas.gas_cost_usd),
            "trace": trace,
            "pool_state": local_state,
            "affected_pools_before": affected_before,
            "affected_pools_after": copy.deepcopy(local_state),
        }

    def _path_composition_coefficients(
        self, path_index: int
    ) -> Optional[Tuple[float, float, float]]:
        path = self.triangular_paths[int(path_index)]
        expected = [path[0], path[1], path[2], path[0]]
        composed: Optional[Tuple[float, float, float]] = None
        for leg, pool_id in enumerate(path[3]):
            pool = self.current_pools.get(pool_id)
            if pool is None:
                return None
            reserve_in, reserve_out, _direction = self._reserves_for_direction(pool, expected[leg])
            if reserve_in <= EPS or reserve_out <= EPS:
                return None
            alpha = 1.0 - float(pool["fee"])
            swap = (alpha * reserve_out, reserve_in, alpha)
            if composed is None:
                composed = swap
            else:
                a_prev, b_prev, c_prev = composed
                a_next, b_next, c_next = swap
                composed = (a_next * a_prev, b_next * b_prev, b_next * c_prev + c_next * a_prev)
        return composed

    def _path_continuous_optimum(self, path_index: int) -> Optional[Dict[str, Any]]:
        if not self._path_is_physically_feasible(path_index):
            return None
        path = self.triangular_paths[int(path_index)]
        wallet_bound = float(self.current_balance.get(path[0], 0.0)) * self.amount_max
        lower_bound = float(self.current_balance.get(path[0], 0.0)) * self.amount_min
        coeff = self._path_composition_coefficients(path_index)
        if coeff is None or wallet_bound <= EPS:
            return None
        a_coeff, b_coeff, c_coeff = coeff
        if a_coeff <= b_coeff + EPS or c_coeff <= EPS:
            return None
        stationary = (math.sqrt(a_coeff * b_coeff) - b_coeff) / c_coeff
        candidates = {
            float(np.clip(stationary, lower_bound, wallet_bound)),
            float(lower_bound),
            float(wallet_bound),
        }
        best: Optional[Dict[str, Any]] = None
        for amount_in in sorted(candidates):
            if amount_in <= EPS:
                continue
            result = self._simulate_path(path, amount_in, mutate=False)
            if best is None or float(result.get("net_profit_usd", -float("inf"))) > float(
                best["net_profit_usd"]
            ):
                best = {
                    **result,
                    "path_index": int(path_index),
                    "amount_in": float(amount_in),
                    "amount_fraction": float(
                        amount_in / max(self.current_balance.get(path[0], 0.0), EPS)
                    ),
                    "composition_A": float(a_coeff),
                    "composition_B": float(b_coeff),
                    "composition_C": float(c_coeff),
                }
        if best is None or float(best["net_profit_usd"]) <= self.min_profit_usd:
            return None
        return best

    def _best_profitable_opportunity(self) -> Optional[Dict[str, Any]]:
        best: Optional[Dict[str, Any]] = None
        for path_index in range(len(self.triangular_paths)):
            candidate = self._path_continuous_optimum(path_index)
            if candidate is None:
                continue
            if best is None or float(candidate["net_profit_usd"]) > float(best["net_profit_usd"]):
                best = candidate
        return best

    def _exact_common_stop_check(self) -> Tuple[str, Optional[Dict[str, Any]]]:
        funded = any(
            (float(self.current_balance.get(path[0], 0.0)) > EPS for path in self.triangular_paths)
        )
        if not funded:
            return ("C2_no_funded_start_token", None)
        physical = any(
            (self._path_is_physically_feasible(i) for i in range(len(self.triangular_paths)))
        )
        if not physical:
            return ("C3_no_reserve_feasible_path", None)
        best = self._best_profitable_opportunity()
        if best is None:
            return ("C1_no_profitable_feasible_transaction", None)
        return ("", best)

    def _normalize_value(self, value: float, reference: float, feature_name: str) -> float:
        raw = float(value) / max(float(reference), EPS)
        if raw < -1e-09 or raw > 1.0 + 1e-09:
            self.normalization_clipping_count += 1
            self.normalization_clipping_by_feature[feature_name] = (
                self.normalization_clipping_by_feature.get(feature_name, 0) + 1
            )
            raise RuntimeError(
                f"Normalization reference exceeded for {feature_name}: value={value}, reference={reference}, ratio={raw}."
            )
        return float(np.clip(raw, 0.0, 1.0))

    @staticmethod
    def _bounded_positive_value(value: float, scale: float, feature_name: str) -> float:
        value = float(value)
        scale = float(scale)
        if not math.isfinite(value) or value < 0.0:
            raise RuntimeError(f"{feature_name} must be finite and nonnegative; received {value}.")
        if not math.isfinite(scale) or scale <= 0.0:
            raise RuntimeError(
                f"{feature_name} scale must be finite and positive; received {scale}."
            )
        return value / (value + scale)

    def _get_observation(
        self,
        balance: Optional[Mapping[str, float]] = None,
        pools: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> np.ndarray:
        balance_state = self.current_balance if balance is None else balance
        pool_state = self.current_pools if pools is None else pools
        obs: List[float] = []
        for token in self.wallet_observation_tokens:
            obs.append(
                self._bounded_positive_value(
                    float(balance_state.get(token, 0.0)) * float(self.prices.get(token, 0.0)),
                    self.wallet_value_references_usd[token],
                    f"wallet:{token}",
                )
            )
        for pool in self.pool_list:
            state = pool_state[pool.pool_id]
            obs.append(
                self._normalize_value(
                    float(state["reserve0"]),
                    self.reserve_amount_references[pool.token0],
                    f"reserve:{pool.pool_id}:0",
                )
            )
            obs.append(
                self._normalize_value(
                    float(state["reserve1"]),
                    self.reserve_amount_references[pool.token1],
                    f"reserve:{pool.pool_id}:1",
                )
            )
        obs.extend(
            [
                self._bounded_positive_value(
                    self.current_gas.gas_cost_usd, self.gas_cost_reference_usd, "gas_cost"
                ),
                self._remaining_time_fraction(),
            ]
        )
        array = np.asarray(obs, dtype=np.float32)
        if array.shape != self.observation_space.shape:
            raise RuntimeError(
                f"Observation shape {array.shape} does not match {self.observation_space.shape}."
            )
        return array

    def step(self, action: np.ndarray):
        self._finish_selection_timing()
        bookkeeping_clock = float(self._clock())
        bookkeeping_cumulative = self.cumulative_decision_time_seconds

        def account_bookkeeping() -> None:
            nonlocal bookkeeping_clock, bookkeeping_cumulative
            now = float(self._clock())
            accounted = self.cumulative_decision_time_seconds - bookkeeping_cumulative
            actual = max(0.0, now - bookkeeping_clock)
            unaccounted = max(0.0, actual - accounted)
            if unaccounted > 0.0:
                self.cumulative_decision_time_seconds += unaccounted
                self.decision_phase_seconds["screening_and_bookkeeping"] = (
                    self.decision_phase_seconds.get("screening_and_bookkeeping", 0.0) + unaccounted
                )
            bookkeeping_clock = now
            bookkeeping_cumulative = self.cumulative_decision_time_seconds

        self.episode_steps += 1
        screening_started = float(self._clock())
        path_index, amount_fraction, action_audit = self._decode_action(action)
        path = self.triangular_paths[path_index]
        start_token = path[0]
        start_balance = float(self.current_balance.get(start_token, 0.0))
        amount_in = amount_fraction * start_balance
        pre_balance = copy.deepcopy(self.current_balance)
        self._record_phase("screening", screening_started)
        current_state_key = self._timed_common_stop_state_key()
        if current_state_key != self._cached_common_stop_state_key:
            refreshed_reason, refreshed_best = self._timed_common_stop_check()
            self._cached_common_stop_reason = str(refreshed_reason)
            self._cached_common_stop_best = copy.deepcopy(refreshed_best)
            self._cached_common_stop_state_key = current_state_key
        pre_stop_reason = self._cached_common_stop_reason
        info: Dict[str, Any] = {
            "protocol_id": self.protocol_id,
            "path_index": path_index,
            "path": path,
            "path_tokens": self.format_path_tokens(path),
            **action_audit,
            "action_mode": self.action_mode,
            "action_space_form": self.action_space_form,
            "action_space_note": self.action_space_note,
            "path_feature_mode": self.path_feature_mode,
            "amount_fraction": amount_fraction,
            "amount_in": amount_in,
            "wallet_file": self.wallet_name,
            "gas_cost_usd": float(self.current_gas.gas_cost_usd),
            "gas_price_gwei": float(self.current_gas.gas_price_gwei),
            "gas_used": float(self.current_gas.gas_used),
            "gas_source_row_index": int(self.current_gas.source_row_index),
            "gas_resample_mode": "episode",
            "block_time_seconds": self.block_time_seconds,
            "submitted_onchain": False,
            "simulated_execution": False,
            "gas_charged_usd": 0.0,
            "execution_semantics_note": c1_execution_semantics_note(
                self.c1_rejection_reward, self.c1_counterfactual_reward
            ),
            "reward_rules": c1_reward_rules(
                self.c1_rejection_reward, self.c1_counterfactual_reward
            ),
            "wallet_before": pre_balance,
            "cumulative_decision_time_seconds": self.cumulative_decision_time_seconds,
            "cumulative_common_stop_check_time_seconds": self.cumulative_common_stop_check_time_seconds,
        }
        reward = 0.0
        staged_balance = pre_balance
        staged_pool_updates: Dict[str, Dict[str, float]] = {}
        accepted_result: Optional[Dict[str, Any]] = None
        rejection_constraint = ""
        next_state_key: Optional[Tuple[float, ...]] = None
        if pre_stop_reason:
            info.update(
                {
                    "success": False,
                    "execution_status": "not_executed",
                    "constraint": pre_stop_reason.split("_", 1)[0],
                    "reason": pre_stop_reason,
                    "gross_profit_usd": 0.0,
                    "net_profit_usd": 0.0,
                    "counterfactual_net_profit_usd_if_submitted": 0.0,
                }
            )
        elif (
            not self._path_is_physically_feasible(path_index)
            or amount_in <= EPS
            or amount_in > start_balance + EPS
        ):
            constraint = (
                "C2"
                if amount_in <= EPS or start_balance <= EPS or amount_in > start_balance + EPS
                else "C3"
            )
            rejection_constraint = constraint
            info.update(
                {
                    "success": False,
                    "execution_status": "screened",
                    "constraint": constraint,
                    "reason": f"{constraint}_selected_action_not_physically_feasible",
                    "gross_profit_usd": 0.0,
                    "net_profit_usd": 0.0,
                    "counterfactual_net_profit_usd_if_submitted": 0.0,
                }
            )
        else:
            simulation_started = float(self._clock())
            result = self._simulate_path(path, amount_in, mutate=False)
            self._record_phase("cpmm_simulation", simulation_started)
            if bool(result.get("success", False)):
                transition_started = float(self._clock())
                accepted_result = result
                staged_balance = copy.deepcopy(pre_balance)
                for pool_id, state in result["pool_state"].items():
                    staged_pool_updates[pool_id] = copy.deepcopy(state)
                final_amount = float(result["final_amount"])
                staged_balance[start_token] = start_balance - amount_in + final_amount
                self._record_phase("tentative_state_transition", transition_started)
                info.update(
                    {
                        **result,
                        "success": True,
                        "execution_status": "tentative",
                        "constraint": "",
                        "submitted_onchain": False,
                        "gas_charged_usd": 0.0,
                    }
                )
            else:
                rejection_constraint = str(result.get("constraint", "C1"))
                info.update(
                    {
                        "success": False,
                        "execution_status": "screened",
                        "constraint": str(result.get("constraint", "C1")),
                        "reason": str(result.get("reason", "C1_not_profitable_after_gas")),
                        "gross_profit_usd": 0.0,
                        "net_profit_usd": 0.0,
                        "counterfactual_gross_profit_usd": float(
                            result.get("gross_profit_usd", 0.0)
                        ),
                        "counterfactual_net_profit_usd_if_submitted": float(
                            result.get("net_profit_usd", 0.0)
                        ),
                        "affected_pools_before": result.get("affected_pools_before", {}),
                        "affected_pools_after": result.get("affected_pools_before", {}),
                    }
                )
        account_bookkeeping()
        c6_expired_before_commit = bool(
            self.cumulative_decision_time_seconds > self.block_time_seconds
        )
        c6_expired_after_commit = False
        stop_reason = ""
        next_best: Optional[Dict[str, Any]] = None
        if c6_expired_before_commit:
            stop_reason = "C6_wall_clock_budget_exhausted"
            accepted_result = None
            reward = self.c6_expiry_reward
            info.update(
                {
                    "success": False,
                    "execution_status": "not_executed",
                    "constraint": "C6",
                    "reason": stop_reason,
                    "submitted_onchain": False,
                    "gas_charged_usd": 0.0,
                    "gross_profit_usd": 0.0,
                    "net_profit_usd": 0.0,
                    "affected_pools_after": info.get("affected_pools_before", {}),
                }
            )
            obs = self._timed_observation(balance=pre_balance)
        elif accepted_result is not None:
            self.current_balance = staged_balance
            for pool_id, state in staged_pool_updates.items():
                self.current_pools[pool_id] = copy.deepcopy(state)
            net_profit = float(accepted_result["net_profit_usd"])
            gross_profit = float(accepted_result["gross_profit_usd"])
            self.total_profit_usd += net_profit
            self.total_gross_usd += gross_profit
            self.total_gas_usd += float(self.current_gas.gas_cost_usd)
            self.successful_arbitrages += 1
            self.last_profit_usd = net_profit
            reward = net_profit / self.reward_reference_usd
            info.update(
                {
                    "success": True,
                    "execution_status": "executed",
                    "submitted_onchain": False,
                    "simulated_execution": True,
                    "gas_charged_usd": float(self.current_gas.gas_cost_usd),
                }
            )
            stop_reason, next_best = self._timed_common_stop_check()
            next_state_key = self._timed_common_stop_state_key()
            obs = self._timed_observation()
            account_bookkeeping()
            c6_expired_after_commit = bool(
                self.cumulative_decision_time_seconds > self.block_time_seconds
            )
            if c6_expired_after_commit:
                info["post_commit_common_stop_reason"] = stop_reason
                stop_reason = "C6_wall_clock_budget_exhausted"
            self._cached_common_stop_reason = str(stop_reason)
            self._cached_common_stop_best = copy.deepcopy(next_best)
            self._cached_common_stop_state_key = next_state_key
        elif pre_stop_reason:
            stop_reason = pre_stop_reason
            obs = self._timed_observation()
        elif rejection_constraint:
            self.failed_actions += 1
            self.last_profit_usd = 0.0
            if rejection_constraint == "C1":
                if self.c1_counterfactual_reward:
                    counterfactual_net_profit = float(
                        info["counterfactual_net_profit_usd_if_submitted"]
                    )
                    if not math.isfinite(counterfactual_net_profit):
                        raise FloatingPointError(
                            "Non-finite selected-action C1 counterfactual net profit."
                        )
                    reward = float(
                        np.clip(
                            counterfactual_net_profit / self.reward_reference_usd,
                            PHYSICAL_REJECTION_REWARD,
                            0.0,
                        )
                    )
                else:
                    reward = self.c1_rejection_reward
            else:
                reward = self.physical_rejection_reward
            next_best = copy.deepcopy(self._cached_common_stop_best)
            obs = self._timed_observation()
            account_bookkeeping()
            if self.cumulative_decision_time_seconds > self.block_time_seconds:
                stop_reason = "C6_wall_clock_budget_exhausted"
                c6_expired_before_commit = True
                reward = self.c6_expiry_reward
                info.update(
                    {"execution_status": "not_executed", "constraint": "C6", "reason": stop_reason}
                )
        else:
            obs = self._timed_observation()
        obs[self.remaining_time_feature_index] = np.float32(self._remaining_time_fraction())
        terminated = bool(stop_reason)
        truncated = False
        portfolio_value = self._portfolio_value_usd(self.current_balance)
        info.update(
            {
                "wallet_after": copy.deepcopy(self.current_balance),
                "portfolio_value_usd": portfolio_value,
                "initial_portfolio_usd": self.initial_portfolio_usd,
                "total_profit_usd": self.total_profit_usd,
                "total_gross_usd": self.total_gross_usd,
                "total_gas_usd": self.total_gas_usd,
                "successful_arbitrages": self.successful_arbitrages,
                "failed_actions": self.failed_actions,
                "episode_steps": self.episode_steps,
                "stop_reason": stop_reason,
                "next_best_net_profit_usd": (
                    float(next_best.get("net_profit_usd", 0.0)) if next_best else 0.0
                ),
                "cumulative_decision_time_seconds": self.cumulative_decision_time_seconds,
                "decision_phase_seconds": dict(sorted(self.decision_phase_seconds.items())),
                "cumulative_common_stop_check_time_seconds": self.cumulative_common_stop_check_time_seconds,
                "common_stop_check_phase_seconds": dict(
                    sorted(self.common_stop_check_phase_seconds.items())
                ),
                "timing_semantics": "C6 includes method inference/selection, screening, CPMM simulation, observation construction, exact common stopping checks, tentative state construction, and execution bookkeeping.",
                "c6_wall_clock_feasible": not c6_expired_before_commit,
                "c6_commit_within_budget": bool(
                    info.get("execution_status") == "executed" and (not c6_expired_before_commit)
                ),
                "c6_episode_deadline_reached": bool(
                    c6_expired_before_commit or c6_expired_after_commit
                ),
                "c6_expired_before_commit": c6_expired_before_commit,
                "c6_expired_after_commit": c6_expired_after_commit,
                "reward_reference_usd": self.reward_reference_usd,
                "normalization": self.normalization_metadata(),
            }
        )
        self.last_info = info
        if GYM_API == "gymnasium":
            return (obs, float(reward), terminated, truncated, info)
        return (obs, float(reward), bool(terminated or truncated), info)


def reset_env_compat(env: Any, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    try:
        out = env.reset(seed=seed)
    except TypeError:
        out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        return (out[0], out[1])
    return (out, {})


def step_env_compat(env: Any, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
    out = env.step(action)
    if isinstance(out, tuple) and len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return (obs, float(reward), bool(terminated or truncated), info)
    obs, reward, done, info = out
    return (obs, float(reward), bool(done), info)


def make_env(
    args: argparse.Namespace,
    seed: int,
    wallet_mode: str = "cycle",
    algo: Optional[str] = None,
    fixed_wallet_usd: Optional[int] = None,
) -> PPR01TriangularArbitrageEnv:
    action_mode = resolve_action_mode(args, algo)
    amount_grid = parse_amount_grid_text(
        getattr(args, "amount_grid", DEFAULT_AMOUNT_GRID_TEXT), args.amount_min, args.amount_max
    )
    return PPR01TriangularArbitrageEnv(
        dataset_dir=Path(args.dataset_dir),
        seed=seed,
        amount_min=args.amount_min,
        amount_max=args.amount_max,
        gas_multiplier=args.gas_multiplier,
        max_paths=args.max_paths,
        wallet_mode=wallet_mode,
        block_time_seconds=float(getattr(args, "block_time_seconds", 12.0)),
        action_mode=action_mode,
        amount_grid=amount_grid,
        reward_reference_usd=float(
            getattr(args, "reward_reference_usd", REWARD_REFERENCE_USD_DEFAULT)
        ),
        protocol_id=str(getattr(args, "protocol_id", PROTOCOL_ID)),
        fixed_wallet_usd=fixed_wallet_usd,
        c1_rejection_reward=resolved_c1_rejection_reward(args),
        path_feature_mode=str(getattr(args, "ppo_path_feature_mode", FIXED_FEATURE_MODE)),
        c1_counterfactual_reward=resolved_c1_counterfactual_reward(args),
    )


def path_policy_spec_from_env(env: Any) -> Dict[str, Any]:
    candidate = env
    visited: set = set()
    while candidate is not None and id(candidate) not in visited:
        visited.add(id(candidate))
        method = getattr(candidate, "path_policy_spec", None)
        if callable(method):
            return dict(method())
        candidate = getattr(candidate, "env", None)
    env_method = getattr(env, "env_method", None)
    if callable(env_method):
        values = env_method("path_policy_spec", indices=[0])
        if values:
            return dict(values[0])
    nested = getattr(env, "envs", None)
    if nested:
        return path_policy_spec_from_env(nested[0])
    raise TypeError("Could not obtain PathPolicySpec from the supplied environment.")


def environment_attribute(env: Any, name: str) -> Any:
    candidate = env
    visited: set = set()
    while candidate is not None and id(candidate) not in visited:
        visited.add(id(candidate))
        if hasattr(candidate, name):
            return getattr(candidate, name)
        candidate = getattr(candidate, "env", None)
    get_attr = getattr(env, "get_attr", None)
    if callable(get_attr):
        values = get_attr(name, indices=[0])
        if values:
            return values[0]
    nested = getattr(env, "envs", None)
    if nested:
        return environment_attribute(nested[0], name)
    raise AttributeError(f"Could not obtain environment attribute {name!r}.")


def set_environment_attribute(env: Any, name: str, value: Any) -> None:
    set_attr = getattr(env, "set_attr", None)
    if callable(set_attr):
        set_attr(name, value, indices=[0])
        return
    candidate = env
    visited: set = set()
    while candidate is not None and id(candidate) not in visited:
        visited.add(id(candidate))
        if name in getattr(candidate, "__dict__", {}):
            setattr(candidate, name, value)
            return
        candidate = getattr(candidate, "env", None)
    nested = getattr(env, "envs", None)
    if nested:
        set_environment_attribute(nested[0], name, value)
        return
    raise AttributeError(f"Could not set environment attribute {name!r}.")


def configure_model_for_env(model: Any, env: Any, algo: str) -> Optional[Dict[str, Any]]:
    if str(algo).upper() != "PPO":
        return None
    policy = getattr(model, "policy", None)
    configure_spec = getattr(policy, "configure_path_policy_spec", None)
    if policy is None or not callable(configure_spec):
        raise TypeError("PPO model does not use the v2.09 variable-path hybrid policy.")
    policy_feature_mode = canonical_ppo_path_feature_mode(
        getattr(policy, "path_feature_mode", FIXED_FEATURE_MODE)
    )
    policy_amount_treatment = canonical_ppo_amount_treatment(
        getattr(policy, "amount_treatment", FIXED_AMOUNT_TREATMENT)
    )
    policy_wallet_curriculum = str(getattr(policy, "wallet_curriculum", "fixed10k")).strip().lower()
    if policy_wallet_curriculum not in PPO_WALLET_CURRICULA:
        raise RuntimeError("Saved PPO has an invalid wallet curriculum.")
    set_environment_attribute(env, "path_feature_mode", policy_feature_mode)
    spec_mapping = path_policy_spec_from_env(env)
    target_action_space = getattr(env, "action_space", None)
    if target_action_space is None:
        raise TypeError("Target environment does not expose an action space.")
    model.action_space = target_action_space
    policy.action_space = target_action_space
    amount_min = float(environment_attribute(env, "amount_min"))
    amount_max = float(environment_attribute(env, "amount_max"))
    policy.amount_min = amount_min
    policy.amount_max = amount_max
    policy.action_dist.set_bounds(amount_min, amount_max)
    configure_spec(spec_mapping)
    if isinstance(getattr(model, "policy_kwargs", None), dict):
        rebound_policy_kwargs = {
            "path_policy_spec": copy.deepcopy(spec_mapping),
            "amount_min": amount_min,
            "amount_max": amount_max,
        }
        if hasattr(policy, "path_feature_mode"):
            rebound_policy_kwargs["path_feature_mode"] = policy_feature_mode
        rebound_policy_kwargs["amount_treatment"] = policy_amount_treatment
        rebound_policy_kwargs["wallet_curriculum"] = policy_wallet_curriculum
        model.policy_kwargs.update(rebound_policy_kwargs)
    return {
        "path_count": int(policy.path_count),
        "path_feature_mode": policy_feature_mode,
        "amount_treatment": policy_amount_treatment,
        "wallet_curriculum": policy_wallet_curriculum,
        "path_feature_dimension": ppo_path_feature_dimension(policy_feature_mode),
    }


# The frozen best_model.zip was serialised by the original pipeline and references its
# classes as ``train.VariablePathHybridPPOPolicy`` / ``train.CategoricalContinuousActionSpace``.
# Alias this module under the legacy name so stable_baselines3 PPO.load() can unpickle it.
sys.modules.setdefault("train", sys.modules[__name__])
