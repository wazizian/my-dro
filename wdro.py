"""Regularized WDRO dual objectives and ULA importance proposals."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import chex
import equinox as eqx
import jax
import jax.numpy as jnp

Array = jax.Array


@chex.dataclass
class RobustConfig:
    """Configure a WDRO dual and proposal; inputs are scalar hyperparameters."""

    enabled: bool = False
    objective: Literal["entropic", "joint_cvar", "per_sample_cvar", "relaxed_kl_cvar"] = "entropic"
    rho: float = 0.1
    kl_epsilon: float = 0.1
    tau: float = 0.0
    alpha: float = 0.95
    proposal_samples: int = 32
    proposal_scale: float = 0.1
    proposal_family: Literal["gaussian", "laplace"] = "gaussian"
    ula_steps: int = 0
    ula_step_size: float = 0.01
    ula_gradient_clip: float = 10.0
    initial_lambda: float = 1.0
    initial_eta: float = 0.0
    fix_lambda: bool = False
    cost_power: Literal[1, 2] = 2
    tail_power: float = 1.0
    exact_eta: bool = False
    ula_temper: bool = False
    anchor_mix_prob: float = 0.0


class RobustState(eqx.Module):
    """Store dual variables and fixed trajectory standardization statistics."""

    raw_lambda: Array
    eta: Array
    location: Array
    scale: Array


def gaussian_log_density(value: Array, mean: Array, variance: Array) -> Array:
    """Evaluate isotropic Gaussian log densities; inputs share shape (..., dimension), output is (...)."""

    chex.assert_equal_shape((value, mean))
    dimension = value.shape[-1]
    chex.assert_shape(variance, ())
    difference = value - mean
    chex.assert_shape(difference, value.shape)
    squared_distance = jnp.sum(jnp.square(difference), axis=-1)
    chex.assert_shape(squared_distance, value.shape[:-1])
    normalization = dimension * jnp.log(2.0 * jnp.pi * variance)
    chex.assert_shape(normalization, ())
    log_density = -0.5 * (squared_distance / variance + normalization)
    chex.assert_shape(log_density, value.shape[:-1])
    return log_density


def laplace_log_density(value: Array, mean: Array, scale: Array) -> Array:
    """Evaluate isotropic Laplace log densities; inputs share shape (..., dimension), output is (...)."""

    chex.assert_equal_shape((value, mean))
    dimension = value.shape[-1]
    chex.assert_shape(scale, ())
    difference = value - mean
    chex.assert_shape(difference, value.shape)
    absolute_distance = jnp.sum(jnp.abs(difference), axis=-1)
    normalization = dimension * jnp.log(2.0 * scale)
    chex.assert_shape(normalization, ())
    log_density = -(absolute_distance / scale + normalization)
    chex.assert_shape(log_density, value.shape[:-1])
    return log_density


def quadratic_transport_cost(
    samples: Array, anchors: Array, reference_scale: Array, cost_power: int = 2
) -> Array:
    """Compute standardized transport cost; inputs are proposed/base coordinates/scale/power (1=norm, 2=half squared norm), output is (batch, proposals)."""

    batch, n_proposals, dimension = samples.shape
    chex.assert_shape(anchors, (batch, dimension))
    chex.assert_shape(reference_scale, ())
    differences = (samples - anchors[:, None, :]) / reference_scale
    chex.assert_shape(differences, (batch, n_proposals, dimension))
    squared_distance = jnp.sum(jnp.square(differences), axis=2)
    chex.assert_shape(squared_distance, (batch, n_proposals))
    if cost_power == 1:
        costs = jnp.sqrt(jnp.maximum(squared_distance, 1e-12))
    else:
        costs = 0.5 * squared_distance
    chex.assert_shape(costs, (batch, n_proposals))
    return costs


def sample_langevin_proposal(
    key: Array,
    anchors: Array,
    log_target: Callable[[Array, Array], Array],
    config: RobustConfig,
) -> tuple[Array, Array]:
    """Sample a reference/ULA proposal; inputs are key, anchors, target, config and outputs samples/log weights."""

    batch, dimension = anchors.shape
    n_proposals = config.proposal_samples
    n_steps = config.ula_steps
    if n_steps < 0:
        raise ValueError("ula_steps must be nonnegative")
    if (config.proposal_scale <= 0.0 or config.ula_step_size <= 0.0
            or config.ula_gradient_clip <= 0.0):
        raise ValueError("proposal scale, ULA step size, and gradient clip must be positive")
    if not 0.0 <= config.anchor_mix_prob <= 1.0:
        raise ValueError("anchor_mix_prob must be in [0, 1]")
    if config.anchor_mix_prob > 0.0 and n_steps > 0:
        raise ValueError("anchor_mix_prob is only supported with ula_steps == 0")
    n_key_slots = n_steps + 2 if config.anchor_mix_prob > 0.0 else n_steps + 1
    keys = jax.random.split(key, n_key_slots)
    chex.assert_shape(keys, (n_key_slots, 2))
    if config.proposal_family == "laplace":
        initial_noise = jax.random.laplace(keys[0], (batch, n_proposals, dimension), dtype=anchors.dtype)
    else:
        initial_noise = jax.random.normal(keys[0], (batch, n_proposals, dimension), dtype=anchors.dtype)
    chex.assert_shape(initial_noise, (batch, n_proposals, dimension))
    initial = anchors[:, None, :] + config.proposal_scale * initial_noise
    chex.assert_shape(initial, (batch, n_proposals, dimension))
    if n_steps == 0:
        log_weights = jnp.zeros((batch, n_proposals), dtype=anchors.dtype)
        chex.assert_shape(log_weights, (batch, n_proposals))
        if config.anchor_mix_prob > 0.0:
            stay_key = keys[-1]
            stay_mask = jax.random.bernoulli(
                stay_key, config.anchor_mix_prob, (batch, n_proposals, 1))
            chex.assert_shape(stay_mask, (batch, n_proposals, 1))
            initial = jnp.where(stay_mask, anchors[:, None, :], initial)
            chex.assert_shape(initial, (batch, n_proposals, dimension))
        samples = jax.lax.stop_gradient(initial)
        chex.assert_shape(samples, (batch, n_proposals, dimension))
        return samples, log_weights

    flat_anchors = jnp.repeat(anchors, n_proposals, axis=0)
    chex.assert_shape(flat_anchors, (batch * n_proposals, dimension))
    gradient_one = jax.grad(log_target, argnums=1)

    def ula_step(current: Array, step_key: Array) -> tuple[Array, tuple[Array, Array]]:
        """Advance one ULA step; inputs are current particles/key, outputs next particles/transition data."""

        chex.assert_shape(current, (batch, n_proposals, dimension))
        flat_current = current.reshape(batch * n_proposals, dimension)
        chex.assert_shape(flat_current, (batch * n_proposals, dimension))
        flat_gradient = jax.vmap(gradient_one)(flat_anchors, flat_current)
        chex.assert_shape(flat_gradient, (batch * n_proposals, dimension))
        gradient = flat_gradient.reshape(batch, n_proposals, dimension)
        chex.assert_shape(gradient, (batch, n_proposals, dimension))
        gradient_norm = jnp.linalg.norm(gradient, axis=2, keepdims=True)
        chex.assert_shape(gradient_norm, (batch, n_proposals, 1))
        gradient_factor = jnp.minimum(
            1.0, config.ula_gradient_clip / jnp.maximum(gradient_norm, 1e-12))
        chex.assert_shape(gradient_factor, (batch, n_proposals, 1))
        gradient = gradient * gradient_factor
        chex.assert_shape(gradient, (batch, n_proposals, dimension))
        transition_mean = current + 0.5 * config.ula_step_size * gradient
        chex.assert_shape(transition_mean, (batch, n_proposals, dimension))
        noise = jax.random.normal(step_key, current.shape, dtype=current.dtype)
        chex.assert_shape(noise, (batch, n_proposals, dimension))
        following = transition_mean + jnp.sqrt(config.ula_step_size) * noise
        chex.assert_shape(following, (batch, n_proposals, dimension))
        return following, (current, transition_mean)

    final, (previous_history, mean_history) = jax.lax.scan(ula_step, initial, keys[1:])
    chex.assert_shape(final, (batch, n_proposals, dimension))
    chex.assert_shape(previous_history, (n_steps, batch, n_proposals, dimension))
    chex.assert_shape(mean_history, (n_steps, batch, n_proposals, dimension))
    final_mean = mean_history[-1]
    chex.assert_shape(final_mean, (batch, n_proposals, dimension))
    reference_mean = jnp.broadcast_to(anchors[:, None, :], final.shape)
    chex.assert_shape(reference_mean, (batch, n_proposals, dimension))
    transition_variance = jnp.asarray(config.ula_step_size, dtype=anchors.dtype)
    chex.assert_shape(transition_variance, ())
    if config.proposal_family == "laplace":
        reference_scale = jnp.asarray(config.proposal_scale, dtype=anchors.dtype)
        chex.assert_shape(reference_scale, ())
        log_reference = laplace_log_density(final, reference_mean, reference_scale)
    else:
        reference_variance = jnp.asarray(config.proposal_scale**2, dtype=anchors.dtype)
        chex.assert_shape(reference_variance, ())
        log_reference = gaussian_log_density(final, reference_mean, reference_variance)
    chex.assert_shape(log_reference, (batch, n_proposals))
    log_transition = gaussian_log_density(final, final_mean, transition_variance)
    chex.assert_shape(log_transition, (batch, n_proposals))
    log_weights = log_reference - log_transition
    chex.assert_shape(log_weights, (batch, n_proposals))
    samples = jax.lax.stop_gradient(final)
    chex.assert_shape(samples, (batch, n_proposals, dimension))
    log_weights = jax.lax.stop_gradient(log_weights)
    chex.assert_shape(log_weights, (batch, n_proposals))
    return samples, log_weights


def entropic_dual(
    state: RobustState,
    pathwise_loss: Array,
    transport_cost: Array,
    log_weights: Array,
    config: RobustConfig,
) -> Array:
    """Compute the entropic WDRO dual; inputs are state/path arrays/config and output is a scalar objective."""

    chex.assert_equal_shape((pathwise_loss, transport_cost, log_weights))
    batch, n_proposals = pathwise_loss.shape
    chex.assert_shape(state.raw_lambda, ())
    lambda_ = jax.nn.softplus(state.raw_lambda)
    chex.assert_shape(lambda_, ())
    temperature = config.kl_epsilon + lambda_ * config.tau
    chex.assert_shape(temperature, ())
    penalized = pathwise_loss - lambda_ * transport_cost
    chex.assert_shape(penalized, (batch, n_proposals))
    scores = penalized / temperature + log_weights
    chex.assert_shape(scores, (batch, n_proposals))
    inner = jax.scipy.special.logsumexp(scores, axis=1) - jnp.log(n_proposals)
    chex.assert_shape(inner, (batch,))
    objective = lambda_ * config.rho + temperature * jnp.mean(inner)
    chex.assert_shape(objective, ())
    return objective


def joint_cvar_dual(
    state: RobustState,
    pathwise_loss: Array,
    transport_cost: Array,
    log_weights: Array,
    config: RobustConfig,
) -> Array:
    """Compute the joint-L^p WDRO dual; inputs are state/path arrays/config and output is a scalar objective.

    With config.tail_power == 1.0 this is the standard CVaR-shaped dual (L^1 norm of the
    positive part). Larger tail_power uses an L^p norm of the positive part instead, which
    is no longer a CVaR but penalizes large individual tail excesses more sharply. When
    config.exact_eta is set, eta is not a dual-ascent variable but the exact per-batch
    (1 - alpha) empirical quantile of the penalized loss (stop-gradient), following the
    closed-form optimal eta for the p=1 case; for p != 1 this is a heuristic plug-in.
    """

    chex.assert_equal_shape((pathwise_loss, transport_cost, log_weights))
    batch, n_proposals = pathwise_loss.shape
    chex.assert_shape(state.raw_lambda, ())
    chex.assert_shape(state.eta, ())
    lambda_ = jax.nn.softplus(state.raw_lambda)
    chex.assert_shape(lambda_, ())
    penalized = pathwise_loss - lambda_ * transport_cost
    chex.assert_shape(penalized, (batch, n_proposals))
    if config.exact_eta:
        eta = jax.lax.stop_gradient(jnp.quantile(penalized, config.alpha))
    else:
        eta = state.eta
    chex.assert_shape(eta, ())
    tail = jax.nn.relu(penalized - eta)
    chex.assert_shape(tail, (batch, n_proposals))
    safe_tail = jnp.maximum(tail, jnp.finfo(tail.dtype).tiny)
    chex.assert_shape(safe_tail, (batch, n_proposals))
    log_tail = jnp.where(tail > 0.0, jnp.log(safe_tail), -jnp.inf)
    chex.assert_shape(log_tail, (batch, n_proposals))
    log_weighted_tail_power = log_weights + config.tail_power * log_tail
    chex.assert_shape(log_weighted_tail_power, (batch, n_proposals))
    log_tail_power_mean = (jax.scipy.special.logsumexp(log_weighted_tail_power)
                           - jnp.log(batch * n_proposals))
    chex.assert_shape(log_tail_power_mean, ())
    weighted_tail_norm = jnp.where(
        jnp.all(tail == 0.0), 0.0, jnp.exp(log_tail_power_mean / config.tail_power))
    chex.assert_shape(weighted_tail_norm, ())
    objective = lambda_ * config.rho + eta + weighted_tail_norm / (1.0 - config.alpha)
    chex.assert_shape(objective, ())
    return objective


def relaxed_kl_cvar_dual(
    state: RobustState,
    pathwise_loss: Array,
    transport_cost: Array,
    log_weights: Array,
    config: RobustConfig,
) -> Array:
    """Compute the relaxed-marginal KL+L^p WDRO dual; inputs are state/path arrays/config, output scalar.

    Implements note_wdro_solve.typ's @relaxed-kl-dual: the per-anchor entropic-smoothed
    value h_lambda(xi) (same inner log-sum-exp as entropic_dual) is wrapped in a joint-style
    single shared eta and an L^p(P) norm of the positive part over the batch, combining KL
    regularization within each anchor's proposal with a CVaR-style constraint across anchors.
    """

    chex.assert_equal_shape((pathwise_loss, transport_cost, log_weights))
    batch, n_proposals = pathwise_loss.shape
    chex.assert_shape(state.raw_lambda, ())
    chex.assert_shape(state.eta, ())
    lambda_ = jax.nn.softplus(state.raw_lambda)
    chex.assert_shape(lambda_, ())
    temperature = config.kl_epsilon + lambda_ * config.tau
    chex.assert_shape(temperature, ())
    penalized = pathwise_loss - lambda_ * transport_cost
    chex.assert_shape(penalized, (batch, n_proposals))
    scores = penalized / temperature + log_weights
    chex.assert_shape(scores, (batch, n_proposals))
    inner = jax.scipy.special.logsumexp(scores, axis=1) - jnp.log(n_proposals)
    chex.assert_shape(inner, (batch,))
    h_lambda = temperature * inner
    chex.assert_shape(h_lambda, (batch,))
    if config.exact_eta:
        eta = jax.lax.stop_gradient(jnp.quantile(h_lambda, config.alpha))
    else:
        eta = state.eta
    chex.assert_shape(eta, ())
    tail = jax.nn.relu(h_lambda - eta)
    chex.assert_shape(tail, (batch,))
    safe_tail = jnp.maximum(tail, jnp.finfo(tail.dtype).tiny)
    chex.assert_shape(safe_tail, (batch,))
    log_tail = jnp.where(tail > 0.0, jnp.log(safe_tail), -jnp.inf)
    chex.assert_shape(log_tail, (batch,))
    log_tail_power_mean = (jax.scipy.special.logsumexp(config.tail_power * log_tail)
                           - jnp.log(batch))
    chex.assert_shape(log_tail_power_mean, ())
    tail_norm = jnp.where(jnp.all(tail == 0.0), 0.0, jnp.exp(log_tail_power_mean / config.tail_power))
    chex.assert_shape(tail_norm, ())
    objective = lambda_ * config.rho + eta + tail_norm / (1.0 - config.alpha)
    chex.assert_shape(objective, ())
    return objective


def per_sample_cvar_dual(
    state: RobustState,
    pathwise_loss: Array,
    transport_cost: Array,
    log_weights: Array,
    config: RobustConfig,
) -> Array:
    """Compute the conditional-L^p WDRO dual; inputs are state/path arrays/config and output is a scalar objective.

    Implements the conditional (per-sample) constraint from note_wdro_solve.typ's
    @conditional-dual, as opposed to joint_cvar_dual's single shared threshold
    (@joint-dual): every anchor xi gets its own eta(xi), computed exactly as the
    (1 - alpha) empirical quantile of that anchor's own proposal batch (stop-gradient).
    There is no persistent per-sample dual variable to learn (anchors change every
    step), so eta is always computed on the fly here regardless of config.exact_eta.
    """

    chex.assert_equal_shape((pathwise_loss, transport_cost, log_weights))
    batch, n_proposals = pathwise_loss.shape
    chex.assert_shape(state.raw_lambda, ())
    lambda_ = jax.nn.softplus(state.raw_lambda)
    chex.assert_shape(lambda_, ())
    penalized = pathwise_loss - lambda_ * transport_cost
    chex.assert_shape(penalized, (batch, n_proposals))
    eta = jax.lax.stop_gradient(jnp.quantile(penalized, config.alpha, axis=1))
    chex.assert_shape(eta, (batch,))
    tail = jax.nn.relu(penalized - eta[:, None])
    chex.assert_shape(tail, (batch, n_proposals))
    safe_tail = jnp.maximum(tail, jnp.finfo(tail.dtype).tiny)
    chex.assert_shape(safe_tail, (batch, n_proposals))
    log_tail = jnp.where(tail > 0.0, jnp.log(safe_tail), -jnp.inf)
    chex.assert_shape(log_tail, (batch, n_proposals))
    log_weighted_tail_power = log_weights + config.tail_power * log_tail
    chex.assert_shape(log_weighted_tail_power, (batch, n_proposals))
    log_tail_power_mean = (jax.scipy.special.logsumexp(log_weighted_tail_power, axis=1)
                           - jnp.log(n_proposals))
    chex.assert_shape(log_tail_power_mean, (batch,))
    weighted_tail_norm = jnp.where(
        jnp.all(tail == 0.0, axis=1), 0.0, jnp.exp(log_tail_power_mean / config.tail_power))
    chex.assert_shape(weighted_tail_norm, (batch,))
    per_sample = eta + weighted_tail_norm / (1.0 - config.alpha)
    chex.assert_shape(per_sample, (batch,))
    objective = lambda_ * config.rho + jnp.mean(per_sample)
    chex.assert_shape(objective, ())
    return objective
