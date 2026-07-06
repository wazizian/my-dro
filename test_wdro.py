"""Pytest coverage for WDRO duals and augmented-space ULA proposals."""

import dataclasses

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from wdro import (
    RobustConfig,
    RobustState,
    entropic_dual,
    joint_cvar_dual,
    quadratic_transport_cost,
    plain_average_dual,
    plain_cvar_of_average_dual,
    relaxed_kl_cvar_dual,
    sample_langevin_proposal,
)


def inverse_softplus(value: float) -> jax.Array:
    """Convert positive lambda to its unconstrained representation; input is lambda, output is scalar."""

    result = jnp.log(jnp.expm1(jnp.asarray(value)))
    chex.assert_shape(result, ())
    return result


def test_quadratic_transport_cost_is_zero_at_anchor_and_nonnegative():
    """Verify the reference-kernel cost is zero at its anchor and positive away from it."""

    anchors = jnp.asarray([[1.0, -1.0], [0.0, 2.0]])
    chex.assert_shape(anchors, (2, 2))
    samples = jnp.stack((anchors, anchors + 0.5), axis=1)
    chex.assert_shape(samples, (2, 2, 2))
    costs = quadratic_transport_cost(samples, anchors, jnp.asarray(0.5))
    chex.assert_shape(costs, (2, 2))
    np.testing.assert_array_equal(costs[:, 0], 0.0)
    assert bool(jnp.all(costs >= 0.0))
    assert bool(jnp.all(costs[:, 1] > 0.0))


@pytest.mark.parametrize("n_steps", [0, 1, 2])
def test_langevin_proposal_shapes_weights_and_determinism(n_steps):
    """Verify ULA proposals for k=0,1,2 have valid shapes, weights, and fixed-key output."""

    anchors = jnp.asarray([[0.0, 1.0], [2.0, -1.0]])
    chex.assert_shape(anchors, (2, 2))
    config = RobustConfig(proposal_samples=8, proposal_scale=0.7,
                          ula_steps=n_steps, ula_step_size=0.2)

    def log_target(anchor, sample):
        """Return a quadratic tilted log target; inputs are anchor/sample and output is scalar."""

        chex.assert_shape(anchor, (2,))
        chex.assert_shape(sample, (2,))
        result = -0.5 * jnp.sum(jnp.square(sample - anchor)) + 0.2 * sample[0]
        chex.assert_shape(result, ())
        return result

    key = jax.random.PRNGKey(21)
    samples, log_weights = sample_langevin_proposal(key, anchors, log_target, config)
    repeated_samples, repeated_weights = sample_langevin_proposal(key, anchors, log_target, config)
    chex.assert_shape(samples, (2, 8, 2))
    chex.assert_shape(log_weights, (2, 8))
    assert bool(jnp.all(jnp.isfinite(samples)))
    assert bool(jnp.all(jnp.isfinite(log_weights)))
    np.testing.assert_array_equal(samples, repeated_samples)
    np.testing.assert_array_equal(log_weights, repeated_weights)
    if n_steps == 0:
        np.testing.assert_array_equal(log_weights, 0.0)
    else:
        assert bool(jnp.any(log_weights != 0.0))


def test_anchor_mix_prob_one_zeroes_out_bar_pi_weight():
    """Verify anchor_mix_prob=1 puts an exact anchor slot first and -inf weight on the rest.

    The mixture is now split analytically rather than by Bernoulli-sampling which branch each
    proposal comes from: the dirac component is always a single deterministic extra sample at
    the anchor (slot 0), and the remaining n_proposals slots stay genuine bar_pi draws whose
    log-weight reflects the (1 - p) mixture mass, so at p=1 they carry -inf weight (i.e.
    contribute nothing) without needing to literally collapse their sample values onto the anchor.
    """

    anchors = jnp.asarray([[0.0, 1.0], [2.0, -1.0]])
    config = RobustConfig(proposal_samples=8, proposal_scale=0.7,
                          ula_steps=0, anchor_mix_prob=1.0)

    def log_target(anchor, sample):
        return -0.5 * jnp.sum(jnp.square(sample - anchor))

    key = jax.random.PRNGKey(7)
    samples, log_weights = sample_langevin_proposal(key, anchors, log_target, config)
    chex.assert_shape(samples, (2, 9, 2))
    chex.assert_shape(log_weights, (2, 9))
    np.testing.assert_allclose(samples[:, 0, :], anchors)
    assert bool(jnp.all(jnp.isfinite(log_weights[:, 0])))
    assert bool(jnp.all(jnp.isneginf(log_weights[:, 1:])))


def test_anchor_mix_prob_partial_matches_closed_form_mixture_weights():
    """Verify 0<anchor_mix_prob<1 yields an exact anchor slot plus correctly-weighted bar_pi draws."""

    anchors = jnp.asarray([[0.0, 1.0]])
    n_proposals = 16
    config = RobustConfig(proposal_samples=n_proposals, proposal_scale=0.7,
                          ula_steps=0, anchor_mix_prob=0.5)

    def log_target(anchor, sample):
        return -0.5 * jnp.sum(jnp.square(sample - anchor))

    key = jax.random.PRNGKey(11)
    samples, log_weights = sample_langevin_proposal(key, anchors, log_target, config)
    chex.assert_shape(samples, (1, n_proposals + 1, 2))
    chex.assert_shape(log_weights, (1, n_proposals + 1))
    np.testing.assert_allclose(samples[:, 0, :], anchors)
    assert bool(jnp.all(jnp.abs(samples[:, 1:, :] - anchors[:, None, :]) > 0.0))
    total_slots = n_proposals + 1
    expected_anchor_weight = jnp.log(0.5) + jnp.log(total_slots)
    expected_bar_weight = jnp.log(0.5) - jnp.log(n_proposals) + jnp.log(total_slots)
    np.testing.assert_allclose(log_weights[:, 0], expected_anchor_weight, rtol=1e-5)
    np.testing.assert_allclose(log_weights[:, 1:], expected_bar_weight, rtol=1e-5)
    # The two mixture components' total probability mass must sum to 1.
    mixture_mass = jnp.exp(log_weights[:, 0]) / total_slots + jnp.sum(
        jnp.exp(log_weights[:, 1:]), axis=1) / total_slots
    np.testing.assert_allclose(mixture_mass, 1.0, rtol=1e-5)


def test_anchor_mix_prob_rejects_positive_ula_steps():
    """Verify anchor_mix_prob>0 with ula_steps>0 raises, since mixing is only defined for k=0."""

    anchors = jnp.asarray([[0.0, 1.0]])
    config = RobustConfig(proposal_samples=4, proposal_scale=0.7,
                          ula_steps=1, ula_step_size=0.1, anchor_mix_prob=0.5)

    def log_target(anchor, sample):
        return -0.5 * jnp.sum(jnp.square(sample - anchor))

    with pytest.raises(ValueError):
        sample_langevin_proposal(jax.random.PRNGKey(3), anchors, log_target, config)


@pytest.mark.parametrize("n_steps", [0, 1, 2])
def test_langevin_importance_weight_is_one_when_transition_equals_reference(n_steps):
    """Verify k=0,1,2 weights equal one when every ULA transition is exactly the reference kernel."""

    anchors = jnp.asarray([[0.0], [1.0]])
    chex.assert_shape(anchors, (2, 1))
    step_size = 0.25
    config = RobustConfig(proposal_samples=16, proposal_scale=np.sqrt(step_size),
                          ula_steps=n_steps, ula_step_size=step_size,
                          ula_gradient_clip=1e6)

    def log_target(anchor, sample):
        """Give ULA a drift whose Gaussian transition mean equals the anchor."""

        chex.assert_shape(anchor, (1,))
        chex.assert_shape(sample, (1,))
        result = -jnp.sum(jnp.square(sample - anchor)) / step_size
        chex.assert_shape(result, ())
        return result

    _, log_weights = sample_langevin_proposal(
        jax.random.PRNGKey(27), anchors, log_target, config)
    chex.assert_shape(log_weights, (2, 16))
    np.testing.assert_allclose(log_weights, 0.0, atol=2e-6)


def test_entropic_dual_matches_per_sample_log_mean_exp():
    """Verify the entropic dual averages separate proposal log-expectations for each base sample."""

    lambda_value = 0.4
    state = RobustState(inverse_softplus(lambda_value), jnp.asarray(0.0),
                        jnp.zeros((0,)), jnp.ones((0,)))
    config = RobustConfig(rho=0.3, kl_epsilon=0.7, tau=0.2)
    losses = jnp.asarray([[1.0, 2.0], [-1.0, 3.0]])
    costs = jnp.asarray([[0.0, 0.5], [1.0, 0.25]])
    log_weights = jnp.log(jnp.asarray([[0.8, 1.2], [1.5, 0.5]]))
    objective = entropic_dual(state, losses, costs, log_weights, config)
    temperature = config.kl_epsilon + lambda_value * config.tau
    scores = (np.asarray(losses) - lambda_value * np.asarray(costs)) / temperature
    expected_inner = np.log(np.mean(np.exp(scores) * np.exp(np.asarray(log_weights)), axis=1))
    expected = lambda_value * config.rho + temperature * np.mean(expected_inner)
    np.testing.assert_allclose(objective, expected, rtol=2e-6)


def test_entropic_dual_is_finite_for_large_losses():
    """Verify log-sum-exp evaluation avoids overflow for large pathwise losses."""

    state = RobustState(inverse_softplus(0.5), jnp.asarray(0.0),
                        jnp.zeros((0,)), jnp.ones((0,)))
    config = RobustConfig(rho=0.1, kl_epsilon=0.1)
    losses = jnp.asarray([[1000.0, 1001.0], [999.0, 1002.0]])
    costs = jnp.zeros((2, 2))
    log_weights = jnp.zeros((2, 2))
    objective = entropic_dual(state, losses, costs, log_weights, config)
    chex.assert_shape(objective, ())
    assert bool(jnp.isfinite(objective))


def test_joint_cvar_dual_uses_one_scalar_eta_and_joint_expectation():
    """Verify joint CVaR pools base/proposal samples and uses one scalar threshold."""

    lambda_value = 0.25
    eta = jnp.asarray(0.6)
    chex.assert_shape(eta, ())
    state = RobustState(inverse_softplus(lambda_value), eta, jnp.zeros((0,)), jnp.ones((0,)))
    config = RobustConfig(objective="joint_cvar", rho=0.2, alpha=0.75)
    losses = jnp.asarray([[0.0, 1.0], [2.0, 4.0]])
    costs = jnp.asarray([[0.0, 0.2], [0.4, 0.8]])
    log_weights = jnp.zeros((2, 2))
    objective = joint_cvar_dual(state, losses, costs, log_weights, config)
    penalized = np.asarray(losses) - lambda_value * np.asarray(costs)
    expected = lambda_value * config.rho + float(eta) + np.mean(
        np.maximum(penalized - float(eta), 0.0)) / (1.0 - config.alpha)
    np.testing.assert_allclose(objective, expected, rtol=2e-6)


def test_relaxed_kl_cvar_dual_matches_manual_h_lambda_wrap():
    """Verify relaxed-kl-cvar wraps the per-anchor entropic h_lambda in a joint L^1 tail."""

    lambda_value = 0.3
    eta = jnp.asarray(0.5)
    state = RobustState(inverse_softplus(lambda_value), eta, jnp.zeros((0,)), jnp.ones((0,)))
    config = RobustConfig(objective="relaxed_kl_cvar", rho=0.2, kl_epsilon=0.6, tau=0.1, alpha=0.8)
    losses = jnp.asarray([[1.0, 2.0], [-1.0, 3.0]])
    costs = jnp.asarray([[0.0, 0.5], [1.0, 0.25]])
    log_weights = jnp.log(jnp.asarray([[0.8, 1.2], [1.5, 0.5]]))
    objective = relaxed_kl_cvar_dual(state, losses, costs, log_weights, config)
    temperature = config.kl_epsilon + lambda_value * config.tau
    scores = (np.asarray(losses) - lambda_value * np.asarray(costs)) / temperature
    inner = np.log(np.mean(np.exp(scores) * np.exp(np.asarray(log_weights)), axis=1))
    h_lambda = temperature * inner
    expected = lambda_value * config.rho + float(eta) + np.mean(
        np.maximum(h_lambda - float(eta), 0.0)) / (1.0 - config.alpha)
    np.testing.assert_allclose(objective, expected, rtol=2e-6)


def test_relaxed_kl_cvar_dual_reduces_to_entropic_when_eta_below_min_h_lambda():
    """Verify a very negative eta makes every h_lambda(xi) fall in the tail, recovering the entropic mean."""

    lambda_value = 0.3
    eta = jnp.asarray(-100.0)
    state = RobustState(inverse_softplus(lambda_value), eta, jnp.zeros((0,)), jnp.ones((0,)))
    config = RobustConfig(objective="relaxed_kl_cvar", rho=0.2, kl_epsilon=0.6, tau=0.1, alpha=0.0)
    losses = jnp.asarray([[1.0, 2.0], [-1.0, 3.0]])
    costs = jnp.asarray([[0.0, 0.5], [1.0, 0.25]])
    log_weights = jnp.zeros((2, 2))
    entropic_config = dataclasses.replace(config, objective="entropic")
    entropic_state = RobustState(inverse_softplus(lambda_value), jnp.asarray(0.0),
                                  jnp.zeros((0,)), jnp.ones((0,)))
    expected = entropic_dual(entropic_state, losses, costs, log_weights, entropic_config)
    objective = relaxed_kl_cvar_dual(state, losses, costs, log_weights, config)
    np.testing.assert_allclose(objective, expected, rtol=1e-5)


def test_plain_average_dual_matches_manual_mean():
    """Verify plain_average_dual is exactly the mean pathwise loss over anchors and proposals."""

    state = RobustState(inverse_softplus(0.3), jnp.asarray(0.5), jnp.zeros((0,)), jnp.ones((0,)))
    config = RobustConfig(objective="plain_average", rho=0.2)
    losses = jnp.asarray([[1.0, 2.0, -3.0], [0.5, -1.5, 4.0]])
    costs = jnp.asarray([[0.0, 0.5, 1.0], [0.2, 0.4, 0.6]])
    log_weights = jnp.zeros((2, 3))
    objective = plain_average_dual(state, losses, costs, log_weights, config)
    np.testing.assert_allclose(objective, np.mean(np.asarray(losses)), rtol=1e-6)


def test_plain_average_dual_ignores_lambda_and_transport_cost():
    """Verify changing lambda or transport_cost leaves plain_average_dual unchanged."""

    losses = jnp.asarray([[1.0, 2.0], [-1.0, 3.0]])
    log_weights = jnp.zeros((2, 2))
    config = RobustConfig(objective="plain_average", rho=0.2)
    baseline_costs = jnp.zeros((2, 2))
    baseline_state = RobustState(inverse_softplus(0.1), jnp.asarray(0.0), jnp.zeros((0,)), jnp.ones((0,)))
    baseline = plain_average_dual(baseline_state, losses, baseline_costs, log_weights, config)

    perturbed_costs = jnp.asarray([[0.0, 5.0], [10.0, 0.25]])
    perturbed_state = RobustState(inverse_softplus(9.0), jnp.asarray(0.0), jnp.zeros((0,)), jnp.ones((0,)))
    perturbed = plain_average_dual(perturbed_state, losses, perturbed_costs, log_weights, config)
    np.testing.assert_allclose(baseline, perturbed, rtol=1e-6)


def test_plain_cvar_of_average_dual_matches_manual_cvar_of_mean():
    """Verify plain_cvar_of_average_dual is a joint CVaR over plain per-anchor proposal means."""

    eta = jnp.asarray(0.4)
    state = RobustState(inverse_softplus(0.3), eta, jnp.zeros((0,)), jnp.ones((0,)))
    config = RobustConfig(objective="plain_cvar_of_average", rho=0.2, alpha=0.75)
    losses = jnp.asarray([[1.0, 2.0, 0.0], [-1.0, 3.0, 5.0]])
    costs = jnp.asarray([[0.0, 0.5, 1.0], [0.2, 0.4, 0.6]])
    log_weights = jnp.log(jnp.asarray([[0.5, 1.0, 1.5], [1.0, 1.0, 1.0]]))
    objective = plain_cvar_of_average_dual(state, losses, costs, log_weights, config)
    weights = np.exp(np.asarray(log_weights))
    weights = weights / weights.sum(axis=1, keepdims=True)
    per_anchor_average = np.sum(weights * np.asarray(losses), axis=1)
    expected = float(eta) + np.mean(np.maximum(per_anchor_average - float(eta), 0.0)) / (1.0 - config.alpha)
    np.testing.assert_allclose(objective, expected, rtol=2e-6)


def test_plain_cvar_of_average_dual_ignores_lambda_and_transport_cost():
    """Verify changing lambda or transport_cost leaves plain_cvar_of_average_dual unchanged."""

    losses = jnp.asarray([[1.0, 2.0], [-1.0, 3.0]])
    log_weights = jnp.zeros((2, 2))
    config = RobustConfig(objective="plain_cvar_of_average", rho=0.2, alpha=0.7)
    baseline_costs = jnp.zeros((2, 2))
    baseline_state = RobustState(inverse_softplus(0.1), jnp.asarray(0.6), jnp.zeros((0,)), jnp.ones((0,)))
    baseline = plain_cvar_of_average_dual(baseline_state, losses, baseline_costs, log_weights, config)

    perturbed_costs = jnp.asarray([[0.0, 5.0], [10.0, 0.25]])
    perturbed_state = RobustState(inverse_softplus(9.0), jnp.asarray(0.6), jnp.zeros((0,)), jnp.ones((0,)))
    perturbed = plain_cvar_of_average_dual(perturbed_state, losses, perturbed_costs, log_weights, config)
    np.testing.assert_allclose(baseline, perturbed, rtol=1e-6)


def test_plain_cvar_of_average_dual_exact_eta_matches_empirical_quantile():
    """Verify exact_eta uses the (1 - alpha) empirical quantile of the per-anchor averages."""

    state = RobustState(inverse_softplus(0.3), jnp.asarray(999.0), jnp.zeros((0,)), jnp.ones((0,)))
    config = RobustConfig(objective="plain_cvar_of_average", rho=0.2, alpha=0.5, exact_eta=True)
    losses = jnp.asarray([[1.0, 3.0], [2.0, 4.0], [-1.0, 1.0], [0.0, 0.0]])
    costs = jnp.zeros((4, 2))
    log_weights = jnp.zeros((4, 2))
    objective = plain_cvar_of_average_dual(state, losses, costs, log_weights, config)
    per_anchor_average = np.mean(np.asarray(losses), axis=1)
    eta = np.quantile(per_anchor_average, config.alpha)
    expected = eta + np.mean(np.maximum(per_anchor_average - eta, 0.0)) / (1.0 - config.alpha)
    np.testing.assert_allclose(objective, expected, rtol=2e-6)


@pytest.mark.parametrize(
    "objective", ["entropic", "joint_cvar", "relaxed_kl_cvar", "plain_average", "plain_cvar_of_average"])
def test_jitted_dual_has_finite_parameter_gradients(objective):
    """Verify each top-level-jitted dual has finite gradients in lambda and scalar eta."""

    state = RobustState(jnp.asarray(-0.5), jnp.asarray(0.2), jnp.zeros((0,)), jnp.ones((0,)))
    config = RobustConfig(objective=objective)
    losses = jnp.asarray([[0.1, 0.7], [1.2, -0.4]])
    costs = jnp.asarray([[0.0, 0.2], [0.1, 0.3]])
    log_weights = jnp.zeros((2, 2))
    dual_by_objective = {
        "entropic": entropic_dual,
        "joint_cvar": joint_cvar_dual,
        "relaxed_kl_cvar": relaxed_kl_cvar_dual,
        "plain_average": plain_average_dual,
        "plain_cvar_of_average": plain_cvar_of_average_dual,
    }
    dual = dual_by_objective[objective]
    compiled_dual = eqx.filter_jit(jax.value_and_grad(dual))
    value, gradient = compiled_dual(state, losses, costs, log_weights, config)
    chex.assert_shape(value, ())
    chex.assert_shape(gradient.raw_lambda, ())
    chex.assert_shape(gradient.eta, ())
    assert bool(jnp.isfinite(value))
    assert bool(jnp.isfinite(gradient.raw_lambda))
    assert bool(jnp.isfinite(gradient.eta))
    assert float(jax.nn.softplus(state.raw_lambda)) >= 0.0
