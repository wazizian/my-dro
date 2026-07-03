"""Pytest coverage for WDRO duals and augmented-space ULA proposals."""

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


@pytest.mark.parametrize("objective", ["entropic", "joint_cvar"])
def test_jitted_dual_has_finite_parameter_gradients(objective):
    """Verify each top-level-jitted dual has finite gradients in lambda and scalar eta."""

    state = RobustState(jnp.asarray(-0.5), jnp.asarray(0.2), jnp.zeros((0,)), jnp.ones((0,)))
    config = RobustConfig(objective=objective)
    losses = jnp.asarray([[0.1, 0.7], [1.2, -0.4]])
    costs = jnp.asarray([[0.0, 0.2], [0.1, 0.3]])
    log_weights = jnp.zeros((2, 2))
    dual = entropic_dual if objective == "entropic" else joint_cvar_dual
    compiled_dual = eqx.filter_jit(jax.value_and_grad(dual))
    value, gradient = compiled_dual(state, losses, costs, log_weights, config)
    chex.assert_shape(value, ())
    chex.assert_shape(gradient.raw_lambda, ())
    chex.assert_shape(gradient.eta, ())
    assert bool(jnp.isfinite(value))
    assert bool(jnp.isfinite(gradient.raw_lambda))
    assert bool(jnp.isfinite(gradient.eta))
    assert float(jax.nn.softplus(state.raw_lambda)) >= 0.0
