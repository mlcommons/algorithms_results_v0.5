# coding=utf-8
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import functools
import os
from typing import Any, Optional, Sequence, Tuple

from absl import logging
import chex
import flax
import haiku as hk
import jax
from jax import lax
import jax.numpy as jnp
from learned_optimization import summary
from learned_optimization import tree_utils
from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.learned_optimizers import common
from learned_optimization.optimizers import base as opt_base
import numpy as onp


def _fractional_tanh_embed(x):
    def one_freq(timescale):
        return jnp.tanh((x - (jnp.float32(timescale))) * 10)

    timescales = jnp.asarray([0.03, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.1], dtype=jnp.float32)
    return jax.vmap(one_freq)(timescales)


def factored_dims(shape: Sequence[int]) -> Optional[Tuple[int, int]]:
    """Whether to use a factored second moment estimator.

    If there are not two dimensions of size >= min_dim_size_to_factor, then we
    do not factor. If we do factor the accumulator, then this function returns a
    tuple of the two largest axes to reduce over.

    Args:
        shape: a Shape

    Returns:
        None or a tuple of ints
    """
    if len(shape) < 2:
        return None
    sorted_dims = onp.argsort(shape)
    return int(sorted_dims[-2]), int(sorted_dims[-1])


def _clip_log_abs(v, scale=1.0):
    mag = jnp.log(1e-8 + jnp.abs(v * scale))
    return jnp.clip(mag, -5, 5) * 0.5


def _sorted_values(dd):
    return list(zip(*sorted(dd.items(), key=lambda x: x[0])))[1]


class BufferLossAccumulators:
    """Rolling accumulator for loss values."""

    def __init__(self):
        pass

    def init(self, num_steps):
        halflife = jnp.logspace(1, jnp.log10(num_steps), 10)
        decays = jnp.exp(-1.0 / halflife)
        return {
            "means": jnp.zeros((len(decays),), dtype=jnp.float32),
            "iteration": jnp.asarray(0, dtype=jnp.int32),
            "running_min": 999999999999.0 * jnp.ones((len(decays),), dtype=jnp.float32),
            "decays": decays,
        }

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(self, state, loss):
        """Update the state with a new loss."""
        # wana clip the losses so it doesn't go absolutely insane.
        jdecays = state["decays"]
        cor_mean = state["means"] / (1 - jdecays ** (state["iteration"] + 1))
        approx_max = jnp.max(cor_mean)
        approx_max = jnp.where(state["iteration"] == 0, loss, approx_max)
        loss = jnp.minimum(jnp.abs(approx_max) * 2, loss)

        means = state["means"] * jdecays + loss * (1.0 - jdecays)

        cor_mean = means / (1 - jdecays ** (state["iteration"] + 1))
        running_min = jnp.minimum(state["running_min"], cor_mean)

        return {
            "means": means,
            "iteration": state["iteration"] + 1,
            "running_min": running_min,
            "decays": state["decays"],
        }

    @functools.partial(jax.jit, static_argnums=(0,))
    def features(self, state):
        """Compute features to pass to NN from state."""
        jdecays = state["decays"]
        cor_mean = state["means"] / (1 - jdecays ** (state["iteration"]))
        # longest running decay
        approx_max = cor_mean[1:]
        cor_mean = cor_mean[0:-1]
        running_min = state["running_min"][0:-1]

        den = jnp.maximum(1e-8, (approx_max - running_min))
        pre_center = (cor_mean - running_min) / den
        feature1 = pre_center - 1.0
        feature1 = jnp.clip(feature1, -1, 1)
        # first couple features are bad.
        return jnp.where(state["iteration"] <= 2, feature1 * 0, feature1)


@flax.struct.dataclass
class State:
    """Inner state of learned optimizer."""

    rms_rolling: chex.ArrayTree
    mom_rolling: chex.ArrayTree
    fac_rolling: chex.ArrayTree
    tf_mean_rolling: chex.ArrayTree
    tf_var_rolling: chex.ArrayTree
    iteration: jnp.ndarray
    state: chex.ArrayTree
    num_steps: jnp.ndarray
    lstm_hidden_state: chex.ArrayTree
    loss_buffer: chex.ArrayTree


def _safe_rsqrt(x):
    return lax.rsqrt(jnp.maximum(x, 1e-9))


def _second_moment_normalizer(x, axis, eps=1e-5):
    return x * jax.lax.rsqrt(eps + jnp.mean(jnp.square(x), axis=axis, keepdims=True))


class Inv(lopt_base.LearnedOptimizer):
    """Experimental inner-task invariant learned optimizer."""

    def __init__(
        self,
        lstm_hidden_size=64,
        ff_hidden_size=4,
        ff_hidden_layers=2,
        initial_momentum_decays=(0.9,),
        initial_rms_decays=(0.999,),
        initial_adafactor_decays=(0.9, 0.99, 0.999),
        initial_tf_decays=(0.999,),
        param_inits=64,
        mix_layers=True,
        exp_mult=0.001,
        step_mult=0.001,
        validation_mode=False,
        with_validation_feature_dim=False,
        initial_epsilon=1e-8,
        # ablation flags.
        with_g=True,
        with_m=True,
        with_m_feat=True,
        with_rms=True,
        with_rms_feat=True,
        with_rms_norm_g=True,
        with_rsqrt_rms=True,
        with_p=True,
        with_fac_norm_g=True,
        with_fac_rms=True,
        with_fac_rsqrt=True,
        with_grad_clip_feat=True,
        with_fac_mom_mult=True,
        with_rms_only_norm_g=True,
        adafactor_accumulator=True,
        param_scale_mult=True,
        # use_bugged_next_lstm_state=False,
        # use_bugged_loss_features=True,
        precondition_output=False,
        reparam_decay=10.0,
        rnn_state_decay=0.0,
        # more summaries
        summarize_each_layer=True,
        summarize_all_control=False,
        # Modify the lopt to probe behavior
        constant_loss=False,
        clip_param_scale_amount=None,
    ):
        """Initializer.

        Args:
            lstm_hidden_size: size of the per tensor LSTM.
            ff_hidden_size: hidden size of the per-parameter MLP.
            ff_hidden_layers: number of layers in per-parameter mlp.
            initial_momentum_decays: The values of momentum accumulators to use
            initial_rms_decays: The values of the second moment gradient accumulators
                to use.
            initial_adafactor_decays: The values of the adafactor style accumulators
                to use.
            param_inits: Number of parameter inputs with which to linearly interpolate
                to create each per-parameter MLP.
            exp_mult: setting to rescale output of lopt
            step_mult: setting to rescale output of lopt  validation model: optionally
                add an additional input to LSTM to denote targeting train or valid loss.
            with_validation_feature: Set the above feature on or off.   <many ablation
                flags>
        """
        # TODO(lmetz): Remove reparam_decay -- is not being used.
        super().__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self.ff_hidden_size = ff_hidden_size
        self.ff_hidden_layers = ff_hidden_layers
        self.initial_momentum_decays = initial_momentum_decays
        self.initial_rms_decays = initial_rms_decays
        self.initial_adafactor_decays = initial_adafactor_decays
        self.initial_tf_decays = initial_tf_decays
        self.initial_epsilon = initial_epsilon
        self.param_inits = param_inits
        self.mix_layers = mix_layers
        self.with_g = with_g
        self.with_m = with_m
        self.with_m_feat = with_m_feat
        self.with_rms = with_rms
        self.with_rms_feat = with_rms_feat
        self.with_rms_norm_g = with_rms_norm_g
        self.with_rsqrt_rms = with_rsqrt_rms
        self.with_p = with_p
        self.with_fac_norm_g = with_fac_norm_g
        self.with_fac_rms = with_fac_rms
        self.with_fac_rsqrt = with_fac_rsqrt
        self.with_grad_clip_feat = with_grad_clip_feat
        self.with_fac_mom_mult = with_fac_mom_mult
        self.with_rms_only_norm_g = with_rms_only_norm_g
        self.adafactor_accumulator = adafactor_accumulator
        self.param_scale_mult = param_scale_mult
        self.exp_mult = exp_mult
        self.step_mult = step_mult
        self.summarize_each_layer = summarize_each_layer
        self.precondition_output = precondition_output
        self.reparam_decay = reparam_decay
        self.rnn_state_decay = rnn_state_decay
        self.with_validation_feature_dim = with_validation_feature_dim
        self.validation_mode = validation_mode
        self.constant_loss = constant_loss
        self.summarize_all_control = summarize_all_control
        self.clip_param_scale_amount = clip_param_scale_amount

        # if self.use_bugged_loss_features:
        #     logging.warning(
        #         "You are using bugged loss features! If you are using a"
        #         "pretrained optimizer, otherwise this is an error."
        #     )

        logging.info(
            f"Validation mode: {self.validation_mode} (with valid feature dim: {with_validation_feature_dim})"
        )
        self.lstm_fn = lambda: hk.LSTM(lstm_hidden_size, name="rnn")

        self.rnn = hk.without_apply_rng(hk.transform(self._rnn_forward))
        # self.ff_mod = hk.transform(self._ff_mod)
        self.buffer_loss_fns = BufferLossAccumulators()

    def _decay_to_param(self, x):
        return jnp.log(1 - x) / self.reparam_decay

    def _param_to_decay(self, x):
        return 1 - jnp.exp(x * self.reparam_decay)

    def accumulators_for_decays(self, mom_param=None, rms_param=None, adafactor_param=None):
        if mom_param is None:
            mom_decay = jnp.asarray(self.initial_momentum_decays)
        else:
            mom_decay = self._param_to_decay(
                self._decay_to_param(jnp.asarray(self.initial_momentum_decays)) + mom_param
            )
        if rms_param is None:
            rms_decay = jnp.asarray(self.initial_rms_decays)
        else:
            rms_decay = self._param_to_decay(
                self._decay_to_param(jnp.asarray(self.initial_rms_decays)) + rms_param
            )

        if adafactor_param is None:
            adafactor_decay = jnp.asarray(self.initial_adafactor_decays)
        else:
            adafactor_decay = self._param_to_decay(
                self._decay_to_param(jnp.asarray(self.initial_adafactor_decays)) + adafactor_param
            )

        mom_roll = common.vec_rolling_mom(mom_decay)
        rms_roll = common.vec_rolling_rms(rms_decay)
        fac_vec_roll = common.vec_factored_rolling(adafactor_decay)
        return mom_roll, rms_roll, fac_vec_roll

    def per_tensor_accumulators(self):
        mom_decay = jnp.asarray(self.initial_tf_decays)
        tf_mean_roll = common.vec_rolling_mom(mom_decay)
        tf_var_roll = common.vec_rolling_mom(mom_decay)
        return tf_mean_roll, tf_var_roll

    def _rnn_forward(self, x, state):
        if self.mix_layers:
            mix_layer = hk.Linear(self.lstm_hidden_size)(x)
            mix_layer = jax.nn.relu(mix_layer)
            mix_layer = hk.Linear(self.lstm_hidden_size)(x)
            mix_layer = jax.nn.relu(mix_layer)
            v = jnp.max(mix_layer, axis=0, keepdims=True)
            x = hk.Linear(self.lstm_hidden_size)(x) + v

        rnn_out, state = self.lstm_fn()(x, state)

        controls = None
        lr_mult = jnp.squeeze(hk.Linear(1, name="step_size")(rnn_out), -1)
        # clip LR between [1e-8. 1.0]
        lr_mult = jnp.clip(lr_mult, 1e-8 / self.step_mult, 1.0 / self.step_mult)
        return controls, lr_mult, state

    def _ff_mod(
        self,
        global_feat,
        extra_step_mult,
        p,
        g,
        m,
        rms,
        fac_g,
        fac_vec_col,
        fac_vec_row,
        fac_vec_v,
        summary_prefix,
    ):
        # step update used in nn_adam
        # step = (lr * (1.0 + epsilon)) * 1.0 / (epsilon + lax.sqrt(rms + 1e-10)) * m
        #
        # here we use original adam update but skipping bias correction since we
        # want our optimizer to generalize beyond training length.
        eps = self.initial_epsilon
        lr = extra_step_mult * self.step_mult
        # t = global_feat["iterations"] + 1
        # b1 = self.initial_momentum_decays[0]
        # b2 = self.initial_rms_decays[0]
        # m = m / (1 - b1**t)
        # rms = rms / (1 - b2**t)
        # jax.debug.print(f"b1 = {b1}")
        # jax.debug.print(f"b2 = {b2}")
        # jax.debug.print(f"t = {t}")
        param_scale = jnp.sqrt(jnp.mean(jnp.square(p)) + 1e-9)
        step = lr * 1.0 / (eps + lax.sqrt(rms + 1e-10)) * m
        step = step[..., -1] * param_scale

        # summaries
        if self.summarize_each_layer:
            avg_step_size = jnp.mean(jnp.abs(step))
            summary.summary(f"inv/{summary_prefix}/avg_step_size", avg_step_size)
            summary.summary(f"inv/{summary_prefix}/extra_step_mult", extra_step_mult)
            summary.summary(f"inv/{summary_prefix}/predicted_lr", lr)
            summary.summary(f"inv/{summary_prefix}/mean_abs_g", jnp.mean(jnp.abs(g)))
            summary.summary(
                f"inv/{summary_prefix}/g_norm", jnp.sqrt(jnp.mean(jnp.square(g)) + 1e-9)
            )
            summary.summary(f"inv/{summary_prefix}/param_scale", param_scale)
            # summary.summary(f"inv/{summary_prefix}/mean_m_abs", jnp.mean(jnp.abs(m)))
            # summary.summary(f"inv/{summary_prefix}/var_m_abs", jnp.std(jnp.abs(m)))
            # summary.summary(f"inv/{summary_prefix}/mean_rms", jnp.mean(rms))
            # summary.summary(f"inv/{summary_prefix}/var_rms", jnp.std(rms))

        new_p = p - step
        return new_p

    def make_tensor_features(self, p, g, m, rms):
        leading_axis = list(range(0, len(p.shape)))
        mean_m = jnp.mean(m, axis=leading_axis, keepdims=True)
        var_m = jnp.mean(jnp.square(m - mean_m), axis=leading_axis)
        # inputs["var_m"] = _clip_log_abs(var_m, scale=1.0)

        mean_rms = jnp.mean(rms, axis=leading_axis, keepdims=True)
        var_rms = jnp.mean(jnp.square(rms - mean_rms), axis=leading_axis)
        # inputs["mean_rms"] = _clip_log_abs(jnp.reshape(mean_rms, [mean_rms.shape[-1]]), scale=1.0)
        # inputs["var_rms"] = _clip_log_abs(var_rms, scale=1.0)
        tensor_feats = jnp.concatenate(
            [var_m.flatten(), mean_rms.flatten(), var_rms.flatten()], axis=0
        )
        return tensor_feats

    def lstm_features_for_tensor(
        self, p, tensor_feats, summary_prefix, tf_mean, tf_var, fraction_trained, loss_features
    ):
        inputs = {}

        # global info
        fraction_left = _fractional_tanh_embed(fraction_trained)
        inputs["fraction_left"] = fraction_left
        inputs["loss_features"] = loss_features

        # normalize tensor features by running mean and std
        tf_mean = tf_mean.flatten()
        tf_std = jax.lax.sqrt(tf_var.flatten())
        normalized_feats = (tensor_feats - tf_mean) / jnp.maximum(tf_std, 1e-9)
        # inputs["tensor_feats"] = jnp.clip(normalized_feats, -20.0, 20.0)

        # rank
        # n_rank = onp.sum(onp.asarray(p.shape) > 1)
        # inputs["rank"] = hk.one_hot(n_rank, 5)

        if self.summarize_each_layer:

            def summarize_dict(inputs):
                for k, v in inputs.items():
                    if len(v.shape) > 0:  # pylint: disable=g-explicit-length-test
                        for vi, vv in enumerate(v):
                            summary.summary(
                                f"per_tensor_feat/{summary_prefix}/{k}__{vi}",
                                vv,
                                aggregation="sample",
                            )
                    else:
                        summary.summary(
                            f"per_tensor_feat/{summary_prefix}/{k}", v, aggregation="sample"
                        )

            summarize_dict(inputs)
            info = {}
            info["raw_tensor_feats"] = tensor_feats
            info["normalized_tensor_feats"] = normalized_feats
            info["tf_mean"] = tf_mean
            info["tf_std"] = tf_std
            summarize_dict(info)

        values = _sorted_values(inputs)
        values = [v if len(v.shape) == 1 else jnp.expand_dims(v, 0) for v in values]

        # add the validation features at the end of the feature vector to make it
        # easier to do surgery into it.
        if self.with_validation_feature_dim:
            values.append(jnp.ones([1], dtype=jnp.float32) * self.validation_mode)

        return jnp.concatenate(values, axis=0)

    def init(self, key) -> lopt_base.MetaParams:
        r = 10
        c = 10
        p = jnp.ones([r, c])
        g = jnp.ones([r, c])

        m = jnp.ones([r, c, len(self.initial_momentum_decays)])
        rms = jnp.ones([r, c, len(self.initial_rms_decays)])
        fac_g = jnp.ones([r, c, len(self.initial_adafactor_decays)])
        fac_vec_row = jnp.ones([r, len(self.initial_adafactor_decays)])
        fac_vec_col = jnp.ones([c, len(self.initial_adafactor_decays)])
        fac_vec_v = jnp.ones([len(self.initial_adafactor_decays)])
        tf_feat = jnp.ones([len(self.initial_momentum_decays) + 2 * len(self.initial_rms_decays)])

        key1, key = jax.random.split(key)

        lstm_inital_state = hk.transform(lambda: self.lstm_fn().initial_state(1))[1](None, key1)

        loss_features = self.buffer_loss_fns.features(self.buffer_loss_fns.init(10))

        # figure out how may m and rms features there are by getting an opt state.
        output_shape = jax.eval_shape(
            self.lstm_features_for_tensor,
            p,
            tf_feat,
            0,
            tf_feat,
            tf_feat,
            fraction_trained=1.0,
            loss_features=loss_features,
        )

        assert len(output_shape.shape) == 1

        rnn_input_features = output_shape.shape[0]

        key1, key = jax.random.split(key)
        return {
            "lstm_init_state": lstm_inital_state,
            "rnn_params": self.rnn.init(
                key1, jnp.zeros([1, rnn_input_features]), lstm_inital_state
            ),
        }

    def opt_fn(self, theta, is_training=True) -> opt_base.Optimizer:
        parent = self

        class _Opt(opt_base.Optimizer):
            """Inner optimizer."""

            def __init__(self, theta):
                super().__init__()
                self.theta = theta

            @functools.partial(jax.jit, static_argnums=(0,))
            def init(self, params: Any, model_state=None, num_steps=None, key=None) -> State:
                mom_roll, rms_roll, adafac_roll = parent.accumulators_for_decays()
                tf_mean_roll, tf_var_roll = parent.per_tensor_accumulators()
                loss_buffer = parent.buffer_loss_fns.init(num_steps)

                n_states = len(jax.tree_util.tree_leaves(params))
                lstm_hidden_state = jax.tree_util.tree_map(
                    lambda x: jnp.tile(x, [n_states] + [1] * len(x.shape[1:])),
                    theta["lstm_init_state"],
                )

                # dummy tree to initialize tensor features running mean and std accumulators
                example_tree = jax.tree_util.tree_map(
                    parent.make_tensor_features, params, params, params, params
                )

                return State(
                    state=model_state,
                    rms_rolling=rms_roll.init(params),
                    mom_rolling=mom_roll.init(params),
                    fac_rolling=adafac_roll.init(params),
                    tf_mean_rolling=tf_mean_roll.init(example_tree),
                    tf_var_rolling=tf_var_roll.init(example_tree),
                    iteration=jnp.asarray(0, dtype=jnp.int32),
                    num_steps=jnp.asarray(num_steps, dtype=jnp.int32),
                    lstm_hidden_state=lstm_hidden_state,
                    loss_buffer=loss_buffer,
                )

            @functools.partial(jax.jit, static_argnums=(0,))
            def update(
                self,
                opt_state,
                grads,
                params,
                loss=None,
                model_state=None,
                is_valid=False,
                key=None,
            ) -> State:
                if parent.constant_loss:
                    loss = 1.0
                assert loss is not None
                summary.summary("validation_mode", parent.validation_mode)
                next_loss_buffer = parent.buffer_loss_fns.update(opt_state.loss_buffer, loss)
                to_lstm_from_loss = parent.buffer_loss_fns.features(next_loss_buffer)

                grads = jax.tree_util.tree_map(lambda x: jnp.clip(x, -1000.0, 1000.0), grads)
                # Run the LSTM to get params for ff.
                m = opt_state.mom_rolling.m
                rms = opt_state.rms_rolling.rms
                tf_mean = opt_state.tf_mean_rolling.m
                tf_var = opt_state.tf_var_rolling.m
                tensor_feats = jax.tree_util.tree_map(
                    parent.make_tensor_features, params, grads, m, rms
                )

                fraction_trained = opt_state.iteration / jnp.asarray(
                    opt_state.num_steps, dtype=jnp.float32
                )
                ff = functools.partial(
                    parent.lstm_features_for_tensor,
                    fraction_trained=fraction_trained,
                    loss_features=to_lstm_from_loss,
                )

                if parent.summarize_each_layer:
                    summary_prefix = tree_utils.map_named(lambda k, v: k, params)
                else:
                    summary_prefix = jax.tree_util.tree_map(lambda x: "None", params)

                rnn_inputs = jax.tree_util.tree_map(
                    ff, params, tensor_feats, summary_prefix, tf_mean, tf_var
                )
                stack = jnp.asarray(jax.tree_util.tree_leaves(rnn_inputs))

                lstm_hidden_state = opt_state.lstm_hidden_state

                _, lr_mult, next_lstm_hidden_state = parent.rnn.apply(
                    theta["rnn_params"], stack, lstm_hidden_state
                )
                lstm_hidden_state = next_lstm_hidden_state

                if parent.rnn_state_decay > 0.0:
                    lstm_hidden_state = tree_utils.tree_mul(
                        lstm_hidden_state, (1.0 - parent.rnn_state_decay)
                    )

                # update per tensor features running mean and var
                curr_mean = tensor_feats
                curr_var = jax.tree_util.tree_map(
                    lambda x, y: jnp.square(x.flatten() - y.flatten()), tensor_feats, tf_mean
                )
                tf_mean_roll, tf_var_roll = parent.per_tensor_accumulators()
                next_tf_mean_rolling = tf_mean_roll.update(opt_state.tf_mean_rolling, curr_mean)
                next_tf_var_rolling = tf_var_roll.update(opt_state.tf_var_rolling, curr_var)

                # one per param.
                # control_params = [d for d in control_params]
                # if parent.summarize_all_control:
                #     for pi, p in enumerate(control_params):
                #         summary.summary(f"control_param/{pi}", p, "tensor")
                struct = jax.tree_util.tree_structure(grads)
                lr_mult = struct.unflatten([lr for lr in lr_mult])

                # Run the FF
                mom_roll, rms_roll, adafac_roll = parent.accumulators_for_decays()
                next_mom_rolling = mom_roll.update(opt_state.mom_rolling, grads)
                next_rms_rolling = rms_roll.update(opt_state.rms_rolling, grads)
                next_adafac_rolling, fac_g = adafac_roll.update(opt_state.fac_rolling, grads)

                global_features = {
                    "iterations": opt_state.iteration,
                    "num_steps": opt_state.num_steps,
                }

                def apply_one(lr_mult, p, g, m, rms, fac_g, v_col, v_row, v, summary_prefix):
                    next_p = parent._ff_mod(
                        global_features,
                        lr_mult,
                        p,
                        g,
                        m=m,
                        rms=rms,
                        fac_g=fac_g,
                        fac_vec_col=v_col,
                        fac_vec_row=v_row,
                        fac_vec_v=v,
                        summary_prefix=summary_prefix,
                    )
                    return next_p

                next_params = jax.tree_util.tree_map(
                    apply_one,
                    lr_mult,
                    params,
                    grads,
                    next_mom_rolling.m,
                    next_rms_rolling.rms,
                    fac_g,
                    next_adafac_rolling.v_col,
                    next_adafac_rolling.v_row,
                    next_adafac_rolling.v_diag,
                    summary_prefix,
                )

                ss = State(
                    state=model_state,
                    mom_rolling=next_mom_rolling,
                    rms_rolling=next_rms_rolling,
                    fac_rolling=next_adafac_rolling,
                    tf_mean_rolling=next_tf_mean_rolling,
                    tf_var_rolling=next_tf_var_rolling,
                    iteration=opt_state.iteration + 1,
                    num_steps=opt_state.num_steps,
                    lstm_hidden_state=lstm_hidden_state,
                    loss_buffer=next_loss_buffer,
                )
                return next_params, tree_utils.match_type(ss, opt_state)

        return _Opt(theta)
