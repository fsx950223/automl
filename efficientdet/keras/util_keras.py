# Copyright 2020 Google Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Common keras utils."""
from typing import Text
from absl import logging
import tensorflow as tf
import utils


def build_batch_norm(is_training_bn: bool,
                     beta_initializer: Text = 'zeros',
                     gamma_initializer: Text = 'ones',
                     data_format: Text = 'channels_last',
                     momentum: float = 0.99,
                     epsilon: float = 1e-3,
                     strategy: Text = None,
                     name: Text = 'tpu_batch_normalization'):
  """Build a batch normalization layer.

  Args:
    is_training_bn: `bool` for whether the model is training.
    beta_initializer: `str`, beta initializer.
    gamma_initializer: `str`, gamma initializer.
    data_format: `str` either "channels_first" for `[batch, channels, height,
      width]` or "channels_last for `[batch, height, width, channels]`.
    momentum: `float`, momentume of batch norm.
    epsilon: `float`, small value for numerical stability.
    strategy: `str`, whether to use tpu, gpus or other version of batch norm.
    name: the name of the batch normalization layer

  Returns:
    A normalized `Tensor` with the same `data_format`.
  """
  axis = 1 if data_format == 'channels_first' else -1
  batch_norm_class = utils.batch_norm_class(is_training_bn, strategy)

  bn_layer = batch_norm_class(
      axis=axis,
      momentum=momentum,
      epsilon=epsilon,
      center=True,
      scale=True,
      beta_initializer=beta_initializer,
      gamma_initializer=gamma_initializer,
      name=name)

  return bn_layer


def get_ema_vars(model):
  """Get all exponential moving average (ema) variables."""
  ema_vars = model.trainable_weights
  for v in model.weights:
    # We maintain mva for batch norm moving mean and variance as well.
    if 'moving_mean' in v.name or 'moving_variance' in v.name:
      ema_vars.append(v)
  ema_vars_dict = dict()
  # Remove duplicate vars
  for var in ema_vars:
    ema_vars_dict[var.ref()] = var
  return ema_vars_dict


def _assign_variable(key, var, ckpt_path_or_file, skip_mismatch=True):
  try:
    var.assign(tf.train.load_variable(ckpt_path_or_file, key))
  except tf.errors.NotFoundError as e:
    if skip_mismatch:
      logging.warning('Not found %s in %s', key, ckpt_path_or_file)
    else:
      raise e
  except ValueError as e:
    if skip_mismatch:
      logging.warning('%s: %s', key, e)
    else:
      raise e


def restore_ckpt(model, ckpt_path_or_file, ema_decay=0.9998, skip_mismatch=True):
  """Restore variables from a given checkpoint.

  Args:
    model: the keras model to be restored.
    ckpt_path_or_file: the path or file for checkpoint.
    ema_decay: ema decay rate. If None or zero or negative value, disable ema.
  """
  if ckpt_path_or_file == '_':
    logging.info('Running test: do not load any ckpt.')
    return
  if tf.io.gfile.isdir(ckpt_path_or_file):
    ckpt_path_or_file = tf.train.latest_checkpoint(ckpt_path_or_file)

  if (tf.train.list_variables(ckpt_path_or_file)[0][0] ==
      '_CHECKPOINTABLE_OBJECT_GRAPH'):
    model.load_weights(ckpt_path_or_file)
  else:
    if ema_decay > 0:
      ema = tf.train.ExponentialMovingAverage(decay=0.0)
      optimizer = model.optimizer
      if isinstance(optimizer, tf.keras.mixed_precision.experimental.LossScaleOptimizer):
        optimizer = optimizer._optimizer
      var_list = model._freeze_vars()
      optimizer._create_all_weights(var_list)
      average_weights = [weight for weight in optimizer.weights if weight.name.split('/')[-1].find('average')>=0]
      for var in average_weights:
        key = '/'.join(var.name.split('/')[:-1] + [ema.name])
        _assign_variable(key, var, ckpt_path_or_file, skip_mismatch)

    ema_vars = get_ema_vars(model)
    var_dict = {
        var.name.split(':')[0]: var for (ref, var) in ema_vars.items()
    }
    # add variables that not in var_dict
    for v in model.weights:
      if v.ref() not in ema_vars:
        var_dict[v.name.split(':')[0]] = v
    # try to load graph-based checkpoint with ema support,
    # else load checkpoint via keras.load_weights which doesn't support ema.
    for key, var in var_dict.items():
      _assign_variable(key, var, ckpt_path_or_file, skip_mismatch)
