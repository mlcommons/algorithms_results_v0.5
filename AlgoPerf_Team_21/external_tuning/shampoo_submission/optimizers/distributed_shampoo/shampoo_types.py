"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

"""

import enum
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch.nn.parameter import Parameter

# Keys for optimizer state (always checkpointed)
FILTERED_GRAD = "filtered_grad"
MOMENTUM = "momentum"
STEP = "step"

# Keys for parameter groups (checkpointed if specified)
BETAS = "betas"
EPSILON = "epsilon"
EXPONENT_MULTIPLIER = "exponent_multiplier"
GRAFTING_CONFIG = "grafting_config"
INV_ROOT_OVERRIDE = "inv_root_override"
LR = "lr"
MAX_PRECONDITIONER_DIM = "max_preconditioner_dim"
PARAMS = "params"  # While this is stored in groups by default, we do not checkpoint this quantity.
PRECONDITION_FREQUENCY = "precondition_frequency"
PRECONDITIONER_DTYPE = "preconditioner_dtype"
START_PRECONDITIONING_STEP = "start_preconditioning_step"
USE_BIAS_CORRECTION = "use_bias_correction"
USE_DECOUPLED_WEIGHT_DECAY = "use_decoupled_weight_decay"
USE_EMA_MOMENTUM = "use_ema_momentum"
USE_MERGE_DIMS = "use_merge_dims"
USE_NADAM = "use_nadam"
USE_NESTEROV = "use_nesterov"
USE_NORMALIZED_GRAFTING = "use_normalized_grafting"
WEIGHT_DECAY = "weight_decay"

# Keys for lists of blocked states and metadata (never checkpointed)
DISTRIBUTOR = "distributor"
FILTERED_GRAD_LIST = "filtered_grad_list"
GRAFTING_PRECONDITIONER_LIST = "grafting_preconditioner_list"
MASKED_BLOCKED_GRADS = "masked_blocked_grads"
MASKED_BLOCKED_PARAMS = "masked_blocked_params"
MASKED_FILTERED_GRAD_LIST = "masked_filtered_grad_list"
MASKED_MOMENTUM_LIST = "masked_momentum_list"
MOMENTUM_LIST = "momentum_list"
PREVIOUS_GRAD_SELECTOR = "previous_grad_selector"
SHAMPOO_PRECONDITIONER_LIST = "shampoo_preconditioner_list"


###### ENUM CLASSES ######
class CommunicationDType(enum.Enum):
    DEFAULT = 0
    FP16 = 1
    BF16 = 2
    FP32 = 3


###### DATACLASSES ######
@dataclass
class FSDPParameterMetadata:
    """FSDP Metadata for a parameter.

    Args:
        fqn (str): Fully qualified name of the parameter.
        shape (torch.Size): Shape of the parameter.
        numel (int): Number of elements in the parameter.
        start_idx (int): Start index of the local shard in the flattened parameter (inclusive).
        end_idx (int): End index of the local shard in the flattened parameter (exclusive).

    """

    fqn: str
    shape: torch.Size
    numel: int
    start_idx: int
    end_idx: int


@dataclass
class AbstractDataclass:
    def __new__(cls, *args: Any, **kwargs: Any) -> Optional["AbstractDataclass"]:
        if cls == AbstractDataclass or cls.__bases__[0] == AbstractDataclass:
            raise TypeError(f"Cannot instantiate abstract class: {cls.__name__}.")
        return super().__new__(cls)


@dataclass
class DistributedConfig(AbstractDataclass):
    """Abstract dataclass for distributed configs in Shampoo."""

    ...


@dataclass
class DDPShampooConfig(DistributedConfig):
    """Configuration for DDP Shampoo.

    Enables distributed computation and optimizer states (like ZeRO-1) via DTensor for Shampoo.

    Args:
        communication_dtype (CommunicationDType): Data type for communication between ranks. (Default: DEFAULT)
        num_trainers_per_group (int): Number of GPUs per distributed process group for distributed computation/memory.
            If num_trainers_per_group = -1 is used, then defaults to using the LOCAL_WORLD_SIZE. (Default: -1)
        communicate_params (bool): Flag for all-gathering updated params across multiple workers.
            If False, all-gathers parameter updates across multiple workers. (Default: False)

    """

    communication_dtype: CommunicationDType = CommunicationDType.DEFAULT
    num_trainers_per_group: int = -1
    communicate_params: bool = False


@dataclass
class FSDPShampooConfig(DistributedConfig):
    """Configuration for FSDP Shampoo.

    Passes in additional metadata necessary to run FSDP Shampoo.

    Args:
        param_to_metadata (Dict[Parameter, FSDPParameterMetadata]): Dictionary mapping parameter to its metadata from FSDP.

    """

    param_to_metadata: Dict[Parameter, FSDPParameterMetadata]


@dataclass
class GraftingConfig(AbstractDataclass):
    """Abstract dataclass for grafting configurations in Shampoo."""

    ...


@dataclass
class SGDGraftingConfig(GraftingConfig):
    """Configuration for grafting from SGD."""

    ...


@dataclass
class AdaGradGraftingConfig(GraftingConfig):
    """Configuration for grafting from AdaGrad.

    Args:
        epsilon (float): Epsilon term for regularizing square-root of the aggregated second moment to ensure positive definiteness.
            (Default: 1e-10)

    """

    epsilon: float = 1e-10

    def __post_init__(self) -> None:
        super().__init__()
        if not self.epsilon > 0.0:
            raise ValueError(f"Invalid epsilon value: {self.epsilon}. Must be > 0.0.")


@dataclass
class RMSpropGraftingConfig(GraftingConfig):
    """Configuration for grafting from RMSprop.

    Args:
        beta2 (float): Exponential moving average factor for second moment. (Default: 0.99)
        epsilon (float): Epsilon term for regularizing square-root of the second moment to ensure positive definiteness.
            (Default: 1e-8)

    """

    beta2: float = 0.99
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        super().__init__()
        if not 0.0 < self.beta2 <= 1.0:
            raise ValueError(
                f"Invalid grafting beta2 parameter: {self.beta2}. Must be in (0.0, 1.0]."
            )
        if not self.epsilon > 0.0:
            raise ValueError(f"Invalid epsilon value: {self.epsilon}. Must be > 0.0.")


@dataclass
class AdamGraftingConfig(GraftingConfig):
    """Configuration for grafting from Adam.

    Args:
        beta2 (float): Exponential moving average factor for second moment. (Default: 0.999)
        epsilon (float): Epsilon term for regularizing square-root of the second moment to ensure positive definiteness.
            (Default: 1e-8)

    """

    beta2: float = 0.999
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        super().__init__()
        if not 0.0 < self.beta2 <= 1.0:
            raise ValueError(
                f"Invalid grafting beta2 parameter: {self.beta2}. Must be in (0.0, 1.0]."
            )
        if not self.epsilon > 0.0:
            raise ValueError(f"Invalid epsilon value: {self.epsilon}. Must be > 0.0.")


@dataclass
class RWSAdaGradGraftingConfig(GraftingConfig):
    """Configuration for grafting from row-wise AdaGrad / Adam.

    NOTE: We merge both Row-Wise AdaGrad, RMSprop, and Adam into this configuration.

    Args:
        beta2 (float): Exponential moving average factor for second moment. (Default: 0.999)
        epsilon (float): Epsilon term for regularizing square-root of the second moment to ensure positive definiteness.
            (Default: 1e-8)
        use_bias_correction (bool): Uses bias correction. (Default: True)


    """

    beta2: float = 0.999
    epsilon: float = 1e-8
    use_bias_correction: bool = True

    def __post_init__(self) -> None:
        super().__init__()
        if not 0.0 < self.beta2 <= 1.0:
            raise ValueError(
                f"Invalid grafting beta2 parameter: {self.beta2}. Must be in (0.0, 1.0]."
            )
        if not self.epsilon > 0.0:
            raise ValueError(f"Invalid epsilon value: {self.epsilon}. Must be > 0.0.")
