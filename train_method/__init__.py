"""Training strategy dispatch for unrolled deblurring.

Available strategies (``stage_wise_train`` config key):
  - end2end:          standard end-to-end backprop
  - one_then_another: train one denoiser per fixed epoch budget
  - gradual_in_epoch: T independent forward-backward passes per batch
  - gradually_freeze: all denoisers → progressively freeze earlier ones

Optional post-training phase (``tail_align`` config key):
  - tail_align: fine-tune all denoisers with small LR for distribution alignment
"""

from .common import (
    TrainContext,
    unwrap_model,
    freeze_denoisers_except,
    freeze_denoisers_up_to,
    unfreeze_all_denoisers,
    get_active_stage,
    get_freeze_boundary,
    build_per_stage_optimizers,
    build_per_stage_schedulers,
)

from .end2end import train_one_epoch_end2end
from .one_then_another import (
    setup_one_then_another,
    train_one_epoch_one_then_another,
)
from .gradual_in_epoch import train_one_epoch_gradual_in_epoch
from .gradually_freeze import (
    setup_gradually_freeze,
    train_one_epoch_gradually_freeze,
)
from .tail_align import run_tail_align

VALID_MODES = frozenset({
    "end2end",
    "one_then_another",
    "gradual_in_epoch",
    "gradually_freeze",
})

__all__ = [
    "TrainContext",
    "VALID_MODES",
    # mode entry points
    "train_one_epoch_end2end",
    "setup_one_then_another",
    "train_one_epoch_one_then_another",
    "train_one_epoch_gradual_in_epoch",
    "setup_gradually_freeze",
    "train_one_epoch_gradually_freeze",
    "run_tail_align",
    # helpers
    "unwrap_model",
    "freeze_denoisers_except",
    "freeze_denoisers_up_to",
    "unfreeze_all_denoisers",
    "get_active_stage",
    "get_freeze_boundary",
    "build_per_stage_optimizers",
    "build_per_stage_schedulers",
]
