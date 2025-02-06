from composer.core import Algorithm, Event, State
import logging
from typing import Type, Optional
import torch

from composer.core import Event, State
from composer.loggers import Logger

log = logging.getLogger(__name__)

__all__ = ['SparsityMask']


def _attach_masks(
        module: torch.nn.Module,
        layer_type: Type[torch.nn.Module] = torch.nn.Linear,
        threshold_for_masking: float = 0.1
    ) -> None:
    """
    Recursively attaches sparsity masks to layers that meet the sparsity threshold.

    Args:
        module: The parent module to process
        layer_type: The type of layer to apply masks to (default: nn.Linear)
        threshold_for_masking: Minimum sparsity ratio required to apply a mask
    """
    for name, submodule in module.named_children():
        if isinstance(submodule, layer_type):
            print(f"ELDAR DEBUG: {name} is a {layer_type} and has a weight of shape {submodule.weight.shape}")
            sparsity = torch.sum(submodule.weight == 0).item() / submodule.weight.numel()
            if sparsity >= threshold_for_masking:
                mask = torch.where(
                    submodule.weight == 0,
                    torch.tensor(0, dtype=torch.uint8, device=submodule.weight.device),
                    torch.tensor(1, dtype=torch.uint8, device=submodule.weight.device)
                )
                submodule.register_buffer("mask", mask, persistent=False)
                log.info(
                    f"Attached sparsity mask to {name} with sparsity = "
                    f"{torch.sum(mask == 0).item() / mask.numel():.2f}"
                )
        else:
            _attach_masks(submodule, layer_type, threshold_for_masking)


@torch.no_grad()
def _mask_weights(module: torch.nn.Module) -> None:
    """
    Applies the sparsity mask to the module's weights if a mask exists.

    Args:
        module: The module whose weights should be masked
    """
    if hasattr(module, 'mask'):
        module.weight.data *= module.mask


class SparsityMask(Algorithm):
    def __init__(
            self,
            threshold_for_masking: float = 0.1,
            targets: object = None, #TODO: right now it is not doing anything, we grab all nn.Linear
        ):
        super().__init__()

        if not 0 <= threshold_for_masking <= 1:
            raise ValueError("Threshold for masking must be between 0 and 1")

        self.threshold_for_masking = threshold_for_masking
        self.targets = targets

    def match(self, event: Event, state: State) -> bool:
        return event in [Event.INIT, Event.BATCH_END]

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        if event == Event.INIT:
            _attach_masks(state.model, torch.nn.Linear, self.threshold_for_masking)
        elif event == Event.BATCH_END:
            state.model.apply(_mask_weights)
