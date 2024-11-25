from zennit.core import Canonizer
from zennit.types import Convolution, Linear, Additive, Container

class ReplaceAttentionCanonizer(Canonizer):
    """
    Canonizer for vision transformer models to replace attention layers with linear approximations.
    """
    def __init__(self):
        super().__init__()
        self.rules = [
            (Convolution, Linear),
            (Additive, Linear),
        ]

    def apply(self, model):
        """
        Apply the canonizer to modify the model for relevance propagation.

        Args:
            model (torch.nn.Module): The model to apply the canonizer on.
        """
        for name, module in model.named_modules():
            for rule in self.rules:
                if isinstance(module, rule[0]):
                    setattr(model, name, rule[1]())
