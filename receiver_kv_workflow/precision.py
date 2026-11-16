"""Optional output-projection precision control for numerical diagnostics."""
import torch
from torch import nn
from torch.nn import functional as F


class FP32OutputHead(nn.Module):
    """Project in FP32, while leaving transformer/embedding/cache dtypes alone."""
    def __init__(self, original):
        super().__init__()
        if not isinstance(original, nn.Linear):
            raise TypeError("FP32 output control requires an ordinary Linear LM head")
        # Create a separate parameter, so tied input embeddings are not upcast.
        # Upcasting cannot restore precision already lost in checkpoint weights.
        self.weight = nn.Parameter(original.weight.detach().to(dtype=torch.float32), requires_grad=False)
        self.bias = (nn.Parameter(original.bias.detach().to(dtype=torch.float32), requires_grad=False)
                     if original.bias is not None else None)
        self.last_output_dtype = None

    def forward(self, hidden):
        # Disable autocast and TF32 for this projection only. Restore global
        # matmul settings before any later attention/embedding computation.
        cuda = hidden.device.type == "cuda"
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32 if cuda else None
        try:
            if cuda:
                torch.backends.cuda.matmul.allow_tf32 = False
            with torch.autocast(device_type=hidden.device.type, enabled=False):
                output = F.linear(hidden.float(), self.weight, self.bias)
            self.last_output_dtype = str(output.dtype)
            return output
        finally:
            if cuda:
                torch.backends.cuda.matmul.allow_tf32 = previous_tf32


def enable_fp32_lm_head(model):
    model.set_output_embeddings(FP32OutputHead(model.get_output_embeddings()).eval())
