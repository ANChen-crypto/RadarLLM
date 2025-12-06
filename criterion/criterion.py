import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class MAECriterion(_Loss):
    def __init__(self):
        super().__init__()

    def forward(self, model, sample, reduce=True):
        results = model(**sample)

        reduction = "mean" if reduce else "sum"
        loss = F.mse_loss(results["pred"], results["target"], reduction=reduction)

        sample_size = results["target"].size(0)

        results = {
            "loss": loss,
            "sample_size": sample_size,
        }

        return results