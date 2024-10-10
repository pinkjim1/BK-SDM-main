import torch
import torch.nn as nn

class SupConLoss(nn.Module):
    def __init__(self, device):
        super(SupConLoss, self).__init__()
        self.device = device
        self.temperature = 1.0
    def forward(self, logits):
        batch_size = logits.size(0)
        mask = torch.ones(batch_size, batch_size, dtype=torch.float32).to(self.device).detach()
        # for numerical stability
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - mean_log_prob_pos.mean()

        return loss