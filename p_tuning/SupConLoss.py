import torch
import torch.nn as nn

class SupConLoss(nn.Module):
    def __init__(self, device):
        super(SupConLoss, self).__init__()
        self.device = device
        self.temperature = 1.0
    def forward(self, logits, t_label, i_targets):
        mask_array = []
        for t in t_label:
            row = []
            for i in i_targets:
                row.append(1.0 if t == i else 0.0)  # 相等则设为 1.0，不等则设为 0.0
            mask_array.append(row)
        mask = torch.tensor(mask_array).to(self.device)
        # for numerical stability
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - mean_log_prob_pos.mean()

        return loss