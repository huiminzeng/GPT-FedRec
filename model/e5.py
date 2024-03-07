import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

def average_pool(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

class E5Model(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()

        self.model = AutoModel.from_pretrained('intfloat/e5-base-v2')
        self.tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-base-v2')

        self.device = device
