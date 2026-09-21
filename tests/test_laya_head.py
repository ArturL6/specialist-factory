import torch

from specialist_factory.models.modernbert_laya import LayaStyleDecisionHead


def test_laya_style_head_scores_class_markers():
    head = LayaStyleDecisionHead(hidden_size=64, layers=1)
    hidden = torch.randn(2, 8, 64)
    mask = torch.ones(2, 8, dtype=torch.long)
    logits = head(hidden, mask, torch.tensor([[2, 4, 6], [1, 3, 5]]))
    assert logits.shape == (2, 3)
