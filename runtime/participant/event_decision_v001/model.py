"""Joint temporal and trajectory event classifier; three explicit classes."""
import torch
from torch import nn

CLASSES = ("table_contact", "racket_hit", "background")


class EventDecisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Conv1d(23, 32, 1)
        self.layers = nn.ModuleList(nn.Conv1d(32, 32, 3, padding=d, dilation=d) for d in (1, 2, 4))
        self.temporal = nn.Linear(32, 3)
        self.summary = nn.Sequential(nn.Linear(44, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, sequence, summary):
        h = self.project(sequence.transpose(1, 2))
        for layer in self.layers:
            h = h + torch.nn.functional.silu(layer(h))
        return (self.temporal(h[:, :, 7]) + self.summary(summary)) * 0.5


def event_interval_nll(log_probabilities, rows, event_class):
    """Exactly one annotated-class contact in a complete disjoint event bag.

    Other candidates in this interval represent alternative frame assignments,
    not extra copies of the event. All assignments are marginalized. Callers
    exclude unknown and intersecting intervals instead of trimming the labels.
    """
    selected = log_probabilities[rows]
    background = selected[:, 2]
    return -(background.sum() + torch.logsumexp(selected[:, event_class] - background, dim=0))
