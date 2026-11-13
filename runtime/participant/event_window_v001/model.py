"""Shared trajectory encoder; window presence and conditional native location."""
import torch
from torch import nn

CLASSES = ("table_contact", "racket_hit", "background")


class WindowEventModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Conv1d(23, 32, 1)
        self.layers = nn.ModuleList(
            nn.Conv1d(32, 32, 3, padding=d, dilation=d) for d in (1, 2, 4))
        self.summary = nn.Sequential(nn.Linear(44, 64), nn.ReLU())
        self.presence = nn.Sequential(nn.Linear(192, 64), nn.ReLU(), nn.Linear(64, 3))
        self.location = nn.Linear(96, 2)

    def encode(self, sequence, summary):
        h = self.project(sequence.transpose(1, 2))
        for layer in self.layers:
            h = h + torch.nn.functional.silu(layer(h))
        return torch.cat((h[:, :, 7], self.summary(summary)), dim=1)

    def window_logits(self, embeddings, member_rows):
        """Pool actual members only; padding contributes neither mean nor max."""
        if not member_rows:
            return embeddings.new_empty((0, 3))
        lengths = [len(rows) for rows in member_rows]
        if min(lengths) == 0:
            raise ValueError("A window must contain at least its native center")
        width = max(lengths)
        indices = torch.tensor([list(rows) + [rows[0]] * (width - len(rows))
                                for rows in member_rows], dtype=torch.long,
                               device=embeddings.device)
        lengths = torch.tensor(lengths, dtype=torch.long, device=embeddings.device)
        valid = torch.arange(width, device=embeddings.device)[None, :] < lengths[:, None]
        values = embeddings[indices]
        mean = (values * valid[:, :, None]).sum(1) / lengths[:, None]
        maximum = values.masked_fill(~valid[:, :, None], -torch.inf).amax(1)
        return self.presence(torch.cat((mean, maximum), dim=1))

    def forward(self, sequence, summary, member_rows):
        embeddings = self.encode(sequence, summary)
        return self.window_logits(embeddings, member_rows), self.location(embeddings)
