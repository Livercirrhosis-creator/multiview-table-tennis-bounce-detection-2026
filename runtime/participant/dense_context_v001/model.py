"""Classify each native center using its own encoding and reliable context."""
import torch
from torch import nn

CLASSES = ("table_contact", "racket_hit", "background")


class DenseContextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Conv1d(23, 32, 1)
        self.layers = nn.ModuleList(nn.Conv1d(32, 32, 3, padding=d, dilation=d) for d in (1, 2, 4))
        self.summary = nn.Sequential(nn.Linear(44, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(288, 64), nn.ReLU(), nn.Linear(64, 3))

    def encode(self, sequence, summary):
        h = self.project(sequence.transpose(1, 2))
        for layer in self.layers:
            h = h + torch.nn.functional.silu(layer(h))
        return torch.cat((h[:, :, 7], self.summary(summary)), dim=1)

    def candidate_logits(self, embeddings, member_rows, center_rows):
        """A shared pool per identical member set; no window identity input."""
        if len(member_rows) != len(center_rows):
            raise ValueError("Each candidate needs one center and reliable context")
        if not member_rows:
            return embeddings.new_empty((0, 3))
        unique, positions, mapping = [], {}, []
        for members, center in zip(member_rows, center_rows):
            key = tuple(sorted(set(members)))
            if not key or center not in key:
                raise ValueError("Reliable context must include its candidate center")
            if key not in positions:
                positions[key] = len(unique)
                unique.append(key)
            mapping.append(positions[key])
        width = max(len(rows) for rows in unique)
        indices = torch.tensor([list(rows)+[rows[0]]*(width-len(rows)) for rows in unique],
                               dtype=torch.long,device=embeddings.device)
        lengths = torch.tensor([len(rows) for rows in unique],dtype=torch.long,device=embeddings.device)
        valid = torch.arange(width,device=embeddings.device)[None,:] < lengths[:,None]
        values = embeddings[indices]
        mean = (values*valid[:,:,None]).sum(1)/lengths[:,None]
        maximum = values.masked_fill(~valid[:,:,None],-torch.inf).amax(1)
        mapping = torch.tensor(mapping,dtype=torch.long,device=embeddings.device)
        centers = torch.tensor(center_rows,dtype=torch.long,device=embeddings.device)
        return self.head(torch.cat((embeddings[centers],mean[mapping],maximum[mapping]),dim=1))

    def forward(self, sequence, summary, member_rows, center_rows):
        return self.candidate_logits(self.encode(sequence, summary), member_rows, center_rows)
