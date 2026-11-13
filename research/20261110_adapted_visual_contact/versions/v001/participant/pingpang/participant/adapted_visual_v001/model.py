"""Task-adapted local RGB evidence with a permanently frozen ImageNet stem."""

import hashlib
from pathlib import Path

import torch
from torch import nn
from torchvision.models import mobilenet_v3_small


CLASSES = ("table_contact", "racket_hit", "background")
RGB_SHAPE = (3, 2, 3, 96, 96)
STEM_SHAPE = (3, 2, 96, 3, 3)
STEM_END = 10
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
WEIGHTS_SHA256 = "047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f"


class ContactVisualModel(nn.Module):
    def __init__(self, weights_path=None):
        """No implicit downloads; absent weights are for structure tests/loading only."""
        super().__init__()
        complete = mobilenet_v3_small(weights=None)
        if weights_path is not None:
            path = Path(weights_path)
            if hashlib.sha256(path.read_bytes()).hexdigest() != WEIGHTS_SHA256:
                raise ValueError("Fixed ImageNet MobileNetV3-small weights SHA256 differs")
            complete.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        self.stem = complete.features[:STEM_END]
        self.tail = complete.features[STEM_END:]
        self.spatial = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(576, 24), nn.SiLU())
        self.contact = nn.Sequential(nn.Linear(5 * 2 * 24 + 6, 32), nn.SiLU(), nn.Linear(32, 3))
        self.register_buffer("image_mean", torch.tensor(MEAN).reshape(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor(STD).reshape(1, 3, 1, 1))
        self.stem.requires_grad_(False)
        for layer in self.modules():
            if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                layer.requires_grad_(False)
        self.train(True)

    def train(self, mode=True):
        super().train(mode)
        self.stem.eval()
        # The same cached stem is safe for all folds only while its state stays
        # fixed. Tail BN is frozen too: source batches cannot change its priors.
        for layer in self.modules():
            if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                layer.eval()
        return self

    @torch.no_grad()
    def extract_stem(self, rgb):
        """Return float32 N/time/scale/channel/height/width features."""
        if rgb.ndim != 6 or tuple(rgb.shape[1:]) != RGB_SHAPE or rgb.dtype != torch.uint8:
            raise ValueError("Expected uint8 RGB in candidate/time/scale/channel/height/width order")
        pixels = rgb.reshape(-1, 3, 96, 96).float().div(255.)
        values = self.stem((pixels - self.image_mean) / self.image_std)
        return values.reshape(len(rgb), *STEM_SHAPE)

    def forward_stem(self, stem_features, time_features):
        if (stem_features.ndim != 6 or tuple(stem_features.shape[1:]) != STEM_SHAPE
                or stem_features.dtype != torch.float32):
            raise ValueError("Expected float32 N/time/scale/96/3/3 frozen stem features")
        if time_features.shape != (len(stem_features), 3, 2):
            raise ValueError("Expected relative seconds and availability for each time slot")
        frames = self.spatial(self.tail(stem_features.reshape(-1, 96, 3, 3)))
        frames = frames.reshape(len(stem_features), 3, 48)
        time_features = time_features.to(dtype=frames.dtype, device=frames.device)
        valid = time_features[:, :, 1:2]
        frames = frames * valid
        changes = (frames[:, 1:] - frames[:, :-1]) * valid[:, 1:] * valid[:, :-1]
        times = torch.cat((time_features[:, :, :1] * valid / .08, valid), dim=-1)
        return self.contact(torch.cat((frames.flatten(1), changes.flatten(1), times.flatten(1)), dim=1))

    def forward(self, rgb, time_features):
        return self.forward_stem(self.extract_stem(rgb), time_features)

    def optimizer_groups(self, tail_lr=1e-4, head_lr=1e-3):
        return [
            dict(params=[p for p in self.tail.parameters() if p.requires_grad], lr=tail_lr, name="tail"),
            dict(params=list(self.spatial.parameters()) + list(self.contact.parameters()), lr=head_lr, name="head"),
        ]
