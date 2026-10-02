"""Reconstruct the large/large LANet described by the reference log."""

import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
import torch.nn.functional as F

from code_lag.model_wrapper import FeatureExtract, preprocess_fft, standardize
from code_lag.model_mlp import conv_layer, StatsPooling


CROP_LENGTHS = (2048, 4096, 8192, 16384, 32768)
REFERENCE_PARAMETERS = 3238312


class TrainingFeatureExtract(FeatureExtract):
    def _stand_randn(self, data, test_len=None):
        lrb, data = torch.split(data, [2, 32768], dim=1)
        if self.training:
            amplitude = self.min_value + torch.rand(1, device=data.device) * (
                self.max_value - self.min_value
            )
            data = data + torch.randn_like(data) * amplitude
            crop_candidates = torch.tensor(
                [2048, 4096, 8192, 16384, 32768], device=data.device
            )
            crop_len = int(crop_candidates[
                torch.randint(0, len(crop_candidates), (1,), device=data.device)
            ].item())
            if crop_len < 32768:
                start = min(
                    int(torch.randint(0, 16380, (1,), device=data.device).item()),
                    32768 - crop_len,
                )
                data = data[:, start:start + crop_len]
        elif test_len is not None:
            if test_len not in CROP_LENGTHS:
                raise ValueError(f"Unsupported evaluation length: {test_len}")
            data = data[:, :test_len]
        self.last_crop_length = data.shape[-1]
        return lrb.real, data


class LargeBranch(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        channels = (in_channels, 32, 48, 64, 96, 128)
        for i in range(1, 6):
            setattr(self, f"conv{i}", conv_layer(channels[i - 1], channels[i], 23))
        self.avg_conv = nn.Conv1d(128, 256, 23, padding=11)
        self.avg_bn = nn.BatchNorm1d(256, affine=False)
        self.stats_pool = StatsPooling()
        self.segment = nn.Sequential(nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU())

    def forward(self, x):
        for i in range(1, 6):
            x = getattr(self, f"conv{i}")(x)
        x = F.relu(self.avg_bn(self.avg_conv(x)))
        return self.segment(self.stats_pool(x))


class LengthConditionedGMUFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_iq = nn.Sequential(nn.Linear(256, 256), nn.Tanh(), nn.Dropout(0.1))
        self.proj_cc = nn.Sequential(nn.Linear(256, 256), nn.Tanh(), nn.Dropout(0.1))
        self.gate = nn.Sequential(nn.Linear(513, 256), nn.Sigmoid())
        self.head = nn.Sequential(nn.Linear(256, 512), nn.GELU(), nn.Linear(512, 8))

    def forward(self, e_iq, e_cc, length):
        ell = e_iq.new_full((e_iq.shape[0], 1), math.log2(length) / math.log2(32768))
        h_iq, h_cc = self.proj_iq(e_iq), self.proj_cc(e_cc)
        weight = self.gate(torch.cat((e_iq, e_cc, ell), dim=1))
        return self.head(weight * h_cc + (1.0 - weight) * h_iq)


class LANet(nn.Module):
    def __init__(self, noise_max=0.3, iq_standardize=True):
        super().__init__()
        self.preprocess = TrainingFeatureExtract({"noise_max": noise_max, "boi_thre": 14.0})
        self.preprocess.boi_utp_en = True
        self.backbone_iq = LargeBranch(2)
        self.backbone_cc = LargeBranch(8)
        self.lag = LengthConditionedGMUFusion()
        self.iq_standardize = iq_standardize

    def forward(self, x, test_len=None):
        x = self.preprocess(x, test_len)
        iq = torch.stack((x.real, x.imag), dim=1).float()
        if self.iq_standardize:
            iq = standardize(iq, dim=-1)
        cc = standardize(preprocess_fft(x), dim=-1)
        logits = self.lag(self.backbone_iq(iq), self.backbone_cc(cc), x.shape[-1])
        return F.log_softmax(logits, dim=1)


def verify_reference_structure(model, checkpoint_path):
    """Compare tensor names/shapes only; never load reference weights into model."""
    reference = torch.load(checkpoint_path, map_location="cpu", weights_only=True)["model_state_dict"]
    expected = {key: tuple(value.shape) for key, value in reference.items()}
    actual = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    if actual != expected:
        missing = sorted(expected.keys() - actual.keys())
        extra = sorted(actual.keys() - expected.keys())
        wrong = [key for key in actual.keys() & expected.keys() if actual[key] != expected[key]]
        raise ValueError(f"Reference structure mismatch: missing={missing}, extra={extra}, shapes={wrong}")
    count = sum(p.numel() for p in model.parameters())
    if count != REFERENCE_PARAMETERS or not all(p.requires_grad for p in model.parameters()):
        raise ValueError(f"Invalid trainable parameter count: {count}")
    return {"state_tensor_count": len(actual), "trainable_parameters": count, "weights_loaded": False}
