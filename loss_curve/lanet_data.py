"""Read the original CSPB.ML split into RAM, without a disk dataset cache."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


MODULATIONS = ("bpsk", "qpsk", "8psk", "dqpsk", "msk", "16qam", "64qam", "256qam")
SIGNAL_LENGTH = 32768
PER_BATCH = 4000


class CSPBDataset(Dataset):
    def __init__(self, root, batch_ids, limit=0):
        root = Path(root)
        indices = np.concatenate([
            np.arange((batch - 1) * PER_BATCH, batch * PER_BATCH) for batch in batch_ids
        ])
        if limit:
            if limit < 8 or limit % 8 or limit > len(indices):
                raise ValueError("Sample limit must be a positive multiple of 8 within the split.")
            indices = indices[:limit]
        label_path = root / "CSPB_ML_2022_Signal_Truth_Labels.txt"
        bounds, targets = [], []
        with label_path.open() as stream:
            for row, line in enumerate(stream):
                fields = line.split()
                if len(fields) != 9 or int(fields[0]) != row + 1:
                    raise ValueError(f"Unexpected truth-label format at row {row + 1}")
                label = MODULATIONS.index(fields[1])
                if label != row % 8:
                    raise ValueError(f"Class order differs from the original cyclic labels at row {row + 1}")
                symb = float(fields[6]) / float(fields[2]) / float(fields[5])
                offset = float(fields[3])
                bounds.append((-symb + offset + 0.5, symb + offset + 0.5))
                targets.append(label)
        if len(bounds) != 28 * PER_BATCH:
            raise ValueError(f"Expected 112000 label rows, got {len(bounds)}")

        # Preserve the original float16 bounds, then cast to complex64.
        bounds = np.asarray(bounds, dtype=np.float16).astype(np.complex64)
        samples = np.empty((len(indices), SIGNAL_LENGTH + 2), dtype=np.complex64)
        samples[:, :2] = bounds[indices]
        for position, index in enumerate(indices):
            batch = index // PER_BATCH + 1
            path = root / f"Batch_Dir_{batch}" / f"signal_{index + 1}.tim"
            raw = np.fromfile(path, dtype=np.float32)
            if raw.size != 2 + 2 * SIGNAL_LENGTH:
                raise ValueError(f"Unexpected signal size ({raw.size}) in {path}")
            samples[position, 2:] = raw[2::2] + 1j * raw[3::2]
            if (position + 1) % 4000 == 0:
                print(f"Loaded {position + 1}/{len(indices)} signals into RAM", flush=True)
        self.data = torch.from_numpy(samples)
        self.targets = torch.tensor(np.asarray(targets)[indices], dtype=torch.long)
        self.batch_ids = tuple(batch_ids)
        self.indices = indices

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]
