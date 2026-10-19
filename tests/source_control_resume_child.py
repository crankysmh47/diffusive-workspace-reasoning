"""Fresh-interpreter continuation helper for test_source_controls."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import test_source_controls as fixture
from dwr.training.checkpoints import load_training_checkpoint, save_training_checkpoint
from dwr.training.controls import source_control_step


def main() -> int:
    torch.set_num_threads(1)
    steps = int(sys.argv[1])
    source_path = Path(sys.argv[2])
    digest = sys.argv[3]
    destination = Path(sys.argv[4])
    random.seed(1201)
    np.random.seed(1203)
    torch.manual_seed(1207)
    host, control, optimizer, scheduler, sampler, noise_generator, progress = (
        fixture.SourceControlTests()._run_parts(steps)
    )
    progress = load_training_checkpoint(source_path, expected_sha256=digest,
        host=host, optimizer=optimizer, scheduler=scheduler, scaler=None,
        sampler=sampler, noise_generator=noise_generator, identities=fixture._IDENTITIES)
    for _ in range(2):
        random.random()
        np.random.random()
        torch.rand((), dtype=torch.float64)
        ids = sampler.next_ids(1)
        batch = fixture._batch(ids[0])
        noise = host.workspace.sample_noise(1, generator=noise_generator,
            device="cpu", dtype=torch.float64)
        result = source_control_step(host, batch, noise, control=control,
            optimizer=optimizer, scheduler=scheduler, gradient_clip_norm=1.0)
        progress["optimizer_step"] += 1
        progress["examples_seen"] += len(ids)
        progress["supervised_tokens_seen"] += result["supervised_tokens"]
    save_training_checkpoint(destination, host=host, optimizer=optimizer,
        scheduler=scheduler, scaler=None, sampler=sampler,
        noise_generator=noise_generator, identities=fixture._IDENTITIES, progress=progress)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
