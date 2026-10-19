"""Deterministic source-example order driven by a private CPU RNG stream."""

from __future__ import annotations

import re

import torch


_ID = re.compile(r"[0-9a-f]{64}\Z")
_STATE_KEYS = {"version", "example_ids", "seed", "epoch", "position",
               "permutation", "generator_state"}


def _integer(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-boolean integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


class SourceSampler:
    """Yield each approved source ID once per epoch, without padding or drops."""

    VERSION = 1

    def __init__(self, example_ids: tuple[str, ...], *, seed: int) -> None:
        if not isinstance(example_ids, tuple) or not example_ids:
            raise TypeError("example_ids must be a nonempty tuple")
        if any(not isinstance(item, str) or _ID.fullmatch(item) is None for item in example_ids):
            raise ValueError("example_ids must contain lowercase SHA-256 strings")
        if len(set(example_ids)) != len(example_ids):
            raise ValueError("example_ids must be unique")
        _integer("seed", seed)
        if seed > 2**63 - 1:
            raise ValueError("seed must be at most 2**63-1")
        self.example_ids = tuple(example_ids)
        self.seed = seed
        self._generator = torch.Generator(device="cpu").manual_seed(seed)
        self.epoch = 0
        self.position = 0
        self.permutation = self._new_permutation()

    def _new_permutation(self) -> tuple[int, ...]:
        return tuple(torch.randperm(len(self.example_ids), generator=self._generator).tolist())

    def next_ids(self, batch_size: int) -> tuple[str, ...]:
        _integer("batch_size", batch_size, minimum=1)
        if self.position == len(self.permutation):
            self.epoch += 1
            self.position = 0
            self.permutation = self._new_permutation()
        end = min(self.position + batch_size, len(self.permutation))
        result = tuple(self.example_ids[self.permutation[i]] for i in range(self.position, end))
        self.position = end
        return result

    def state_dict(self) -> dict[str, object]:
        return {
            "version": self.VERSION,
            "example_ids": tuple(self.example_ids),
            "seed": self.seed,
            "epoch": self.epoch,
            "position": self.position,
            "permutation": tuple(self.permutation),
            "generator_state": self._generator.get_state().clone(),
        }

    def load_state_dict(self, state: object) -> None:
        if not isinstance(state, dict) or set(state) != _STATE_KEYS:
            raise ValueError("sampler state has missing or unknown keys")
        if type(state["version"]) is not int or state["version"] != self.VERSION:
            raise ValueError("unsupported sampler state version")
        if not isinstance(state["example_ids"], tuple) or state["example_ids"] != self.example_ids:
            raise ValueError("sampler source IDs or order do not match")
        if type(state["seed"]) is not int or state["seed"] != self.seed:
            raise ValueError("sampler seed does not match")
        epoch = _integer("epoch", state["epoch"])
        position = _integer("position", state["position"])
        permutation = state["permutation"]
        if not isinstance(permutation, tuple) or len(permutation) != len(self.example_ids):
            raise ValueError("sampler permutation has an incompatible shape")
        if any(type(index) is not int for index in permutation) or set(permutation) != set(range(len(self.example_ids))):
            raise ValueError("sampler permutation must contain each source index exactly once")
        if position > len(permutation):
            raise ValueError("sampler position exceeds the current epoch")
        generator_state = state["generator_state"]
        if not isinstance(generator_state, torch.Tensor) or generator_state.dtype != torch.uint8 or generator_state.device.type != "cpu":
            raise ValueError("sampler generator state must be a CPU uint8 tensor")
        generator_copy = generator_state.detach().clone()
        probe = torch.Generator(device="cpu")
        try:
            probe.set_state(generator_copy)
        except RuntimeError as error:
            raise ValueError("sampler generator state is invalid") from error
        self.epoch = epoch
        self.position = position
        self.permutation = tuple(permutation)
        self._generator.set_state(generator_copy)
