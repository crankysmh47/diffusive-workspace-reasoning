"""Shared tiny-native fixture adapters for controller/job integration tests only."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))
import test_source_pilot_learning as _native


def tiny_host(seed: int = 501):
    return _native.fixture_host(seed)


def tiny_tokenizer():
    return _native.FixtureTokenizer()


def examples(count: int = 5, split: str = "source_validation", start: int = 0):
    return _native.fixture_examples(count, split, start)


def metadata(rows):
    return _native.fixture_metadata(rows)


def kernel_registration(arm_id: str, steps: int = 2):
    return _native.fixture_registration(arm_id, steps=steps)


def scaled_policy(host: Any, tokenizer: Any, *, steps: int = 2, validation_examples: int = 5,
                  decoder_max_new_tokens: int = 8) -> dict[str, Any]:
    from dwr.experiments.source_pilot import _parameter_counts
    counts = _parameter_counts(host)
    return {"validation_examples": validation_examples, "optimizer_steps": steps,
        "expected_original_parameter_count": len(tuple(host.backbone.named_parameters())),
        "expected_parameter_counts": counts, "expected_model_vocab_size": int(host.backbone.config.vocab_size),
        "expected_tokenizer_vocab_size": int(tokenizer.vocab_size),
        "expected_eos_token_id": int(tokenizer.eos_token_id),
        "decoder_max_new_tokens": decoder_max_new_tokens}


def identities(registration, *, source_commit: str, arm_id: str) -> dict[str, str]:
    """Shared paired identities except for the distinct per-arm config digest."""
    config = registration["config"]
    shared = {
        "source": hashlib.sha256(source_commit.encode("utf-8")).hexdigest(),
        "config": hashlib.sha256((registration["config_sha256"] + ":" + arm_id).encode("utf-8")).hexdigest(),
        "data": hashlib.sha256(str(registration["cohort_sha256"]).encode("utf-8")).hexdigest(),
        "model": str(registration["model_pins"]["weight_sha256"]),
        "tokenizer": str(registration["model_pins"]["artifact_inventory_sha256"]),
        "schedule": hashlib.sha256(str(config["training"]["schedule"]).encode("utf-8")).hexdigest(),
    }
    return shared
