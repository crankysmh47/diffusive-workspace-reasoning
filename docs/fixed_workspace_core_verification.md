# Fixed-N workspace and set reader verification

2026-10-14. The [component design](fixed_workspace_core_design.md) is implemented in `dwr/models/workspace.py` and `dwr/models/interface.py`. Luna implemented; root made scientific decisions and independent reviewers checked the components and release. This establishes actual tensor correctness on CPU, not native pretrained-model/GPU correctness, training, transfer, statistical power or independent-person reproduction.

## Implemented behavior

EX uses IID Gaussian slots and one shared `Z_next=Z+F(LN(Z),X)` block. IDX adds exactly N*d learned identities once at initialization. Defaults are N64,d768,K4; tests declare tiny fixed dimensions. The reader implements `H+g*CA(LN_H(H),LN_Z(Z),LN_Z(Z))`, gate0.1 and set attention without slot positions. Caller isolates model/reader initialization streams; explicit local noise generators replay first and second draws byte-exactly without consuming global CPU RNG.

CPUfloat64/dropout0 checks cover K1/K4 EX equivariance; whole-final-vector reader invariance for both EX and IDX; nonuniform future-query perturbation with actual changed future output and unchanged earlier output; padding/batching, strict shapes/dtypes/finite values; live terminal gradients through Z0..Z4; state_dict keys/shapes roundtrip. Actual SGD fixtures for **both** methods assert finite nonzero gradients and actual byte changes for every new parameter, including IDX identities, while frozen linear-tail bytes remain unchanged. The native LM, its causal masks, prompt provenance and shifted answer loss are not yet integrated. State_dict roundtrip is not complete training resume.

A review-derived edge fix rejects a finite Python gate value that overflows the constructor's parameter dtype. Its explicit1e308 regression failed before the fix and passed afterward. It does not change default0.1 or the corpus. Earlier logs, failed assertions and snapshots are retained rather than replaced.

## Actual root release checks

Root's first monolithic `.venv-ml/Scripts/python.exe -m unittest discover -s tests -v` ran548 tests in45.417s, wrapper46.922, exit1/no180s timeout: one failure and one error in unchanged256MiB CPU diagnosis guards. A fresh measured process grew from19,238,912 bytes before Torch import to484,696,064 afterward. The failure is environmental coupling during unittest discovery, not a passing suite; no historical caps/configs/guards were changed.

The [dated operational departure](fixed_workspace_core_design.md) preceded root's separated verification. Under one combined180s watchdog, stable source/config bytes:

- `.venv-task/Scripts/python.exe -m unittest discover -s tests -v`: **528 standard-library tests passed**,539 reported including11 explicit optional-ML skips; unittest41.875s, wrapper42.031, exit0. Root first asserted Torch absent in this fresh Python3.11.9 environment. These skips supply no tensor evidence.
- `.venv-ml/Scripts/python.exe -m unittest discover -s tests -p "test_workspace*.py" -v`: **20 actual tensor tests passed, zero skips**, unittest0.752s, wrapper2.062, exit0.
- Combined44.140s; before/after hashes match for all dwr/tests Python, config JSON and pyproject files. All548 distinct real checks executed across their appropriate environments. No monolithic-PASS claim.

Root receipts/logs remain ignored in `.superpowers/sdd/workspace-core-20261014/`:

- Failed monolithic log SHA256 `f471f268b42d746eb782eacef5010e49cd0efa3764a965f5cfb955c07ca4f0b0` and adjacent controller-full-suite JSON.
- `controller-separated-suites-20261014T183704694294Z.json`; stdlib log suffix-1 SHA256 `2fb1a0fc4b3e0ddbcf21b35bfb594412c5f11c0cccd8a8c9d8dd6f8044d0b6e1`; actual ML log suffix-2 SHA256 `8bc0b83fe953f1b76dc97b0b285eece25d9c41d429bfaf61425c97c27408c7e9`.
- `environment.json` binds actual Torch2.11.0+cu128/Python3.11.9 and exact10-package inventory. Import's NumPy-missing warning remains visible; NumPy was not needed or installed. Global Torch2.12.dev remains untouched. No GPU tensor execution occurred.
- Task1/task2 actual RED/GREEN, self-review/API receipts, original snapshots and separate independent READY reviews/rereviews. Initial reader RED filename000000 is only a filename, not an actual invocation timestamp. Original missing five-stratum test-order logs remain explicitly missing in that earlier verification; this core does not reconstruct them.
- Whole-core `release-review.md` and `release-rereview.md`: READY_FOR_CORE_CHECKPOINT on their captured current hashes, conditional on root's actual verification above. Reviews ran no GPU/model experiments.

Current SHA256: workspace.py `8535b0d1c9a748910ea794f522c0bd04d5022df45a2ea751019f0b12d598e44d`; interface.py `f09e83880db35ce6f1839431cec2baf3550bcab83d3fc93bd17f604486a3a543`; models/__init__.py `79e309517bf545f60c7a7e5ab748f700054a0bf2ed60aa0f96baef7b84412143`. Final test hashes are recorded in root's stable-byte receipt. Small source/config/docs go in one consolidated checkpoint; environments/raw evidence stay ignored.

## Next prerequisites for GPU task training

Implement the [native Qwen2 integration](native_backbone_design.md), strict corpus/run/evaluation/checkpoint/resume machinery, and a separately registered full pretrained forward/backward/optimizer memory probe. Freeze actual precision/layout/length/batch settings after that probe. Then complete an exploratory pilot registration before task training. Confirmatory EX-IDX effects need their separate precision analysis, effect/sample/seed counts and caps; the seven-example EQ/NEQ diagnostic A3 limitation remains unchanged. K0 initialization exists as a forward path, not a separately trained noise-memory control. No G3/G4 learning/transfer gate is passed by these component tests.
