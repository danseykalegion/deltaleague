# Candidate D: public lineage plus final-RMSNorm 1.02

Competition: `qwen35-08b-v1`. Submission parent: immutable original
`0ea62f40-7cd8-4d98-9ebb-484192fa05eb`.

## What this is

Candidate D combines two published weight edits and one new, intentionally small
parameter choice:

1. xf3s' public little-Qwen v0.1 projection edit and layer-16 neuron-998 flip;
2. half-strength interpolation of RikiyaT's public layer-15/19 sparse edit;
3. a new `1.02` scale on the effective final RMSNorm gain: `(1 + w) * 1.02 - 1`.

The first two steps are credited to their authors. This PR does not claim them
as original work. It records the exact derivative and its limitations.

## Reproduce

```sh
delta competition init qwen35-08b-v1 --output experiment
python research/danseykalegion-candidate-d-norm102/build_edit.py \
  experiment/model/model.safetensors candidate-d.safetensors
delta patch build --parent experiment/model --modified candidate-d.safetensors --output submission
delta validate submission
```

The script fetches only public, pinned files:

- xf3s' layer-13 direction at commit `0f93e2f340ff74f1f409f25aae78cd6cd2389da4`;
- RikiyaT's manifest and sparse values at commit
  `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`.

## Artifact

The local artifact passed `delta validate` with these properties:

| Field | Value |
|---|---|
| format | sparse diff |
| modified tensors | 32 |
| artifact SHA-256 | `7600d2bec3943c44f98f589cdbcb8d4d8e85f078f5e3c0f0f87a10c34d9bf0cf` |
| full edited checkpoint SHA-256 | `5e3620e9f0f0ebfd84154559aa1bdaa1a4569ee9ab8f8c6bb439f5c9c758ab67` |

## Evidence and limits

The local public evaluator is not usable with the active validation inventory,
so this PR reports no local public score. A one-prompt CPU smoke run produced a
different answer from the immutable original. That is not a benchmark and does
not establish either a gain or a loss.

No official submission ID, server validation, private score, or leaderboard
rank exists for Candidate D at publication time. Delta League's server result
will be the only ranking evidence.
