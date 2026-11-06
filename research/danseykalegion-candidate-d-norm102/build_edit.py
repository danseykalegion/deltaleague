"""Build Candidate D from pinned public inputs; no weights are committed."""
import hashlib
import io
import json
import sys
import urllib.request

import numpy as np
import torch
from safetensors.torch import load_file, save_file

if len(sys.argv) != 3:
    raise SystemExit("usage: build_edit.py PARENT.safetensors OUT.safetensors")

parent, output = sys.argv[1:]
xf3s = "https://raw.githubusercontent.com/xf3s/deltaleague/0f93e2f340ff74f1f409f25aae78cd6cd2389da4/research/xf3s-little-qwen-doubt/doubt_direction_L13.json"
upstream = "https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/edit"

with urllib.request.urlopen(xf3s) as response:
    direction = torch.tensor(json.load(response)["values"], dtype=torch.float32)
direction /= direction.norm()
with urllib.request.urlopen(upstream + "/manifest.json") as response:
    manifest = json.load(response)
with urllib.request.urlopen(upstream + "/edit.npz") as response:
    sparse = np.load(io.BytesIO(response.read()), allow_pickle=False)

weights = load_file(parent)
for layer in range(14):
    for part in ("self_attn.o_proj", "linear_attn.out_proj", "mlp.down_proj"):
        name = f"model.language_model.layers.{layer}.{part}.weight"
        if name in weights:
            matrix = weights[name].float()
            matrix -= torch.outer(direction, direction @ matrix)
            weights[name] = matrix.to(weights[name].dtype)
weights["model.language_model.layers.16.mlp.down_proj.weight"][:, 998] *= -2

for index, entry in enumerate(manifest["tensors"]):
    name, base = entry["name"], weights[entry["name"]]
    bits = base.view(torch.uint16).numpy().copy()
    digest = hashlib.sha256(bits.tobytes()).hexdigest()
    assert digest == entry["base_sha256"], name
    bits.reshape(-1)[sparse[f"t{index}_idx"]] = sparse[f"t{index}_val"]
    published = torch.from_numpy(bits).view(torch.bfloat16)
    assert hashlib.sha256(bits.tobytes()).hexdigest() == entry["edited_sha256"], name
    weights[name] = (base.float() + 0.5 * (published.float() - base.float())).to(torch.bfloat16)

norm = "model.language_model.norm.weight"
weights[norm] = ((1 + weights[norm].float()) * 1.02 - 1).to(torch.bfloat16)
save_file(weights, output, metadata={"format": "pt"})
