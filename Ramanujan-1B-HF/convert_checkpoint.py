#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=Path(__file__).resolve().parent, type=Path)
    parser.add_argument("--max-shard-size", default="1900MB")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    loaded = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = loaded.get("model", loaded)
    state = {name: tensor.detach().contiguous() for name, tensor in state.items()}
    if state["embed_tokens.weight"].data_ptr() == state["lm_head.weight"].data_ptr():
        del state["lm_head.weight"]

    split = split_torch_state_dict_into_shards(
        state, filename_pattern="model{suffix}.safetensors", max_shard_size=args.max_shard_size
    )
    for filename, tensor_names in split.filename_to_tensors.items():
        save_file({name: state[name] for name in tensor_names}, args.output / filename, metadata={"format": "pt"})
    if split.is_sharded:
        index = {"metadata": split.metadata, "weight_map": split.tensor_to_filename}
        (args.output / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"saved {len(state):,} tensors in {len(split.filename_to_tensors)} shard(s) to {args.output}")


if __name__ == "__main__":
    main()

