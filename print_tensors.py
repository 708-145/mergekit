import argparse
import sys
import torch
from mergekit.io.lazy_tensor_loader import LazyTensorLoader

def main():
    parser = argparse.ArgumentParser(description="Print all tensors with details for a model directory.")
    parser.add_argument("model_path", type=str, help="Path to the model directory")
    args = parser.parse_args()

    try:
        loader = LazyTensorLoader.from_disk(args.model_path)
    except Exception as e:
        print(f"Error loading model from {args.model_path}: {e}")
        sys.exit(1)

    print(f"Model found at: {args.model_path}")
    print(f"Type: {'Safetensors' if loader.index.is_safetensors else 'PyTorch Bin'}")
    print("-" * 80)
    print(f"{'Tensor Name':<50} | {'Shape':<20} | {'Dtype':<10}")
    print("-" * 80)

    # Sort shards to have deterministic order
    sorted_shards = sorted(loader.index.shards, key=lambda s: s.filename)

    for shard in sorted_shards:
        # Sort keys within shard
        sorted_keys = sorted(shard.contained_keys)
        for key in sorted_keys:
            try:
                tensor = loader.get_tensor(key)
                shape_str = str(tuple(tensor.shape))
                dtype_str = str(tensor.dtype).replace("torch.", "")
                print(f"{key:<50} | {shape_str:<20} | {dtype_str:<10}")
            except Exception as e:
                 print(f"{key:<50} | {'ERROR':<20} | {str(e):<10}")

if __name__ == "__main__":
    main()
