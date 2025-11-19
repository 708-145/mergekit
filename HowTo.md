- merge
  - conda activate mergekit
  - pip install -e . 
  - mergekit-moe clowncar.yaml ./Qwen3-0.6Bx2 --i-understand-this-is-not-useful-without-training --clone-tensors
  - python3 inference.py --model_name ./Qwen3-0.6Bx2 
  - conda create -n ggufconvert python=3.10
  - conda activate ggufconvert
  - (ggufconvert) tobiulla@Graf2:~/sandbox/upstream/llama.cpp$ pip install -e gguf-py
  - python3 ./convert_hf_to_gguf.py --outfile Qwen3-clown.gguf --outtype q8_0 ./Qwen3-0.6Bx2
  - ~/sandbox/llama-cli-6268-blas -m Qwen3-clown.gguf -p "Hi there"

- prune layers
  - conda activate mergekit
  - mergekit-yaml layer2.yaml ./Qwen3-2layers
  - manually edit Qwen3-2layers/config.json and adjust layer_types to correct number of layers (2)
    - should be fixed in mergekit-yaml
  - python3 inference.py --model_name ./Qwen3-2layers
  - python3 ./convert_hf_to_gguf.py --outfile 2layer.gguf --outtype q8_0 ./Qwen3-2layers/
  - ~/sandbox/llama-cli-6268-blas -m 2layer.gguf -p "Hi there"

