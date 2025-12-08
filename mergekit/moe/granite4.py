# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

import logging
from typing import List, Optional

import torch
import tqdm
import transformers

from mergekit.architecture.json_definitions import NAME_TO_ARCH
from mergekit.moe.arch import MoEOutputArchitecture
from mergekit.moe.common import initialize_io, select_dtype
from mergekit.moe.config import MoEMergeConfig
from mergekit.options import MergeOptions

try:
    from transformers.models.granitemoehybrid import GraniteMoeHybridConfig
except ImportError:
    GraniteMoeHybridConfig = None

GRANITE4_INFO = NAME_TO_ARCH["GraniteMoeHybridForCausalLM"][0]


class Granite4MoE(MoEOutputArchitecture):
    def name(self) -> str:
        return "Granite 4 MoE"

    def supports_config(
        self,
        config: MoEMergeConfig,
        explain: bool = False,
        trust_remote_code: bool = False,
    ) -> bool:
        if len(config.shared_experts or []) != 0:
            if explain:
                logging.warning("Granite 4 MoE merge does not support shared experts configuration in mergekit config (model has native shared MLP)")
            return False

        for model_ref in (
            [config.base_model]
            + [e.source_model for e in config.experts]
        ):
            model_cfg = model_ref.config(trust_remote_code=trust_remote_code)
            if model_cfg.model_type != "granitemoehybrid":
                if explain:
                    logging.warning(f"Granite 4 MoE only supports GraniteMoeHybrid input models, got {model_cfg.model_type}")
                return False
        return True

    def _generate_config(
        self,
        base_config: transformers.PretrainedConfig,
        num_experts: int,
        experts_per_token: Optional[int] = None,
        intermediate_size: Optional[int] = None,
    ) -> transformers.PretrainedConfig:
        if GraniteMoeHybridConfig is None:
            raise ImportError("GraniteMoeHybridConfig not available in transformers")

        out_cfg = GraniteMoeHybridConfig(**base_config.to_dict())
        out_cfg.architectures = ["GraniteMoeHybridForCausalLM"]
        out_cfg.num_local_experts = num_experts
        out_cfg.num_experts_per_tok = experts_per_token or 2

        # In Granite 4, intermediate_size controls the expert size
        if intermediate_size is not None:
             out_cfg.intermediate_size = intermediate_size

        return out_cfg

    def write_model(
        self,
        out_path: str,
        config: MoEMergeConfig,
        merge_options: MergeOptions,
        router_weights: List[torch.Tensor],
        shared_router_weights: Optional[List[torch.Tensor]] = None,
    ):
        base_model = config.base_model
        base_cfg = base_model.config(trust_remote_code=merge_options.trust_remote_code)

        # Determine expert intermediate size from the first expert
        # We assume all experts have the same shared_intermediate_size (which becomes the expert size)
        first_expert = config.experts[0].source_model
        first_expert_cfg = first_expert.config(trust_remote_code=merge_options.trust_remote_code)
        expert_intermediate_size = first_expert_cfg.shared_intermediate_size

        out_dtype = select_dtype(config, base_cfg)
        out_cfg = self._generate_config(
            base_cfg,
            len(config.experts),
            config.experts_per_token,
            intermediate_size=expert_intermediate_size
        )
        if out_dtype is not None:
            out_cfg.torch_dtype = out_dtype
        out_cfg.save_pretrained(out_path)

        loaders, base_loader, writer = initialize_io(config, out_path, merge_options)

        # Iterate over all possible weights in the output architecture
        for weight_info in tqdm.tqdm(
            GRANITE4_INFO.all_weights(out_cfg),
            desc="Weights",
        ):
            tensor_name = weight_info.name

            if "block_sparse_moe" in tensor_name:
                # This is an MoE weight, construct it from experts
                if "input_linear" in tensor_name or "output_linear" in tensor_name:
                    # Collect weights from experts' shared_mlp
                    expert_tensors = []

                    # Determine source name in dense model
                    # block_sparse_moe.input_linear -> shared_mlp.input_linear
                    source_suffix = tensor_name.split("block_sparse_moe.")[1] # e.g. input_linear.weight
                    source_name_template = tensor_name.replace("block_sparse_moe." + source_suffix, f"shared_mlp.{source_suffix}")

                    for expert in config.experts:
                        expert_loader = loaders.get(expert.source_model)

                        # We need to find the exact name in the expert model.
                        # Since we are iterating per layer (implicitly by weight_info), we need to handle layer indices.
                        # weight_info.name has the index expanded (e.g. model.layers.0...)

                        # However, source_name_template already has the correct layer index because tensor_name has it.
                        # So simply using source_name_template should work.

                        # But wait, does source model have the same layer index?
                        # Usually yes, unless we are mixing layers.
                        # mergekit config.experts can specify distinct models.
                        # But standard merge assumes compatible layers or we handle it?
                        # initialize_io returns loaders.

                        # Wait, we need to load the specific tensor.
                        # The expert definition in config might have a filter/parameter but here we assume full model match?
                        # config.experts is a list of MoEExpertConfig.

                        t = expert_loader.get_tensor(source_name_template)
                        if t is None:
                             # Try fallback if names are different?
                             # Dense Granite 4 has shared_mlp.
                             raise ValueError(f"Could not find {source_name_template} in expert model {expert.source_model}")

                        expert_tensors.append(t.to(dtype=out_dtype))

                    # Stack experts
                    # input_linear: (hidden*2, input) -> (num_experts, hidden*2, input)
                    # output_linear: (input, hidden) -> (num_experts, input, hidden)
                    # GraniteMoeHybridParallelExperts expects (num_experts, out, in)
                    # transformers Linear weights are (out, in)
                    # So for input_linear: weight is (hidden*2, input). MoE wants (num_experts, hidden*2, input).
                    # This matches simple stacking dim=0.

                    moe_tensor = torch.stack(expert_tensors, dim=0)
                    writer.save_tensor(tensor_name, moe_tensor, clone=merge_options.clone_tensors)

                elif "router.layer.weight" in tensor_name:
                    # Extract layer index to find corresponding router weight
                    # tensor_name: model.layers.{i}.block_sparse_moe.router.layer.weight
                    parts = tensor_name.split(".")
                    try:
                        layer_idx = int(parts[2])
                        router_weight = router_weights[layer_idx]
                        writer.save_tensor(tensor_name, router_weight.to(dtype=out_dtype).contiguous(), clone=merge_options.clone_tensors)
                    except (IndexError, ValueError):
                        logging.warning(f"Could not determine layer index for router weight {tensor_name}")

            else:
                # Regular weight (shared_mlp, attention, etc.) - copy from base model
                tensor = base_loader.get_tensor(
                    tensor_name,
                    aliases=weight_info.aliases,
                    raise_on_missing=not weight_info.optional,
                )
                if tensor is None:
                    continue

                writer.save_tensor(
                    tensor_name,
                    tensor.to(dtype=out_dtype),
                    clone=merge_options.clone_tensors,
                )

        writer.finalize()
