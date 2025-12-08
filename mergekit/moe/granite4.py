# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

import logging
from typing import List, Optional

import torch
import tqdm
import transformers
from transformers.models.granitemoe import GraniteMoeConfig

from mergekit.architecture.json_definitions import NAME_TO_ARCH
from mergekit.moe.arch import MoEOutputArchitecture
from mergekit.moe.common import copy_tensor_out, initialize_io, select_dtype
from mergekit.moe.config import MoEMergeConfig
from mergekit.options import MergeOptions

GRANITE_INFO = NAME_TO_ARCH["GraniteForCausalLM"][0]


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
                logging.warning("Granite 4 MoE merge does not support shared experts")
            return False

        for model_ref in (
            [config.base_model]
            + [e.source_model for e in config.experts]
            + [e.source_model for e in (config.shared_experts or [])]
        ):
            model_cfg = model_ref.config(trust_remote_code=trust_remote_code)
            if model_cfg.model_type != "granite":
                if explain:
                    logging.warning("Granite 4 MoE only supports Granite input models")
                return False
        return True

    def _generate_config(
        self,
        base_config: transformers.PretrainedConfig,
        num_experts: int,
        experts_per_token: Optional[int] = None,
    ) -> GraniteMoeConfig:
        # GraniteMoeConfig structure needs to be respected
        out_cfg = GraniteMoeConfig(**base_config.to_dict())
        out_cfg.architectures = ["GraniteMoeForCausalLM"]
        out_cfg.model_type = "granitemoe"
        out_cfg.num_local_experts = num_experts
        out_cfg.num_experts_per_tok = experts_per_token or 2

        # Check if we need to set any other specific config params
        # GraniteMoeConfig: num_local_experts, num_experts_per_tok, etc.
        # It inherits from GraniteConfig so other params like hidden_size are fine.

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

        out_dtype = select_dtype(config, base_cfg)
        out_cfg = self._generate_config(
            base_cfg,
            len(config.experts),
            config.experts_per_token,
        )
        if out_dtype is not None:
            out_cfg.torch_dtype = out_dtype
        out_cfg.save_pretrained(out_path)

        loaders, base_loader, writer = initialize_io(config, out_path, merge_options)

        # We need to map dense weights to MoE weights.
        # GraniteMoe uses stacked weights for experts.
        # input_linear: [num_experts, intermediate_size * 2, hidden_size]
        # output_linear: [num_experts, hidden_size, intermediate_size]

        num_experts = len(config.experts)

        for weight_info in tqdm.tqdm(
            GRANITE_INFO.all_weights(base_cfg),
            desc="Weights",
        ):
            tensor_name = weight_info.name

            # Handle MLP weights which become MoE experts
            if ".mlp.gate_proj.weight" in tensor_name:
                # We handle the entire expert block when we encounter gate_proj
                # tensor_name is model.layers.{i}.mlp.gate_proj.weight

                layer_prefix = tensor_name.replace(".mlp.gate_proj.weight", "")

                # Construct input_linear
                expert_input_weights = []
                for expert in config.experts:
                    expert_loader = loaders.get(expert.source_model)

                    gate_name = f"{layer_prefix}.mlp.gate_proj.weight"
                    up_name = f"{layer_prefix}.mlp.up_proj.weight"

                    w_gate = expert_loader.get_tensor(gate_name)
                    w_up = expert_loader.get_tensor(up_name)

                    # Ensure same dtype
                    if out_dtype is not None:
                        w_gate = w_gate.to(dtype=out_dtype)
                        w_up = w_up.to(dtype=out_dtype)

                    # Stack gate and up: [intermediate_size * 2, hidden_size]
                    # Since w_gate is [inter, hidden] and w_up is [inter, hidden]
                    # We concatenate them along dim 0.
                    w_expert_in = torch.cat([w_gate, w_up], dim=0)
                    expert_input_weights.append(w_expert_in)

                # Stack all experts: [num_experts, intermediate_size * 2, hidden_size]
                combined_input = torch.stack(expert_input_weights, dim=0)

                out_name_input = f"{layer_prefix}.block_sparse_moe.input_linear.weight"
                writer.save_tensor(
                    out_name_input,
                    combined_input,
                    clone=merge_options.clone_tensors,
                )

                # Construct output_linear
                expert_output_weights = []
                for expert in config.experts:
                    expert_loader = loaders.get(expert.source_model)
                    down_name = f"{layer_prefix}.mlp.down_proj.weight"
                    w_down = expert_loader.get_tensor(down_name)

                    if out_dtype is not None:
                        w_down = w_down.to(dtype=out_dtype)

                    expert_output_weights.append(w_down)

                # Stack all experts: [num_experts, hidden_size, intermediate_size]
                combined_output = torch.stack(expert_output_weights, dim=0)

                out_name_output = f"{layer_prefix}.block_sparse_moe.output_linear.weight"
                writer.save_tensor(
                    out_name_output,
                    combined_output,
                    clone=merge_options.clone_tensors,
                )

            elif ".mlp.up_proj.weight" in tensor_name or ".mlp.down_proj.weight" in tensor_name:
                # Handled with gate_proj
                continue

            elif ".mlp.gate_proj.bias" in tensor_name:
                 # GraniteMoe doesn't seem to support bias in experts in the code I read?
                 # GraniteMoeParallelExperts weight is [num_experts, out, in]. No bias.
                 # GraniteMLP has bias.
                 # If bias exists in dense, we drop it?
                 # Wait, GraniteMoeParallelExperts does NOT have bias parameter.
                 # So if dense model has bias, we can't support it easily in GraniteMoe unless we fuse it or something.
                 # But standard Granite models usually have bias=False?
                 # GraniteConfig default is mlp_bias=False.
                 # If it is True, we have a problem.
                 # For now, I'll ignore MLP biases as they are not in the target architecture.
                 # If they are critical, this merge type might not work for bias=True models.
                 logging.warning(f"Skipping MLP bias {tensor_name} as Granite MoE does not support expert bias.")
                 continue

            elif ".mlp.up_proj.bias" in tensor_name or ".mlp.down_proj.bias" in tensor_name:
                continue

            else:
                # Standard weights (attention, norm, etc.) - copy from base
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

        # Save router weights
        for layer_idx, weight in enumerate(
            tqdm.tqdm(router_weights, desc="Router weights")
        ):
            writer.save_tensor(
                f"model.layers.{layer_idx}.block_sparse_moe.router.layer.weight",
                weight.to(dtype=out_dtype).contiguous(),
                clone=merge_options.clone_tensors,
            )

        writer.finalize()
