"""Decision Transformer training script for JSSP problem, based on Qwen2.5-7B."""

import argparse
import torch
import torch.nn.functional as F
# 注释掉wandb导入以避免依赖问题
# import wandb
import os
import sys
import json
import hashlib
from collections import defaultdict
import matplotlib.pyplot as plt
# 注释掉unsloth导入，因为前面已经移除了相关功能
# from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import TrainingArguments, TrainerCallback
from torch.utils.data import DataLoader, Sampler
import numpy as np
from ssnc_mask import SSNCMaskController, extract_probe_structure_features

# 在文件顶部确保项目根目录在 sys.path 中，使所有子模块导入可靠
_DT_TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
if _DT_TRAIN_DIR not in sys.path:
    sys.path.insert(0, _DT_TRAIN_DIR)

# 顶层导入 JSSPEnv，供 load_or_generate_trajectories 使用
try:
    from Envs.JSSPEnv.JSSPEnv import JSSPEnv as _TopLevelJSSPEnv
except Exception:
    _TopLevelJSSPEnv = None

def save_peft_model(dt_model, save_path):
    """
    统一的PEFT模型保存函数，确保正确保存adapter_model.bin文件
    
    Args:
        dt_model: Decision Transformer模型实例
        save_path: 保存路径
    """
    # 确保目录存在
    os.makedirs(save_path, exist_ok=True)
    
    try:
        # 对于PEFT模型，直接使用base_model.save_pretrained应该会保存adapter_model.bin
        dt_model.base_model.save_pretrained(save_path)
        print(f"Saved model to {save_path}")
        
        # 验证文件是否生成
        if os.path.exists(os.path.join(save_path, 'adapter_model.bin')):
            print(f"Successfully created adapter_model.bin in {save_path}")
        else:
            print(f"Warning: adapter_model.bin was not created in {save_path}. Check PEFT version.")
            # 手动保存PEFT状态字典
            try:
                from peft import PeftModel, get_peft_model_state_dict
                
                if isinstance(dt_model.base_model, PeftModel):
                    # 使用get_peft_model_state_dict获取PEFT状态字典
                    peft_state_dict = get_peft_model_state_dict(dt_model.base_model)
                    # 手动保存adapter_model.bin
                    torch.save(peft_state_dict, os.path.join(save_path, 'adapter_model.bin'))
                    print(f"Manually saved adapter_model.bin to {save_path}")
            except Exception as e:
                print(f"Failed to manually save adapter weights: {e}")

        custom_state = {
            "rtg_proj": dt_model.rtg_proj.state_dict() if hasattr(dt_model, "rtg_proj") else None,
            "state_proj": dt_model.state_proj.state_dict() if hasattr(dt_model, "state_proj") and dt_model.state_proj is not None else None,
            "state_proj_norm": dt_model.state_proj_norm.state_dict() if hasattr(dt_model, "state_proj_norm") and dt_model.state_proj_norm is not None else None,
            "state_encoder": dt_model.state_encoder.state_dict() if hasattr(dt_model, "state_encoder") and dt_model.state_encoder is not None else None,
            "action_proj": dt_model.action_proj.state_dict() if hasattr(dt_model, "action_proj") else None,
            "action_head": dt_model.action_head.state_dict() if hasattr(dt_model, "action_head") else None,
            "candidate_job_proj": dt_model.candidate_job_proj.state_dict() if hasattr(dt_model, "candidate_job_proj") and dt_model.candidate_job_proj is not None else None,
            "candidate_context_proj": dt_model.candidate_context_proj.state_dict() if hasattr(dt_model, "candidate_context_proj") and dt_model.candidate_context_proj is not None else None,
            "candidate_fuse_norm": dt_model.candidate_fuse_norm.state_dict() if hasattr(dt_model, "candidate_fuse_norm") and dt_model.candidate_fuse_norm is not None else None,
            "candidate_action_head": dt_model.candidate_action_head.state_dict() if hasattr(dt_model, "candidate_action_head") and dt_model.candidate_action_head is not None else None,
            "candidate_harmony_mask": dt_model.candidate_harmony_mask.detach().cpu() if hasattr(dt_model, "candidate_harmony_mask") and torch.is_tensor(dt_model.candidate_harmony_mask) else None,
            "scale_context_adapter_norm": dt_model.scale_context_adapter_norm.state_dict() if hasattr(dt_model, "scale_context_adapter_norm") and dt_model.scale_context_adapter_norm is not None else None,
            "scale_context_adapter_down": dt_model.scale_context_adapter_down.state_dict() if hasattr(dt_model, "scale_context_adapter_down") and dt_model.scale_context_adapter_down is not None else None,
            "scale_context_adapter_up": dt_model.scale_context_adapter_up.state_dict() if hasattr(dt_model, "scale_context_adapter_up") and dt_model.scale_context_adapter_up is not None else None,
            "scale_context_adapter_alpha": dt_model.scale_context_adapter_alpha.detach().cpu() if hasattr(dt_model, "scale_context_adapter_alpha") and dt_model.scale_context_adapter_alpha is not None else None,
            "scale_context_harmony_mask": dt_model.scale_context_harmony_mask.detach().cpu() if hasattr(dt_model, "scale_context_harmony_mask") and torch.is_tensor(dt_model.scale_context_harmony_mask) else None,
            "scale_candidate_residual_norm": dt_model.scale_candidate_residual_norm.state_dict() if hasattr(dt_model, "scale_candidate_residual_norm") and dt_model.scale_candidate_residual_norm is not None else None,
            "scale_candidate_residual_down": dt_model.scale_candidate_residual_down.state_dict() if hasattr(dt_model, "scale_candidate_residual_down") and dt_model.scale_candidate_residual_down is not None else None,
            "scale_candidate_residual_up": dt_model.scale_candidate_residual_up.state_dict() if hasattr(dt_model, "scale_candidate_residual_up") and dt_model.scale_candidate_residual_up is not None else None,
            "scale_candidate_residual_alpha": dt_model.scale_candidate_residual_alpha.detach().cpu() if hasattr(dt_model, "scale_candidate_residual_alpha") and dt_model.scale_candidate_residual_alpha is not None else None,
            "scale_candidate_harmony_mask": dt_model.scale_candidate_harmony_mask.detach().cpu() if hasattr(dt_model, "scale_candidate_harmony_mask") and torch.is_tensor(dt_model.scale_candidate_harmony_mask) else None,
            "scale_param_mask_gnn_encoder": dt_model.scale_param_mask_gnn_encoder.detach().cpu() if hasattr(dt_model, "scale_param_mask_gnn_encoder") and torch.is_tensor(dt_model.scale_param_mask_gnn_encoder) else None,
            "scale_param_mask_state_projection": dt_model.scale_param_mask_state_projection.detach().cpu() if hasattr(dt_model, "scale_param_mask_state_projection") and torch.is_tensor(dt_model.scale_param_mask_state_projection) else None,
            "scale_param_mask_lora": dt_model.scale_param_mask_lora.detach().cpu() if hasattr(dt_model, "scale_param_mask_lora") and torch.is_tensor(dt_model.scale_param_mask_lora) else None,
            "scale_param_mask_candidate_head": dt_model.scale_param_mask_candidate_head.detach().cpu() if hasattr(dt_model, "scale_param_mask_candidate_head") and torch.is_tensor(dt_model.scale_param_mask_candidate_head) else None,
            "scale_param_mask_scale_adapter": dt_model.scale_param_mask_scale_adapter.detach().cpu() if hasattr(dt_model, "scale_param_mask_scale_adapter") and torch.is_tensor(dt_model.scale_param_mask_scale_adapter) else None,
            "ssnc_mask_state": dt_model._ssnc_mask_controller.state_dict() if getattr(dt_model, "_ssnc_mask_controller", None) is not None else None,
            "value_head": dt_model.value_head.state_dict() if hasattr(dt_model, "value_head") and dt_model.value_head is not None else None,
            "timestep_embedding": dt_model.timestep_embedding.state_dict() if hasattr(dt_model, "timestep_embedding") else None,
            "type_embedding": dt_model.type_embedding.state_dict() if hasattr(dt_model, "type_embedding") else None,
            "scale_embedding": dt_model.scale_embedding.state_dict() if hasattr(dt_model, "scale_embedding") and dt_model.scale_embedding is not None else None,
            "scale_alpha": dt_model.scale_alpha.detach().cpu() if hasattr(dt_model, "scale_alpha") and dt_model.scale_alpha is not None else None,
            "meta": {
                "n_jobs": getattr(dt_model, "n_jobs", None),
                "n_machines": getattr(dt_model, "n_machines", None),
                "hidden_size": getattr(dt_model, "hidden_size", None),
                "state_encoder_type": getattr(dt_model, "state_encoder_type", None),
                "use_candidate_action_head": getattr(dt_model, "use_candidate_action_head", None),
                "use_candidate_harmony_mask": getattr(dt_model, "use_candidate_harmony_mask", None),
                "candidate_harmony_group_by": getattr(dt_model, "candidate_harmony_group_by", None),
                "use_scale_adapter_expert": getattr(dt_model, "use_scale_adapter_expert", None),
                "use_scale_context_adapter": getattr(dt_model, "use_scale_context_adapter", None),
                "use_scale_candidate_residual": getattr(dt_model, "use_scale_candidate_residual", None),
                "use_scale_adapter_harmony_mask": getattr(dt_model, "use_scale_adapter_harmony_mask", None),
                "scale_adapter_group_by": getattr(dt_model, "scale_adapter_group_by", None),
                "scale_adapter_bottleneck": getattr(dt_model, "scale_adapter_bottleneck", None),
                "use_scale_embedding": getattr(dt_model, "use_scale_embedding", None),
                "use_scale_param_mask": getattr(dt_model, "use_scale_param_mask", None),
                "mask_gnn_encoder": getattr(dt_model, "mask_gnn_encoder", None),
                "mask_state_projection": getattr(dt_model, "mask_state_projection", None),
                "mask_lora": getattr(dt_model, "mask_lora", None),
                "mask_candidate_head": getattr(dt_model, "mask_candidate_head", None),
                "mask_scale_adapter": getattr(dt_model, "mask_scale_adapter", None),
                "scale_param_mask_group_by": getattr(dt_model, "scale_param_mask_group_by", None),
                "scale_param_mask_conflict_mode": getattr(dt_model, "scale_param_mask_conflict_mode", "legacy_global_mean"),
                "state_tensorization_mode": get_state_tensorization_mode(),
            },
        }
        torch.save(custom_state, os.path.join(save_path, "dt_custom_modules.pt"))
        print(f"Saved dt_custom_modules.pt to {save_path}")
            
    except Exception as e:
        print(f"Error saving model to {save_path}: {e}")
        # 尝试备选方案
        try:
            from peft import PeftModel
            if isinstance(dt_model.base_model, PeftModel):
                dt_model.base_model.save_pretrained(save_path)
                print(f"Saved model using PeftModel save method to {save_path}")
        except Exception as e2:
            print(f"Alternative save method also failed for {save_path}: {e2}")


def load_adapter_checkpoint_into_dt_model(dt_model, checkpoint_dir, custom_modules_path=None, is_trainable=True):
    """Load a PEFT adapter and saved DT custom modules into a DecisionTransformer."""
    from peft import PeftModel
    import torch as _torch

    dt_model.base_model = PeftModel.from_pretrained(
        dt_model.base_model,
        checkpoint_dir,
        is_trainable=bool(is_trainable),
    )

    if custom_modules_path is None:
        candidate = os.path.join(checkpoint_dir, "dt_custom_modules.pt")
        if os.path.exists(candidate):
            custom_modules_path = candidate

    if not (custom_modules_path and os.path.exists(custom_modules_path)):
        return dt_model

    try:
        custom_state = torch.load(custom_modules_path, map_location="cpu", weights_only=True)
    except TypeError:
        custom_state = torch.load(custom_modules_path, map_location="cpu")

    def _safe_load_module(module, state_dict):
        if module is None or state_dict is None:
            return
        if isinstance(module, _torch.nn.Embedding) and "weight" in state_dict:
            w_src = state_dict["weight"]
            w_tgt = module.weight.data
            if hasattr(w_src, "shape") and hasattr(w_tgt, "shape") and w_src.shape[1] == w_tgt.shape[1] and w_src.shape[0] != w_tgt.shape[0]:
                n = min(w_src.shape[0], w_tgt.shape[0])
                w_tgt[:n].copy_(w_src[:n])
                return
        current = module.state_dict()
        filtered = {k: v for k, v in state_dict.items() if k in current and hasattr(v, "shape") and current[k].shape == v.shape}
        module.load_state_dict(filtered, strict=False)

    for k in [
        "rtg_proj", "state_proj", "state_proj_norm", "state_encoder", "action_proj", "action_head",
        "candidate_job_proj", "candidate_context_proj", "candidate_fuse_norm", "candidate_action_head",
        "scale_context_adapter_norm", "scale_context_adapter_down", "scale_context_adapter_up",
        "scale_candidate_residual_norm", "scale_candidate_residual_down", "scale_candidate_residual_up",
        "value_head", "timestep_embedding", "type_embedding", "scale_embedding",
    ]:
        sd = custom_state.get(k, None)
        if sd is None:
            continue
        m = getattr(dt_model, k, None)
        if m is None:
            continue
        _safe_load_module(m, sd)

    if getattr(dt_model, "scale_alpha", None) is not None and custom_state.get("scale_alpha", None) is not None:
        with torch.no_grad():
            dt_model.scale_alpha.copy_(custom_state["scale_alpha"].to(device=dt_model.scale_alpha.device, dtype=dt_model.scale_alpha.dtype))
    for param_name in ["scale_context_adapter_alpha", "scale_candidate_residual_alpha"]:
        param = getattr(dt_model, param_name, None)
        src = custom_state.get(param_name, None)
        if param is not None and src is not None:
            with torch.no_grad():
                rows = min(int(src.shape[0]), int(param.shape[0]))
                cols = min(int(src.shape[1]), int(param.shape[1]))
                param[:rows, :cols].copy_(src[:rows, :cols].to(device=param.device, dtype=param.dtype))
    meta = custom_state.get("meta", {}) if isinstance(custom_state, dict) else {}
    saved_group_by = meta.get("candidate_harmony_group_by", None) if isinstance(meta, dict) else None
    current_group_by = getattr(dt_model, "candidate_harmony_group_by", None)
    if saved_group_by and current_group_by and str(saved_group_by) != str(current_group_by):
        print(
            f"[LOAD] Warning: checkpoint candidate_harmony_group_by={saved_group_by} "
            f"but current model uses {current_group_by}"
        )
    if getattr(dt_model, "candidate_harmony_mask", None) is not None and custom_state.get("candidate_harmony_mask", None) is not None:
        src = custom_state["candidate_harmony_mask"]
        dst = dt_model.candidate_harmony_mask
        if torch.is_tensor(src) and torch.is_tensor(dst):
            with torch.no_grad():
                rows = min(int(src.shape[0]), int(dst.shape[0]))
                cols = min(int(src.shape[1]), int(dst.shape[1]))
                dst[:rows, :cols].copy_(src[:rows, :cols].to(device=dst.device, dtype=dst.dtype))
    for mask_name in [
        "scale_context_harmony_mask", "scale_candidate_harmony_mask",
        "scale_param_mask_gnn_encoder", "scale_param_mask_state_projection",
        "scale_param_mask_lora", "scale_param_mask_candidate_head",
        "scale_param_mask_scale_adapter",
    ]:
        src = custom_state.get(mask_name, None)
        dst = getattr(dt_model, mask_name, None)
        if torch.is_tensor(src) and torch.is_tensor(dst):
            with torch.no_grad():
                rows = min(int(src.shape[0]), int(dst.shape[0]))
                cols = min(int(src.shape[1]), int(dst.shape[1]))
                dst[:rows, :cols].copy_(src[:rows, :cols].to(device=dst.device, dtype=dst.dtype))

    # The controller is constructed later, after dataset scale groups are known.
    # Retain its serialized EMA state on the model until then.
    if isinstance(custom_state, dict) and custom_state.get("ssnc_mask_state", None) is not None:
        dt_model._loaded_ssnc_mask_state = custom_state["ssnc_mask_state"]

    return dt_model


DT_CUSTOM_MODULE_NAMES = [
    "rtg_proj", "state_proj", "state_proj_norm", "state_encoder", "action_proj", "action_head",
    "candidate_job_proj", "candidate_context_proj", "candidate_fuse_norm", "candidate_action_head",
    "candidate_harmony_mask",
    "scale_context_adapter_norm", "scale_context_adapter_down", "scale_context_adapter_up", "scale_context_adapter_alpha", "scale_context_harmony_mask",
    "scale_candidate_residual_norm", "scale_candidate_residual_down", "scale_candidate_residual_up", "scale_candidate_residual_alpha", "scale_candidate_harmony_mask",
    "scale_param_mask_gnn_encoder", "scale_param_mask_state_projection", "scale_param_mask_lora",
    "scale_param_mask_candidate_head", "scale_param_mask_scale_adapter",
    "value_head", "timestep_embedding", "type_embedding", "scale_embedding", "scale_alpha",
]


def move_dt_custom_modules_to_device(dt_model, device):
    """Move non-backbone DT modules to device; safe for quantized backbones."""
    for module_name in DT_CUSTOM_MODULE_NAMES:
        module = getattr(dt_model, module_name, None)
        if isinstance(module, torch.nn.Parameter):
            module.data = module.data.to(device=device)
        elif torch.is_tensor(module):
            setattr(dt_model, module_name, module.to(device=device))
        elif module is not None and hasattr(module, "to"):
            module.to(device)
    return dt_model

# Import our custom modules
from decision_transformer import DecisionTransformer, create_decision_transformer_for_jssp
from dt_data_processing import (
    JSSPTrajectoryCollector,
    JSSPDecisionTransformerDataset,
    LazyJSSPDecisionTransformerDataset,
    build_jsonl_offsets,
    collate_fn,
    create_data_loader,
    process_jssp_trajectories_for_dt,
    generate_synthetic_jssp_trajectories,
    ORToolsAgent,
    collect_ortools_trajectories,
    collect_mixed_scale_trajectories
)
from state_tensorization import (
    GRAPH_STATE_KEYS,
    LEGACY_TRAIN_COMPATIBLE,
    VALID_TENSORIZATION_MODES,
    get_state_tensorization_mode,
    set_state_tensorization_mode,
    tensorize_graph_state_history,
)


def _scale_label_sort_key(label, order="area_desc"):
    label = str(label or "unknown").lower()
    try:
        left, right = label.split("x", 1)
        jobs = int(float(left))
        machines = int(float(right))
    except Exception:
        jobs, machines = 0, 0
    area = jobs * machines
    order = str(order or "area_desc").lower()
    if order == "area_asc":
        return (area, jobs, machines, label)
    if order == "name":
        return (label,)
    return (-area, -jobs, -machines, label)


class ScaleRoundRobinSampler(Sampler):
    """Balanced scale sampler that cycles through scales while shuffling inside each scale."""

    def __init__(self, scale_labels, seed=42, order="area_desc", epoch_size=None):
        self.scale_labels = [str(x or "unknown").lower() for x in scale_labels]
        self.seed = int(seed)
        self.order = str(order or "area_desc")
        self.epoch_size = int(epoch_size) if epoch_size else len(self.scale_labels)
        by_scale = defaultdict(list)
        for idx, label in enumerate(self.scale_labels):
            by_scale[label].append(int(idx))
        self.scales = sorted(by_scale.keys(), key=lambda s: _scale_label_sort_key(s, self.order))
        self.by_scale = {k: list(v) for k, v in by_scale.items()}
        self._epoch = 0

    def __len__(self):
        return int(self.epoch_size)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1
        pools = {}
        positions = {}
        for scale in self.scales:
            items = np.array(self.by_scale[scale], dtype=np.int64)
            rng.shuffle(items)
            pools[scale] = items.tolist()
            positions[scale] = 0
        if not self.scales:
            return iter([])

        out = []
        scale_pos = 0
        while len(out) < self.epoch_size:
            scale = self.scales[scale_pos % len(self.scales)]
            scale_pos += 1
            pool = pools[scale]
            if not pool:
                continue
            pos = positions[scale]
            if pos >= len(pool):
                items = np.array(pool, dtype=np.int64)
                rng.shuffle(items)
                pools[scale] = items.tolist()
                pool = pools[scale]
                pos = 0
            out.append(pool[pos])
            positions[scale] = pos + 1
        return iter(out)


def create_train_data_loader(dataset, args):
    strategy = str(getattr(args, "train_scale_sampler", "random") or "random").lower()
    batch_size = int(getattr(args, "per_device_train_batch_size", 1) or 1)
    if strategy in {"random", "shuffle", "none"}:
        return create_data_loader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    labels = getattr(dataset, "scale_labels", None)
    if not labels or len(labels) != len(dataset):
        print("[ScaleSampler] scale_labels unavailable; falling back to global shuffle")
        return create_data_loader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    if strategy not in {"scale_round_robin", "round_robin", "balanced"}:
        print(f"[ScaleSampler] unknown strategy={strategy}; falling back to global shuffle")
        return create_data_loader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    sampler = ScaleRoundRobinSampler(
        labels,
        seed=int(getattr(args, "seed", 42)),
        order=str(getattr(args, "train_scale_order", "area_desc") or "area_desc"),
        epoch_size=len(dataset),
    )
    counts = defaultdict(int)
    for label in labels:
        counts[str(label).lower()] += 1
    print(
        "[ScaleSampler] enabled: "
        f"strategy=scale_round_robin, order={getattr(args, 'train_scale_order', 'area_desc')}, "
        f"epoch_size={len(sampler)}, scales={dict(sorted(counts.items(), key=lambda kv: _scale_label_sort_key(kv[0], getattr(args, 'train_scale_order', 'area_desc'))))}"
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Decision Transformer Trainer for JSSP problem')
    # --disable_wandb参数已移除，因为WandB已完全禁用

    # Model and data parameters

    parser.add_argument('--dtype', type=str, default='float32', choices=['bfloat16', 'float16', 'float32'],
                        help='Data type (bfloat16, float16, or float32) - default float32 for AMP compatibility')
    parser.add_argument('--load_in_4bit', action='store_true', default=False,
                        help='Use 4-bit quantization to reduce memory usage')
    # Model name
    parser.add_argument('--model_name', type=str, default='unsloth/Qwen2.5-7B', help='Model name')
    parser.add_argument('--init_model_dir', type=str, default=None, help='Initialize from an existing adapter checkpoint dir (best_model_step_*)')
    parser.add_argument('--init_custom_modules', type=str, default=None, help='Optional dt_custom_modules.pt path (defaults to <init_model_dir>/dt_custom_modules.pt)')
    parser.add_argument('--distill_old_model_logits', action='store_true', default=False,
                        help='Use a frozen old-model teacher to distill candidate action logits on old-scale samples')
    parser.add_argument('--teacher_model_dir', type=str, default=None,
                        help='Frozen teacher adapter checkpoint dir used by --distill_old_model_logits')
    parser.add_argument('--teacher_custom_modules', type=str, default=None,
                        help='Optional teacher dt_custom_modules.pt path (defaults to <teacher_model_dir>/dt_custom_modules.pt)')
    parser.add_argument('--teacher_load_in_4bit', action='store_true', default=False,
                        help='Load the frozen teacher in 4-bit to reduce memory during online distillation')
    parser.add_argument('--teacher_model_max_jobs', type=int, default=None,
                        help='Teacher model job capacity; defaults to student capacity')
    parser.add_argument('--teacher_model_max_machines', type=int, default=None,
                        help='Teacher model machine capacity; defaults to distill_max_machines when set, otherwise student capacity')
    parser.add_argument('--distill_weight', type=float, default=0.3,
                        help='Weight for old-model logits distillation loss')
    parser.add_argument('--distill_temperature', type=float, default=2.0,
                        help='Softmax temperature for old-model logits distillation')
    parser.add_argument('--distill_max_machines', type=int, default=10,
                        help='Only distill states with problem_n_machines <= this value; use <=0 to disable scale filtering')
    # Model source (huggingface or modelscope)
    parser.add_argument('--model_source', type=str, default='huggingface', choices=['huggingface', 'modelscope'],
                        help='Source of the model (huggingface or modelscope)')
    # Problem name
    parser.add_argument('--problem', type=str, default='jssp', help='Problem name')

    # LoRA hyperparameters
    parser.add_argument('--lora_r', type=int, default=16, help='Rank of the LoRA decomposition')
    parser.add_argument('--lora_alpha', type=int, default=32, help='Scaling factor for LoRA updates')
    parser.add_argument('--bias', type=str, default='lora_only', choices=['none', 'all', 'lora_only'], help='Bias type')

    # Decision Transformer specific hyperparameters
    parser.add_argument('--state_dim', type=int, default=64, help='Dimension of state embeddings')
    parser.add_argument('--max_length', type=int, default=64, help='Maximum sequence length for DT')
    parser.add_argument('--max_ep_len', type=int, default=50, help='Maximum episode length')

    # Training hyperparameters
    parser.add_argument('--per_device_train_batch_size', type=int, default=2,
                        help='Batch size per device during training')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16,
                        help='Number of gradient accumulation steps')
    parser.add_argument('--warmup_ratio', type=float, default=0.05, help='Warmup ratio')
    parser.add_argument('--num_train_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--logging_steps', type=int, default=10, help='Logging steps')
    parser.add_argument('--optim', type=str, default='adamw_8bit', help='Optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping max norm')
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine', help='Learning rate scheduler type')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_total_limit', type=int, default=5, help='Total save limit for model checkpoints')
    parser.add_argument('--save_step', type=int, default=1000, help='Steps interval to save model checkpoints')
    parser.add_argument('--early_stop_patience', type=int, default=8, help='Early stopping: stop after N consecutive eval steps with no improvement')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=4,
                        help='Batch size per device during evaluation')
    parser.add_argument('--use_gradient_checkpointing', action='store_true', default=True,
                        help='Use gradient checkpointing to reduce memory usage')

    # Output and evaluation
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory name')
    parser.add_argument('--eval_steps', type=int, default=100, help='Steps interval to evaluate the model')
    parser.add_argument('--rollout_episodes', type=int, default=5, help='Number of rollout episodes per evaluation')
    parser.add_argument('--rollout_eval_n_jobs', type=int, default=None,
                        help='Problem size used for rollout gap evaluation during training. Defaults to n_jobs.')
    parser.add_argument('--rollout_eval_n_machines', type=int, default=None,
                        help='Problem size used for rollout gap evaluation during training. Defaults to n_machines.')
    parser.add_argument('--rollout_target_returns', type=str, default='0.8,0.9,1.0,1.1',
                        help='Comma-separated target returns used by rollout evaluation, e.g. "0.85,0.9,0.95,1.0"')
    parser.add_argument('--rollout_rtg_mode', type=str, default='normalized_budget',
                        choices=['constant', 'budget', 'normalized_budget'],
                        help='RTG mode used by rollout gap evaluation during training')
    parser.add_argument('--rollout_top_k', type=int, default=1,
                        help='During rollout, sample only from the top-k legal actions when temperature > 0')
    parser.add_argument('--rollout_temperature', type=float, default=0.0,
                        help='Rollout sampling temperature; 0.0 keeps greedy argmax')
    parser.add_argument('--rollout_best_of_n', type=int, default=1,
                        help='For each eval instance, run N stochastic rollouts and keep the best makespan')
    parser.add_argument('--rollout_panel_scales', type=str, default='auto',
                        help='Fixed rollout panel scales. Use "auto", "none", or comma-separated scales like "10x10,20x10,20x20".')
    parser.add_argument('--rollout_panel_size_per_scale', type=int, default=0,
                        help='Fixed rollout panel instances per scale. 0 keeps rollout_episodes for one scale and 1 for multi-scale auto panels.')
    parser.add_argument('--rollout_panel_jsonl', type=str, default=None,
                        help='Optional fixed rollout panel jsonl path(s), comma-separated. Lines need current_instance or instance.')
    parser.add_argument('--rollout_panel_score_weights', type=str, default='',
                        help='Optional comma-separated scale weights for checkpoint score, e.g. "20x10:0.35,15x15:0.2,20x15:0.2,20x20:0.25".')
    parser.add_argument('--rollout_crop_history', action='store_true', default=True,
                        help='Crop rollout history to model max_length before each action prediction')
    parser.add_argument('--no_rollout_crop_history', action='store_false', dest='rollout_crop_history',
                        help='Disable rollout history cropping')
    parser.add_argument('--rollout_eval_max_length', type=int, default=64,
                        help='Max history length used only for training-time rollout evaluation. '
                             '0 means use the model max_length.')
    parser.add_argument('--rollout_offload_optimizer_state', action='store_true', default=False,
                        help='Temporarily move optimizer state tensors to CPU during rollout eval to free GPU memory.')

    # Data collection
    parser.add_argument('--trajectories_path', type=str, default=None, help='Path to pre-collected trajectories')
    parser.add_argument('--trajectory_filename', type=str, default=None, help='Custom filename for saved trajectories')
    parser.add_argument('--lazy_trajectory_loading', action='store_true', default=True,
                        help='Load large jsonl trajectory files lazily to avoid CPU OOM')
    parser.add_argument('--no_lazy_trajectory_loading', action='store_false', dest='lazy_trajectory_loading',
                        help='Disable lazy jsonl loading and parse all trajectories into memory')
    parser.add_argument('--lazy_trajectory_threshold_mb', type=float, default=2048.0,
                        help='Use lazy loading for jsonl trajectory files larger than this size in MB')
    parser.add_argument('--instance_hash_split', action='store_true', default=True,
                        help='Split train/eval by canonical JSSP instance hash, stratified by problem scale')
    parser.add_argument('--no_instance_hash_split', action='store_false', dest='instance_hash_split',
                        help='Use legacy row-level random train/eval split')
    parser.add_argument('--eval_ratio', type=float, default=0.2,
                        help='Evaluation split ratio for trajectory datasets')
    parser.add_argument('--train_loss_ma_window', type=int, default=100,
                        help='Moving-average window for logged train loss diagnostics')
    parser.add_argument('--n_trajectories', type=int, default=1000, help='Number of trajectories to generate')
    parser.add_argument('--generate_trajectories', action='store_true', default=False,
                        help='Generate synthetic trajectories if no pre-collected ones are provided')
    parser.add_argument('--only_generate_trajectories', action='store_true', default=False,
                        help='Only generate trajectories and skip training')
    parser.add_argument('--use_ortools_agent', action='store_true', default=False,
                        help='Use OR-ToolsAgent for trajectory generation instead of random generation')
    parser.add_argument('--ortools_time_limit', type=float, default=None,
                        help='Optional CP-SAT time limit in seconds for OR-Tools trajectory generation/evaluation')
    parser.add_argument('--save_light_trajectories', action='store_true', default=False,
                        help='Save compact JSSP trajectories (instance + actions) and replay states during training')
    parser.add_argument('--validate_light_trajectories', action='store_true', default=False,
                        help='When saving compact trajectories, replay each one and compare key states before writing')
    parser.add_argument('--mixed_scale_training', action='store_true', default=False,
                        help='Use mixed scale training (5x5, 5x10, 10x10) for better generalization')
    parser.add_argument('--mixed_scale_config', type=str, default='10x5:0.7,10x10:0.3',
                        help='Mixed scale distribution, e.g. 10x5:0.7,10x10:0.3')
    parser.add_argument('--n_jobs', type=int, default=10, help='Number of jobs in JSSP')
    parser.add_argument('--n_machines', type=int, default=10, help='Number of machines in JSSP')
    parser.add_argument('--model_max_jobs', type=int, default=None,
                        help='Model capacity for max jobs / padded state width. '
                             'Can be larger than the dataset problem size for cross-size generalization.')
    parser.add_argument('--model_max_machines', type=int, default=None,
                        help='Model capacity for max machines / padded state width. '
                             'Can be larger than the dataset problem size for cross-size generalization.')
    parser.add_argument('--state_encoder_type', type=str, default='jssp', choices=['jssp', 'graph'], help='State encoder type')
    parser.add_argument('--graph_disable_queue_pos', action='store_true', default=False,
                        help='Disable graph queue-position feature to reduce absolute job-index bias for cross-size generalization')
    parser.add_argument('--use_candidate_action_head', action='store_true', default=False,
                        help='Use experimental per-job candidate scorer on top of graph job tokens. Disabled by default to match the low-gap v5 job-attention graph baseline.')
    parser.add_argument('--use_candidate_harmony_mask', action='store_true', default=False,
                        help='Apply HarmoDT-style machine-specific channel masks inside the candidate action head')
    parser.add_argument('--candidate_harmony_group_by', type=str, default='machines',
                        choices=['machines', 'scale'],
                        help='Group candidate harmony masks by problem_n_machines or exact (n_jobs, n_machines) scale')
    parser.add_argument('--candidate_harmony_groups', type=str, default='5,10,15,20',
                        help='Comma-separated harmony groups. Use machines like 5,10,15,20 or exact scales like 10x5,15x10; auto scans train split.')
    parser.add_argument('--candidate_harmony_update_interval', type=int, default=0,
                        help='Optimizer-step interval for updating candidate harmony masks. 0 disables updates.')
    parser.add_argument('--candidate_harmony_sparsity', type=float, default=0.10,
                        help='Target fraction of closed candidate-head channels per machine group')
    parser.add_argument('--candidate_harmony_change_frac', type=float, default=0.01,
                        help='Fraction of candidate-head channels changed at each harmony update')
    parser.add_argument('--candidate_harmony_gamma', type=float, default=10.0,
                        help='Weight on gradient agreement score when ranking candidate harmony mask channels')
    parser.add_argument('--candidate_harmony_probe_batch_size', type=int, default=1,
                        help='Samples per machine group used to estimate candidate harmony mask gradients')
    parser.add_argument('--candidate_harmony_probe_max_attempts', type=int, default=200,
                        help='Maximum random dataset probes per group when estimating candidate harmony gradients')
    parser.add_argument('--candidate_harmony_log_flip', action='store_true', default=False,
                        help='Log top-1 action flip rate caused by candidate harmony masks on the probe batches')
    parser.add_argument('--use_scale_adapter_expert', action='store_true', default=False,
                        help='Enable scale-conditioned adapter/expert residual branches without cutting the shared candidate path')
    parser.add_argument('--use_scale_context_adapter', action='store_true', default=False,
                        help='Enable the scale-conditioned transformer-context adapter branch')
    parser.add_argument('--use_scale_candidate_residual', action='store_true', default=False,
                        help='Enable the scale-conditioned candidate-score residual expert branch')
    parser.add_argument('--use_scale_adapter_harmony_mask', action='store_true', default=False,
                        help='Apply HarmoDT-style masks to scale adapter/expert bottleneck channels')
    parser.add_argument('--scale_adapter_group_by', type=str, default='scale',
                        choices=['machines', 'scale'],
                        help='Group scale adapter/expert masks by machine count or exact (n_jobs, n_machines) scale')
    parser.add_argument('--scale_adapter_groups', type=str, default='auto',
                        help='Comma-separated scale adapter groups. Use exact scales like 10x5,15x10 or auto scans train split.')
    parser.add_argument('--scale_adapter_bottleneck', type=int, default=256,
                        help='Bottleneck width for scale adapter/expert branches')
    parser.add_argument('--scale_adapter_alpha_init', type=float, default=0.01,
                        help='Initial per-scale residual strength for scale adapter/expert branches. Use 0.0 for exact old-model behavior.')
    parser.add_argument('--scale_adapter_harmony_update_interval', type=int, default=0,
                        help='Optimizer-step interval for updating scale adapter/expert harmony masks. 0 disables updates.')
    parser.add_argument('--scale_adapter_harmony_sparsity', type=float, default=0.05,
                        help='Target fraction of closed adapter/expert bottleneck channels per scale group')
    parser.add_argument('--scale_adapter_harmony_change_frac', type=float, default=0.005,
                        help='Fraction of adapter/expert bottleneck channels changed at each harmony update')
    parser.add_argument('--scale_adapter_harmony_gamma', type=float, default=10.0,
                        help='Weight on gradient agreement score when ranking scale adapter/expert mask channels')
    parser.add_argument('--scale_adapter_harmony_probe_batch_size', type=int, default=1,
                        help='Samples per scale group used to estimate scale adapter/expert harmony gradients')
    parser.add_argument('--scale_adapter_harmony_probe_max_attempts', type=int, default=300,
                        help='Maximum random dataset probes per group when estimating scale adapter/expert harmony gradients')
    parser.add_argument('--use_scale_param_mask', action='store_true', default=False,
                        help='Enable scale-specific channel/rank masks across selected trainable modules')
    parser.add_argument('--mask_gnn_encoder', action='store_true', default=False,
                        help='When --use_scale_param_mask is enabled, mask GNN state/job-token output channels')
    parser.add_argument('--mask_state_projection', action='store_true', default=False,
                        help='When --use_scale_param_mask is enabled, mask projected state-token channels before Qwen')
    parser.add_argument('--mask_lora', action='store_true', default=False,
                        help='When --use_scale_param_mask is enabled, mask LoRA rank gradients per scale')
    parser.add_argument('--mask_candidate_head', action='store_true', default=False,
                        help='When --use_scale_param_mask is enabled, mask candidate-head fused channels')
    parser.add_argument('--mask_scale_adapter', action='store_true', default=False,
                        help='When --use_scale_param_mask is enabled, mask scale adapter/expert bottleneck channels')
    parser.add_argument('--scale_param_mask_group_by', type=str, default='scale',
                        choices=['machines', 'scale'],
                        help='Group full scale-param masks by machine count or exact (n_jobs, n_machines) scale')
    parser.add_argument('--scale_param_mask_groups', type=str, default='auto',
                        help='Comma-separated scale-param mask groups. Use exact scales like 10x5,15x10 or auto scans train split.')
    parser.add_argument('--scale_param_mask_update_interval', type=int, default=0,
                        help='Optimizer-step interval for outer-loop scale-param mask updates. 0 disables updates.')
    parser.add_argument('--scale_param_mask_warmup_steps', type=int, default=0,
                        help='Do not update scale-param masks before this optimizer step')
    parser.add_argument('--scale_param_mask_sparsity', type=float, default=0.10,
                        help='Target fraction of closed channels/ranks per scale-param mask row')
    parser.add_argument('--scale_param_mask_change_frac', type=float, default=0.005,
                        help='Fraction of channels/ranks changed at each scale-param mask update')
    parser.add_argument('--scale_param_mask_min_change', type=int, default=1,
                        help='Minimum number of channels/ranks closed and recovered at each scale-param mask update')
    parser.add_argument('--scale_param_mask_gamma', type=float, default=10.0,
                        help='Weight on gradient agreement score when ranking scale-param mask channels')
    parser.add_argument('--scale_param_mask_probe_batch_size', type=int, default=1,
                        help='Samples per scale group used to estimate scale-param mask gradients')
    parser.add_argument('--scale_param_mask_probe_max_attempts', type=int, default=300,
                        help='Maximum random dataset probes per group when estimating scale-param mask gradients')
    parser.add_argument('--scale_param_mask_conflict_mode', type=str, default='legacy_global_mean',
                        choices=['legacy_global_mean', 'ssnc_neighbor'],
                        help='Mask conflict reference: original all-scale mean or SSNC structure-neighborhood reference')
    parser.add_argument('--ssnc_neighbor_k', type=int, default=3,
                        help='Number of nearest scale/structure neighbors used by SSNC')
    parser.add_argument('--ssnc_structure_ema_decay', type=float, default=0.90,
                        help='EMA decay for dynamic scale-structure descriptors')
    parser.add_argument('--ssnc_grad_norm_ema_decay', type=float, default=0.90,
                        help='EMA decay for per-module/per-scale gradient RMS normalization')
    parser.add_argument('--ssnc_fisher_ema_decay', type=float, default=0.90,
                        help='EMA decay for SSNC channel importance')
    parser.add_argument('--ssnc_score_ema_decay', type=float, default=0.90,
                        help='EMA decay for SSNC channel ranking scores')
    parser.add_argument('--ssnc_diagnostics_filename', type=str, default='ssnc_mask_diagnostics.jsonl',
                        help='JSONL diagnostic filename under output_dir when SSNC is enabled')
    parser.add_argument('--max_optimizer_steps', type=int, default=0,
                        help='Development smoke-test cap. 0 keeps the original full training behavior.')
    parser.add_argument('--eval_max_batches', type=int, default=0,
                        help='Optional cap on validation batches per eval. 0 evaluates the full validation split.')
    parser.add_argument('--train_scale_sampler', type=str, default='random',
                        choices=['random', 'scale_round_robin', 'round_robin', 'balanced'],
                        help='Training sample order. scale_round_robin cycles through problem scales while shuffling inside each scale.')
    parser.add_argument('--train_scale_order', type=str, default='area_desc',
                        choices=['area_desc', 'area_asc', 'name'],
                        help='Scale order used by --train_scale_sampler scale_round_robin')
    parser.add_argument('--use_scale_embedding', action='store_true', default=False,
                        help='Add a learnable scale condition to each state embedding using problem_n_jobs/problem_n_machines')
    parser.add_argument('--scale_alpha_init', type=float, default=0.0,
                        help='Initial multiplier for scale embedding. 0.0 preserves old-checkpoint behavior at initialization')
    parser.add_argument('--scale_embedding_init_std', type=float, default=1e-3,
                        help='Initialization std for scale embedding weights')
    parser.add_argument('--freeze_graph_encoder_layers', type=int, default=0,
                        help='Freeze the first N typed GNN message-passing layers in GraphJSSPStateEncoder after checkpoint loading')
    parser.add_argument('--input_grad_clip', type=float, default=1.0,
                        help='Clip and sanitize gradients flowing from the frozen backbone back into custom DT frontend inputs')
    parser.add_argument('--max_steps_per_trajectory', type=int, default=200, help='Maximum steps per trajectory')
    parser.add_argument('--ortools_pure_expert_ratio', type=float, default=0.7, help='Target ratio of expert / low-gap trajectories')
    parser.add_argument('--ortools_epsilon_min', type=float, default=0.1, help='Minimum epsilon for epsilon-greedy expert')
    parser.add_argument('--ortools_epsilon_max', type=float, default=0.3, help='Maximum epsilon for epsilon-greedy expert')
    parser.add_argument('--processing_time_mutation_ratio', type=float, default=0.5, help='Ratio of mutated instances')
    parser.add_argument('--processing_time_mutation_pct', type=float, default=0.1, help='Processing time mutation percentage')
    parser.add_argument('--use_sequence_swap', action='store_true', default=True,
                        help='Use sequence swap method for suboptimal trajectory generation (recommended)')
    parser.add_argument('--no_sequence_swap', action='store_false', dest='use_sequence_swap',
                        help='Use traditional epsilon-greedy method (not recommended, may cause efficiency collapse)')
    parser.add_argument('--unlikelihood_weight', type=float, default=0.0, help='Unlikelihood loss weight')
    parser.add_argument('--unlikelihood_negatives', type=int, default=4, help='Number of negative samples per token')
    parser.add_argument('--training_rtg_mode', type=str, default='normalized_budget',
                        choices=['future_return', 'normalized_budget'],
                        help='RTG mode used when repairing/processing training trajectories')
    parser.add_argument(
        '--state_tensorization_mode',
        type=str,
        default=LEGACY_TRAIN_COMPATIBLE,
        choices=VALID_TENSORIZATION_MODES,
        help=(
            'Graph-state layout shared by dataset and rollout. Use '
            'legacy_train_compatible when continuing/evaluating existing '
            'checkpoints; use canonical_v2 for new training.'
        ),
    )
    parser.add_argument('--use_train_action_mask', action='store_true', default=True,
                        help='Mask unavailable JSSP jobs during supervised training/eval when graph states are available')
    parser.add_argument('--no_train_action_mask', action='store_false', dest='use_train_action_mask',
                        help='Disable supervised-training action masking')
    parser.add_argument('--use_gap_weighted_loss', action='store_true', default=True,
                        help='Upweight lower-gap / higher-quality trajectories in the CE loss')
    parser.add_argument('--no_gap_weighted_loss', action='store_false', dest='use_gap_weighted_loss',
                        help='Disable gap/quality weighted CE loss')
    parser.add_argument('--gap_weight_alpha', type=float, default=6.0,
                        help='Strength of relative-gap weighting; larger values emphasize low-gap trajectories more')
    parser.add_argument('--gap_weight_power', type=float, default=2.0,
                        help='Exponent used in relative-gap weighting')
    parser.add_argument('--quality_weight_mix', type=float, default=0.25,
                        help='Mix ratio between gap-based weight and trajectory quality score weight')
    parser.add_argument('--min_sample_weight', type=float, default=0.35,
                        help='Lower bound for per-trajectory sample weights')
    parser.add_argument('--max_sample_weight', type=float, default=3.0,
                        help='Upper bound for per-trajectory sample weights')
    parser.add_argument('--use_value_aux_loss', action='store_true', default=True,
                        help='Add an auxiliary value / remaining-return regression loss')
    parser.add_argument('--no_value_aux_loss', action='store_false', dest='use_value_aux_loss',
                        help='Disable auxiliary value / remaining-return regression loss')
    parser.add_argument('--value_loss_weight', type=float, default=0.2,
                        help='Weight for the auxiliary value loss')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing used by action cross-entropy loss')
    parser.add_argument('--single_action_loss_weight', type=float, default=0.1,
                        help='Loss weight for states with only one legal action; reduces trivial masked decisions')
    parser.add_argument('--freeze_lora_epochs', type=int, default=1,
                        help='Freeze LoRA adapters for the first N epochs so the custom modules can stabilize first')
    parser.add_argument('--transfer_freeze_epochs', type=int, default=None,
                        help='When --init_model_dir is used, freeze LoRA for this many epochs; default is num_train_epochs//3')
    parser.add_argument('--lora_lr_scale', type=float, default=0.1,
                        help='Scale factor applied to LoRA learning rate after unfreezing')
    parser.add_argument('--debug_prints', action='store_true', default=False, help='Enable verbose debug prints')
    
    # 数据集优化参数（用于降低 gap）
    parser.add_argument('--use_data_augmentation', action='store_true', default=False,
                        help='Enable data augmentation (state noise, trajectory crop, RTG scaling)')
    parser.add_argument('--augmentation_mode', type=str, default='offline',
                        choices=['offline', 'online', 'both'],
                        help='Augmentation mode: offline (expand dataset once), online (per-sample during training), both. '
                             '警告：online/both 模式会导致训练与eval数据分布不一致（训练见到带噪声数据，eval是干净数据），'
                             '建议优先使用offline模式')
    parser.add_argument('--augmentation_ratio', type=float, default=0.3,
                        help='Ratio of trajectories to augment (0.0-1.0)')
    parser.add_argument('--state_noise_std', type=float, default=0.01,
                        help='Standard deviation for state noise augmentation')
    parser.add_argument('--rtg_scale', type=float, default=1.1,
                        help='RTG scaling factor for encouraging better solutions')
    parser.add_argument('--use_curriculum_learning', action='store_true', default=False,
                        help='Enable curriculum learning (train from easy to hard)')
    parser.add_argument('--curriculum_stages', type=int, default=3,
                        help='Number of stages for curriculum learning')
    parser.add_argument('--use_multi_scale', action='store_true', default=False,
                        help='Enable multi-scale training (5x5, 5x10, 10x10)')

    args = parser.parse_args()

    # 保留原始的n_jobs参数值，不要覆盖它
    # 模型的动作数量由n_actions控制，而JSSP问题的作业数量由n_jobs控制
    # args_dict = vars(args)
    # args_dict['n_jobs'] = args_dict.get('n_actions', 5)

    return args


def _resolve_model_capacity(args):
    max_jobs = int(getattr(args, "model_max_jobs", 0) or getattr(args, "n_jobs", 0) or 0)
    max_machines = int(getattr(args, "model_max_machines", 0) or getattr(args, "n_machines", 0) or 0)
    return max_jobs, max_machines


def _resolve_rollout_eval_scale(args):
    eval_jobs = int(getattr(args, "rollout_eval_n_jobs", 0) or getattr(args, "n_jobs", 0) or 0)
    eval_machines = int(getattr(args, "rollout_eval_n_machines", 0) or getattr(args, "n_machines", 0) or 0)
    return eval_jobs, eval_machines


def _parse_rollout_target_returns(args):
    raw = str(getattr(args, "rollout_target_returns", "0.8,0.9,1.0,1.1") or "").strip()
    vals = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    return tuple(vals) if vals else (0.8, 0.9, 1.0, 1.1)


def _jssp_makespan_lower_bound(instance):
    """Simple valid lower bound: max(job total load, machine total load)."""
    inst = np.asarray(instance)
    if inst.ndim != 2 or inst.shape[1] % 2 != 0:
        return 0.0
    n_jobs, two_m = inst.shape
    n_machines = two_m // 2
    job_loads = inst[:, 1::2].sum(axis=1)
    machine_loads = np.zeros(n_machines, dtype=float)
    for job in range(n_jobs):
        for op in range(n_machines):
            machine = int(inst[job, op * 2])
            if 0 <= machine < n_machines:
                machine_loads[machine] += float(inst[job, op * 2 + 1])
    return float(max(float(job_loads.max(initial=0.0)), float(machine_loads.max(initial=0.0))))


def _jssp_all_operations_completed(env):
    try:
        return bool(np.all(np.asarray(env.job_status) >= int(env.n_machines)))
    except Exception:
        return False


def compute_loss(logits, target_actions, mask, label_smoothing=0.1, sample_weights=None, action_valid_mask=None):
    """
    计算Decision Transformer的损失，仅在action token位置计算

    Args:
        logits: (B, T, n_jobs, n_machines)  ← 来自 JSSPActionDecoder
        target_actions: (B, T)                    ← 实际动作（扁平索引）
        mask: (B, T)                               ← 时间步是否有效

    Returns:
        loss: scalar tensor           ← 平均损失值
    """
    import torch.nn.functional as F

    # 现在logits是3D: (B, T, act_dim)
    B, T, act_dim = logits.shape

    # 添加数值稳定性检查
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        print("Warning: Logits contain NaN or inf values!")
        return torch.tensor(0.0, requires_grad=True, device=logits.device)

    if torch.isnan(target_actions).any() or torch.isinf(target_actions).any():
        print("Warning: Target actions contain NaN or inf values!")
        return torch.tensor(0.0, requires_grad=True, device=logits.device)

    # Label smoothing should not be mixed with hard action masks: smoothing assigns
    # probability mass to masked logits, which can inflate CE by orders of magnitude.
    label_smoothing = float(label_smoothing or 0.0)
    if label_smoothing > 0.0 and action_valid_mask is not None:
        label_smoothing = 0.0
    elif label_smoothing > 0.0 and bool((logits.detach() <= -9990).any().item()):
        label_smoothing = 0.0

    # ignore_index=-100 自动忽略 padding 位置
    loss = F.cross_entropy(
        logits.reshape(-1, act_dim),
        target_actions.reshape(-1),
        ignore_index=-100,
        label_smoothing=label_smoothing,
        reduction='none'
    ).reshape(B, T)

    # Apply optional per-trajectory weights so lower-gap demonstrations matter more.
    weighted_mask = mask
    if sample_weights is not None:
        sample_weights = sample_weights.to(device=loss.device, dtype=loss.dtype)
        if sample_weights.dim() == 1:
            sample_weights = sample_weights.unsqueeze(-1)
        elif sample_weights.dim() == 0:
            sample_weights = sample_weights.view(1, 1)
        weighted_mask = mask * sample_weights

    # Apply mask to ensure we only compute loss on valid time steps
    loss = (loss * weighted_mask).sum() / (weighted_mask.sum() + 1e-8)

    # 检查损失是否为NaN或inf
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"Warning: Loss is NaN or inf! Loss value: {loss}")
        return torch.tensor(0.0, requires_grad=True, device=logits.device)

    return loss


def compute_unlikelihood_loss(logits, target_actions, mask, num_negatives=4, action_valid_mask=None):
    import torch

    if num_negatives is None or int(num_negatives) <= 0:
        return torch.tensor(0.0, device=logits.device)

    B, T, act_dim = logits.shape
    if act_dim <= 1:
        return torch.tensor(0.0, device=logits.device)

    flat_logits = logits.reshape(-1, act_dim)
    flat_targets = target_actions.reshape(-1)
    flat_mask = mask.reshape(-1)

    valid = (flat_mask > 0) & (flat_targets != -100)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=logits.device)

    logits_v = flat_logits[valid]
    targets_v = flat_targets[valid]
    probs_v = torch.softmax(logits_v, dim=-1)

    k = int(num_negatives)
    if action_valid_mask is not None:
        flat_valid_actions = action_valid_mask.reshape(-1, act_dim).to(device=logits.device)
        candidates = flat_valid_actions[valid].bool()
        candidates.scatter_(1, targets_v.unsqueeze(1), False)
        row_has = candidates.any(dim=1)
        if row_has.sum() == 0:
            return torch.tensor(0.0, device=logits.device)
        candidates = candidates[row_has]
        probs_v = probs_v[row_has]
        weights = candidates.float()
        neg = torch.multinomial(weights, num_samples=k, replacement=True)
    else:
        neg = torch.randint(low=0, high=act_dim - 1, size=(targets_v.shape[0], k), device=logits.device)
        neg = neg + (neg >= targets_v.unsqueeze(1)).long()

    p_neg = probs_v.gather(1, neg)
    ul = -torch.log(torch.clamp(1.0 - p_neg, min=1e-6))
    return ul.mean()


def compute_value_loss(value_preds, value_targets, mask, sample_weights=None):
    import torch.nn.functional as F

    if value_preds is None or value_targets is None:
        return None

    if value_preds.shape != value_targets.shape:
        min_t = min(int(value_preds.shape[1]), int(value_targets.shape[1]))
        value_preds = value_preds[:, :min_t]
        value_targets = value_targets[:, :min_t]
        mask = mask[:, :min_t]

    per_token = F.smooth_l1_loss(value_preds, value_targets, reduction='none')
    weighted_mask = mask.to(device=per_token.device, dtype=per_token.dtype)
    if sample_weights is not None:
        sample_weights = sample_weights.to(device=per_token.device, dtype=per_token.dtype)
        if sample_weights.dim() == 1:
            sample_weights = sample_weights.unsqueeze(-1)
        elif sample_weights.dim() == 0:
            sample_weights = sample_weights.view(1, 1)
        weighted_mask = weighted_mask * sample_weights

    denom = weighted_mask.sum() + 1e-8
    return (per_token * weighted_mask).sum() / denom


def build_prev_action_inputs(actions, padding_action):
    """Build action tokens aligned with states: [dummy, A0, A1, ...]."""
    prev_actions = torch.full_like(actions, fill_value=padding_action)
    if actions.shape[1] > 1:
        if actions.dim() == 3:
            prev_actions[:, 1:, :] = actions[:, :-1, :]
        else:
            prev_actions[:, 1:] = actions[:, :-1]
    return prev_actions


def build_graph_action_mask(states, n_jobs):
    """Reconstruct the same legal job mask used by rollout whenever possible."""
    if not isinstance(states, dict) or "job_next_op" not in states:
        return None

    job_next_op = states["job_next_op"]
    if not torch.is_tensor(job_next_op) or job_next_op.dim() != 3:
        return None

    device = job_next_op.device
    batch_size, seq_len, max_jobs = job_next_op.shape

    if "legal_action_mask" in states and torch.is_tensor(states["legal_action_mask"]):
        legal_action_mask = states["legal_action_mask"].to(device=device)
        if legal_action_mask.dim() == 3:
            if legal_action_mask.shape[-1] < n_jobs:
                pad = torch.zeros(
                    (batch_size, seq_len, n_jobs - legal_action_mask.shape[-1]),
                    dtype=legal_action_mask.dtype,
                    device=device,
                )
                legal_action_mask = torch.cat([legal_action_mask, pad], dim=-1)
            elif legal_action_mask.shape[-1] > n_jobs:
                legal_action_mask = legal_action_mask[..., :n_jobs]
            return legal_action_mask > 0.5

    job_ids = torch.arange(max_jobs, device=device).view(1, 1, max_jobs)

    if "problem_n_jobs" in states and torch.is_tensor(states["problem_n_jobs"]):
        problem_n_jobs = states["problem_n_jobs"].to(device=device).view(batch_size, seq_len, -1)[..., 0]
    else:
        problem_n_jobs = torch.full((batch_size, seq_len), float(max_jobs), device=device)

    if "problem_n_machines" in states and torch.is_tensor(states["problem_n_machines"]):
        problem_n_machines = states["problem_n_machines"].to(device=device).view(batch_size, seq_len, -1)[..., 0]
    elif "machine_busy_mask" in states and torch.is_tensor(states["machine_busy_mask"]):
        problem_n_machines = torch.full((batch_size, seq_len), float(states["machine_busy_mask"].shape[-1]), device=device)
    else:
        problem_n_machines = torch.full((batch_size, seq_len), float(max(1, int(n_jobs))), device=device)

    valid_jobs = (job_ids < problem_n_jobs.unsqueeze(-1)) & (job_next_op < problem_n_machines.unsqueeze(-1))

    required = ("machine_sequence", "normalized_machine_availability", "overall_info")
    if all(k in states and torch.is_tensor(states[k]) for k in required):
        try:
            max_machines = int(max(1, states["normalized_machine_availability"].shape[-1]))
            machine_sequence = states["machine_sequence"].to(device=device).view(batch_size, seq_len, max_jobs, -1)
            if machine_sequence.shape[-1] < max_machines:
                pad = torch.zeros(
                    (*machine_sequence.shape[:-1], max_machines - machine_sequence.shape[-1]),
                    dtype=machine_sequence.dtype,
                    device=device,
                )
                machine_sequence = torch.cat([machine_sequence, pad], dim=-1)
            elif machine_sequence.shape[-1] > max_machines:
                machine_sequence = machine_sequence[..., :max_machines]

            machine_availability = states["normalized_machine_availability"].to(device=device).view(batch_size, seq_len, -1)
            if machine_availability.shape[-1] < max_machines:
                pad = torch.zeros(
                    (batch_size, seq_len, max_machines - machine_availability.shape[-1]),
                    dtype=machine_availability.dtype,
                    device=device,
                )
                machine_availability = torch.cat([machine_availability, pad], dim=-1)
            elif machine_availability.shape[-1] > max_machines:
                machine_availability = machine_availability[..., :max_machines]

            op_idx = job_next_op.long().clamp(min=0, max=max_machines - 1)
            required_machine = torch.gather(
                machine_sequence.long(),
                dim=3,
                index=op_idx.unsqueeze(-1),
            ).squeeze(-1).clamp(min=0, max=max_machines - 1)
            required_machine_avail = torch.gather(machine_availability, dim=2, index=required_machine)

            overall_info = states["overall_info"].to(device=device).view(batch_size, seq_len, -1)
            current_time = overall_info[..., 0]
            valid_jobs = valid_jobs & (required_machine_avail <= current_time.unsqueeze(-1) + 1e-6)
        except Exception:
            pass

    if max_jobs < n_jobs:
        pad = torch.zeros((batch_size, seq_len, n_jobs - max_jobs), dtype=torch.bool, device=device)
        valid_jobs = torch.cat([valid_jobs, pad], dim=-1)
    elif max_jobs > n_jobs:
        valid_jobs = valid_jobs[..., :n_jobs]

    return valid_jobs.to(dtype=torch.bool)


def normalize_action_mask_for_n_jobs(action_mask, n_jobs, device=None):
    """Pad/truncate a legal-action mask so it matches the model action dimension."""
    if action_mask is None:
        return None
    if not torch.is_tensor(action_mask):
        action_mask = torch.as_tensor(action_mask, dtype=torch.bool, device=device)
    else:
        action_mask = action_mask.to(device=device, dtype=torch.bool) if device is not None else action_mask.to(dtype=torch.bool)

    if action_mask.dim() == 2:
        action_mask = action_mask.any(dim=-1)
    if action_mask.dim() == 1:
        if action_mask.numel() < n_jobs:
            pad = torch.zeros((n_jobs - action_mask.numel(),), dtype=torch.bool, device=action_mask.device)
            action_mask = torch.cat([action_mask, pad], dim=0)
        elif action_mask.numel() > n_jobs:
            action_mask = action_mask[:n_jobs]
    elif action_mask.dim() == 3:
        if action_mask.shape[-1] < n_jobs:
            pad = torch.zeros(
                (*action_mask.shape[:-1], n_jobs - action_mask.shape[-1]),
                dtype=torch.bool,
                device=action_mask.device,
            )
            action_mask = torch.cat([action_mask, pad], dim=-1)
        elif action_mask.shape[-1] > n_jobs:
            action_mask = action_mask[..., :n_jobs]
    else:
        raise ValueError(f"Unsupported action_mask dim: {action_mask.dim()}")
    return action_mask


def include_target_actions_in_mask(action_mask, target_actions, n_jobs):
    """Never mask the demonstrated target action, even if a reconstructed mask is imperfect."""
    if action_mask is None:
        return None
    if target_actions.dim() == 3:
        targets = target_actions.squeeze(-1)
    else:
        targets = target_actions
    targets = targets.to(device=action_mask.device)
    valid_targets = (targets >= 0) & (targets < n_jobs)
    safe_targets = targets.clamp(min=0, max=n_jobs - 1).long().unsqueeze(-1)
    action_mask = action_mask.clone()
    action_mask.scatter_(-1, safe_targets, action_mask.gather(-1, safe_targets) | valid_targets.unsqueeze(-1))
    return action_mask


def build_choice_weighted_loss_mask(mask, action_valid_mask=None, single_action_weight=1.0):
    """Downweight trivial states where the legal mask leaves only one possible action."""
    if action_valid_mask is None:
        return mask
    legal_counts = action_valid_mask.to(device=mask.device, dtype=torch.float32).sum(dim=-1)
    weights = torch.ones_like(mask, dtype=torch.float32)
    weights = torch.where(
        legal_counts <= 1.0,
        torch.full_like(weights, float(single_action_weight)),
        weights,
    )
    return mask.to(dtype=torch.float32) * weights


def compute_action_mask_stats(action_valid_mask, mask, targets):
    """Small diagnostics for masked-action training: compare against uniform legal guessing."""
    if action_valid_mask is None:
        return None
    action_valid_mask = action_valid_mask.to(device=mask.device, dtype=torch.bool)
    targets = targets.to(device=mask.device)
    valid = (mask > 0) & (targets != -100)
    if valid.sum() == 0:
        return None
    legal_counts = action_valid_mask.to(dtype=torch.float32).sum(dim=-1).clamp(min=1.0)
    valid_counts = legal_counts[valid]
    target_safe = targets.clamp(min=0, max=action_valid_mask.shape[-1] - 1).long()
    target_in_mask = action_valid_mask.gather(-1, target_safe.unsqueeze(-1)).squeeze(-1)
    return {
        "avg_legal": float(valid_counts.mean().item()),
        "single_pct": float((valid_counts <= 1.0).float().mean().item()),
        "uniform_acc": float((1.0 / valid_counts).mean().item()),
        "uniform_ce": float(torch.log(valid_counts).mean().item()),
        "target_coverage": float(target_in_mask[valid].float().mean().item()),
    }


def compute_non_single_accuracy(logits, targets, mask, action_valid_mask=None):
    import torch
    if action_valid_mask is None:
        return None
    action_valid_mask = action_valid_mask.to(device=logits.device, dtype=torch.bool)
    targets = targets.to(device=logits.device)
    mask = mask.to(device=logits.device)
    valid = (mask > 0) & (targets != -100)
    if valid.sum() == 0:
        return None
    legal_counts = action_valid_mask.float().sum(dim=-1).clamp(min=1.0)
    decision_valid = valid & (legal_counts > 1.0)
    if decision_valid.sum() == 0:
        return {
            "non_single_acc": None,
            "non_single_rand": None,
            "non_single_gain": None,
            "non_single_count": 0,
        }
    masked_logits = logits.to(torch.float32).masked_fill(~action_valid_mask, -float("inf"))
    pred = masked_logits.argmax(dim=-1)
    acc = (pred == targets).float()[decision_valid].mean()
    rand = (1.0 / legal_counts[decision_valid]).mean()
    return {
        "non_single_acc": float(acc.item()),
        "non_single_rand": float(rand.item()),
        "non_single_gain": float((acc - rand).item()),
        "non_single_count": int(decision_valid.sum().item()),
    }


def _mean_buffer_metric(buffer, key):
    vals = []
    for item in buffer:
        val = item.get(key, None)
        if val is None:
            continue
        try:
            val = float(val)
        except Exception:
            continue
        if np.isfinite(val):
            vals.append(val)
    return float(np.mean(vals)) if vals else None


def _append_metric_if_finite(target, key, value):
    try:
        value = float(value)
    except Exception:
        return
    if np.isfinite(value):
        target[key] = value


def _fmt_float(value, precision=3, default="nan"):
    try:
        value = float(value)
    except Exception:
        return default
    if not np.isfinite(value):
        return default
    return f"{value:.{int(precision)}f}"


def _move_optimizer_state_tensors(optimizer, device):
    moved = 0
    bytes_moved = 0

    def move_value(value):
        nonlocal moved, bytes_moved
        if torch.is_tensor(value):
            bytes_moved += int(value.numel() * value.element_size())
            moved += 1
            return value.to(device=device, non_blocking=False)
        if isinstance(value, dict):
            for k, v in list(value.items()):
                value[k] = move_value(v)
            return value
        if isinstance(value, list):
            for i, v in enumerate(value):
                value[i] = move_value(v)
            return value
        if isinstance(value, tuple):
            return tuple(move_value(v) for v in value)
        return value

    for state in optimizer.state.values():
        move_value(state)
    return moved, bytes_moved


def _offload_optimizer_state_for_rollout(optimizer, enabled=True, restore_device=None):
    class _OptimizerStateOffload:
        def __init__(self, opt, enable, restore_dev):
            self.opt = opt
            self.enable = bool(enable)
            self.restore_device = restore_dev
            self.did_offload = False

        def __enter__(self):
            if not self.enable or self.opt is None or not torch.cuda.is_available():
                return self
            moved, bytes_moved = _move_optimizer_state_tensors(self.opt, torch.device("cpu"))
            self.did_offload = moved > 0
            if self.did_offload:
                torch.cuda.empty_cache()
                print(
                    f"[RolloutEval] Offloaded optimizer state to CPU: "
                    f"{moved} tensors, {bytes_moved / (1024 ** 3):.2f} GiB"
                )
            return self

        def __exit__(self, exc_type, exc, tb):
            if self.did_offload and torch.cuda.is_available():
                restore_device = self.restore_device if self.restore_device is not None else torch.device("cuda")
                _move_optimizer_state_tensors(self.opt, restore_device)
                torch.cuda.empty_cache()
                print("[RolloutEval] Restored optimizer state to GPU")
            return False

    return _OptimizerStateOffload(optimizer, enabled, restore_device)


def compute_total_loss(logits, target_actions, mask, unlikelihood_weight=0.0, num_negatives=4,
                       action_valid_mask=None, label_smoothing=0.1, sample_weights=None,
                       value_preds=None, value_targets=None, value_loss_weight=0.0):
    import torch
    ce = compute_loss(
        logits,
        target_actions,
        mask,
        label_smoothing=label_smoothing,
        sample_weights=sample_weights,
        action_valid_mask=action_valid_mask,
    )
    w = float(unlikelihood_weight or 0.0)
    total = ce
    if w > 0:
        ul = compute_unlikelihood_loss(logits, target_actions, mask, num_negatives=num_negatives, action_valid_mask=action_valid_mask)
        total = total + (torch.tensor(w, device=logits.device) * ul)

    vw = float(value_loss_weight or 0.0)
    if vw > 0 and value_preds is not None and value_targets is not None:
        vloss = compute_value_loss(value_preds, value_targets, mask, sample_weights=sample_weights)
        if vloss is not None:
            total = total + (torch.tensor(vw, device=logits.device) * vloss)
    return total


def build_old_scale_distill_mask(states, base_mask, max_machines=10):
    """Select valid old-scale positions for teacher distillation."""
    if not isinstance(states, dict):
        return None
    problem_machines = states.get("problem_n_machines", None)
    if not torch.is_tensor(problem_machines):
        return None
    problem_machines = problem_machines.to(device=base_mask.device, dtype=torch.float32)
    if problem_machines.dim() >= 3:
        problem_machines = problem_machines[..., 0]
    elif problem_machines.dim() == 1:
        problem_machines = problem_machines.unsqueeze(0).expand_as(base_mask)
    max_machines = int(max_machines or 0)
    scale_mask = problem_machines > 0
    if max_machines > 0:
        scale_mask = scale_mask & (problem_machines <= float(max_machines))
    return base_mask.to(dtype=torch.float32) * scale_mask.to(dtype=torch.float32)


def compute_logits_distillation_loss(student_logits, teacher_logits, mask, temperature=2.0, action_valid_mask=None):
    """KL teacher||student over legal actions for selected non-padding positions."""
    if teacher_logits is None or mask is None:
        return None
    if student_logits.shape != teacher_logits.shape:
        seq_len = min(student_logits.shape[1], teacher_logits.shape[1])
        act_dim = min(student_logits.shape[-1], teacher_logits.shape[-1])
        student_logits = student_logits[:, :seq_len, :act_dim]
        teacher_logits = teacher_logits[:, :seq_len, :act_dim]
        mask = mask[:, :seq_len]
        if action_valid_mask is not None:
            action_valid_mask = action_valid_mask[:, :seq_len, :act_dim]

    valid = mask.to(device=student_logits.device, dtype=torch.float32)
    if action_valid_mask is not None:
        legal = action_valid_mask.to(device=student_logits.device, dtype=torch.bool)
        student_logits = student_logits.masked_fill(~legal, -1e9)
        teacher_logits = teacher_logits.masked_fill(~legal, -1e9)
        valid = valid * (legal.sum(dim=-1) > 0).to(dtype=valid.dtype)

    if valid.sum() <= 0:
        return None

    temp = max(1e-6, float(temperature or 1.0))
    student_log_probs = F.log_softmax(student_logits.to(torch.float32) / temp, dim=-1)
    teacher_probs = F.softmax(teacher_logits.detach().to(torch.float32) / temp, dim=-1)
    per_pos = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
    return (per_pos * valid).sum() / (valid.sum() + 1e-8) * (temp * temp)


def compute_accuracy(logits, targets, mask):
    """
    计算Decision Transformer的准确率，仅在action token位置计算

    Args:
        logits: (B, T, act_dim)  ← 来自模型的动作预测（作业级别）
        targets: (B, T)          ← 实际动作（作业ID，0~n_jobs-1）
        mask: (B, T)             ← 时间步是否有效

    Returns:
        accuracy: float          ← 平均准确率
    """
    # 现在logits是3D: (B, T, act_dim)
    B, T, act_dim = logits.shape

    pred = logits.argmax(dim=-1)  # (B, T)
    valid = (mask > 0) & (targets != -100)
    if valid.sum() == 0:
        return 0.0
    acc = (pred == targets).float()
    acc = acc[valid].mean()
    return float(acc.item())


def compute_topk_accuracy(logits, targets, mask, k=3):
    import torch
    k = int(k)
    k = max(1, k)
    valid = (mask > 0) & (targets != -100)
    if valid.sum() == 0:
        return 0.0
    topk = logits.topk(k, dim=-1).indices  # (B, T, k)
    hit = (topk == targets.unsqueeze(-1)).any(dim=-1).float()
    return float(hit[valid].mean().item())


def compute_valid_action_accuracy(logits, targets, mask, action_valid_mask=None):
    import torch
    if action_valid_mask is None:
        return None
    valid = (mask > 0) & (targets != -100)
    if valid.sum() == 0:
        return None
    masked_logits = logits.clone()
    action_valid_mask = action_valid_mask.to(device=logits.device, dtype=torch.bool)
    masked_logits[~action_valid_mask.bool()] = -float("inf")
    pred = masked_logits.argmax(dim=-1)
    acc = (pred == targets).float()
    return float(acc[valid].mean().item())


def setup_wandb(args, dir_out):
    """返回一个虚拟的wandb对象，用于替代实际的wandb日志功能。"""
    # 无论是否设置disable_wandb，都返回虚拟对象
    print("使用虚拟WandB对象进行训练（WandB已完全禁用）。")
    # 创建虚拟wandb对象，提供log和finish方法
    class DummyWandb:
        def log(self, *args, **kwargs):
            # 可选：将日志打印到控制台以查看训练进度
            # print(f"[Training Log]: {kwargs}")
            pass
        def finish(self):
            pass
    return DummyWandb()


def clean_trajectory(traj):
    """
    清洗轨迹数据，确保rewards和returns_to_go字段都是数值类型

    Args:
        traj: 轨迹数据字典

    Returns:
        清洗后的轨迹数据字典
    """
    def to_float(x):
        try:
            return float(x)
        except:
            print(f"Invalid reward/rtg value: {x}, replacing with 0.0")
            return 0.0

    traj['rewards'] = [to_float(r) for r in traj['rewards']]
    traj['returns_to_go'] = [to_float(rtg) for rtg in traj['returns_to_go']]

    # 验证长度一致性
    T = len(traj['actions'])
    assert len(traj['states']) == T + 1, f"States length ({len(traj['states'])}) must be T + 1 ({T + 1})"  # states 包含初始状态
    assert len(traj['rewards']) == T, f"Rewards length ({len(traj['rewards'])}) must be T ({T})"
    assert len(traj['returns_to_go']) == T + 1, f"Returns_to_go length ({len(traj['returns_to_go'])}) must be T + 1 ({T + 1})"
    assert len(traj['timesteps']) == T + 1, f"Timesteps length ({len(traj['timesteps'])}) must be T + 1 ({T + 1})"

    return traj


def _is_lazy_trajectory_source(obj):
    return isinstance(obj, dict) and bool(obj.get("__lazy_jsonl_path__"))


def _finite_float(value, default=None):
    try:
        out = float(value)
        if np.isfinite(out):
            return out
    except Exception:
        pass
    return default


def _extract_reference_makespan(traj):
    metrics = traj.get("metrics", {}) if isinstance(traj, dict) else {}
    if not isinstance(metrics, dict):
        metrics = {}
    for key in (
        "optimal_makespan",
        "ortools_makespan",
        "best_known_makespan",
        "reference_makespan",
        "target_makespan",
    ):
        value = _finite_float(metrics.get(key, traj.get(key, None) if isinstance(traj, dict) else None), None)
        if value is not None and value > 0:
            return value
    return None


def _scale_from_shape(n_jobs, n_machines):
    try:
        return f"{int(n_jobs)}x{int(n_machines)}"
    except Exception:
        return "unknown"


def _parse_scale_list(raw):
    raw = str(raw or "").strip()
    if not raw:
        return []
    if raw.lower() in {"none", "off", "false", "0"}:
        return []
    if raw.lower() == "auto":
        return ["auto"]
    out = []
    for part in raw.split(","):
        s = part.strip().lower()
        if not s:
            continue
        out.append(s)
    return out


def _parse_scale_weights(raw):
    out = {}
    raw = str(raw or "").strip()
    if not raw:
        return out
    for part in raw.split(","):
        if ":" not in part:
            continue
        scale, val = part.split(":", 1)
        try:
            weight = float(val)
        except Exception:
            continue
        if weight > 0:
            out[scale.strip().lower()] = weight
    return out


def _normalise_instance_array(instance):
    if instance is None:
        return None
    try:
        arr = np.asarray(instance, dtype=np.int32)
    except Exception:
        return None
    if arr.ndim != 2 or arr.shape[0] <= 0 or arr.shape[1] <= 0 or arr.shape[1] % 2 != 0:
        return None
    return np.ascontiguousarray(arr, dtype=np.int32)


def _instance_from_state_dict(state, n_jobs=None, n_machines=None):
    if not isinstance(state, dict):
        return None
    machine_sequence = state.get("machine_sequence", None)
    processing_times = state.get("processing_times", None)
    if machine_sequence is None or processing_times is None:
        return None
    try:
        machine_sequence = np.asarray(machine_sequence, dtype=np.int32)
        processing_times = np.asarray(processing_times, dtype=np.int32)
        if n_jobs is None:
            if "problem_n_jobs" in state:
                n_jobs = int(np.asarray(state["problem_n_jobs"]).reshape(-1)[0])
            elif "job_next_op" in state:
                n_jobs = int(np.asarray(state["job_next_op"]).reshape(-1).shape[0])
        if n_machines is None:
            if "problem_n_machines" in state:
                n_machines = int(np.asarray(state["problem_n_machines"]).reshape(-1)[0])
            elif "normalized_machine_availability" in state:
                n_machines = int(np.asarray(state["normalized_machine_availability"]).reshape(-1).shape[0])
        if n_jobs is None or n_machines is None or int(n_jobs) <= 0 or int(n_machines) <= 0:
            return None
        n_jobs = int(n_jobs)
        n_machines = int(n_machines)
        machine_sequence = machine_sequence.reshape(n_jobs, -1)[:, :n_machines]
        processing_times = processing_times.reshape(n_jobs, -1)[:, :n_machines]
        inst = np.zeros((n_jobs, n_machines * 2), dtype=np.int32)
        inst[:, 0::2] = machine_sequence
        inst[:, 1::2] = processing_times
        return _normalise_instance_array(inst)
    except Exception:
        return None


def _extract_instance_array(traj):
    if not isinstance(traj, dict):
        return None
    for key in ("instance", "current_instance"):
        arr = _normalise_instance_array(traj.get(key, None))
        if arr is not None:
            return arr
    n_jobs = traj.get("n_jobs", None)
    n_machines = traj.get("n_machines", None)
    metrics = traj.get("metrics", {})
    if isinstance(metrics, dict):
        n_jobs = n_jobs if n_jobs is not None else metrics.get("n_jobs", None)
        n_machines = n_machines if n_machines is not None else metrics.get("n_machines", None)
        scale = metrics.get("scale", None)
        if (n_jobs is None or n_machines is None) and isinstance(scale, str) and "x" in scale:
            parts = scale.lower().split("x")
            if len(parts) == 2:
                try:
                    n_jobs = int(parts[0])
                    n_machines = int(parts[1])
                except Exception:
                    pass
    states = traj.get("states", [])
    if isinstance(states, list) and states:
        return _instance_from_state_dict(states[0], n_jobs=n_jobs, n_machines=n_machines)
    return None


def _instance_hash(instance):
    arr = _normalise_instance_array(instance)
    if arr is None:
        return None
    h = hashlib.sha1()
    h.update(str(tuple(arr.shape)).encode("ascii"))
    h.update(arr.tobytes())
    return h.hexdigest()


def _trajectory_metadata(traj, row_idx=None, offset=None):
    instance = _extract_instance_array(traj)
    if instance is None:
        fallback = json.dumps(traj, sort_keys=True, separators=(",", ":"), default=str)
        h = hashlib.sha1(fallback.encode("utf-8")).hexdigest()
        scale = "unknown"
        n_jobs = None
        n_machines = None
    else:
        h = _instance_hash(instance)
        n_jobs = int(instance.shape[0])
        n_machines = int(instance.shape[1] // 2)
        scale = _scale_from_shape(n_jobs, n_machines)
    return {
        "row_idx": row_idx,
        "offset": offset,
        "instance_hash": h,
        "scale": scale,
        "n_jobs": n_jobs,
        "n_machines": n_machines,
        "reference_makespan": _extract_reference_makespan(traj),
    }


def _read_lazy_metadata(path, offsets):
    records = []
    errors = 0
    total = len(offsets)
    with open(path, "r", encoding="utf-8") as fh:
        for row_idx, offset in enumerate(offsets):
            try:
                fh.seek(offset)
                traj = json.loads(fh.readline())
                meta = _trajectory_metadata(traj, row_idx=row_idx, offset=offset)
                records.append(meta)
            except Exception as exc:
                errors += 1
                fallback_hash = hashlib.sha1(f"{path}:{offset}".encode("utf-8")).hexdigest()
                records.append({
                    "row_idx": row_idx,
                    "offset": offset,
                    "instance_hash": fallback_hash,
                    "scale": "unknown",
                    "n_jobs": None,
                    "n_machines": None,
                    "reference_makespan": None,
                    "metadata_error": str(exc),
                })
            if (row_idx + 1) % 5000 == 0 or (row_idx + 1) == total:
                print(f"[InstanceHashSplit] parsed metadata {row_idx + 1}/{total}")
    if errors:
        print(f"[InstanceHashSplit] Warning: metadata parse fallback used for {errors} rows")
    return records


def _split_records_by_instance(records, eval_ratio=0.2, seed=42):
    eval_ratio = min(0.9, max(0.0, float(eval_ratio)))
    rng = np.random.default_rng(int(seed))
    groups_by_scale = defaultdict(dict)
    for rec in records:
        scale = str(rec.get("scale") or "unknown").lower()
        h = str(rec.get("instance_hash"))
        group = groups_by_scale[scale].setdefault(h, {"hash": h, "scale": scale, "records": []})
        group["records"].append(rec)

    train_records = []
    eval_records = []
    summary = {
        "split": "instance_hash_scale_stratified",
        "eval_ratio": eval_ratio,
        "seed": int(seed),
        "hash_method": "sha1(shape + int32 current_instance bytes)",
        "rows": int(len(records)),
        "unique_instances": int(sum(len(v) for v in groups_by_scale.values())),
        "scales": {},
    }

    for scale in sorted(groups_by_scale.keys()):
        groups = list(groups_by_scale[scale].values())
        rng.shuffle(groups)
        total_rows = sum(len(g["records"]) for g in groups)
        target_eval_rows = int(round(total_rows * eval_ratio))
        if total_rows >= 2 and eval_ratio > 0:
            target_eval_rows = max(1, min(target_eval_rows, total_rows - 1))
        else:
            target_eval_rows = 0

        scale_train = []
        scale_eval = []
        eval_rows = 0
        for group in groups:
            group_rows = len(group["records"])
            remaining_after_eval = total_rows - (eval_rows + group_rows)
            can_put_eval = target_eval_rows > 0 and eval_rows < target_eval_rows and remaining_after_eval >= 1
            if can_put_eval:
                scale_eval.extend(group["records"])
                eval_rows += group_rows
            else:
                scale_train.extend(group["records"])

        if not scale_eval and len(groups) > 1 and target_eval_rows > 0:
            moved = scale_train[-len(groups[-1]["records"]):]
            scale_train = scale_train[:-len(groups[-1]["records"])]
            scale_eval.extend(moved)
        if not scale_train and scale_eval:
            scale_train.append(scale_eval.pop())

        train_records.extend(scale_train)
        eval_records.extend(scale_eval)
        unique_instances = len(groups)
        summary["scales"][scale] = {
            "rows": int(total_rows),
            "unique_instances": int(unique_instances),
            "duplicate_rows": int(total_rows - unique_instances),
            "train_rows": int(len(scale_train)),
            "eval_rows": int(len(scale_eval)),
            "train_instances": int(len({r["instance_hash"] for r in scale_train})),
            "eval_instances": int(len({r["instance_hash"] for r in scale_eval})),
        }

    train_hashes = {r["instance_hash"] for r in train_records}
    eval_hashes = {r["instance_hash"] for r in eval_records}
    summary["train_rows"] = int(len(train_records))
    summary["eval_rows"] = int(len(eval_records))
    summary["cross_split_instances"] = int(len(train_hashes & eval_hashes))
    rng.shuffle(train_records)
    rng.shuffle(eval_records)
    return train_records, eval_records, summary


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _write_split_manifest(args, summary):
    out_dir = getattr(args, "output_dir", None)
    if not out_dir:
        return
    path = os.path.join(out_dir, "dataset_split_manifest.json")
    try:
        _write_json(path, summary)
        print(f"[InstanceHashSplit] Saved split manifest to {path}")
    except Exception as exc:
        print(f"[InstanceHashSplit] Warning: failed to save split manifest: {exc}")


def _legacy_row_split(items, eval_ratio=0.2, seed=42):
    rng = np.random.default_rng(int(seed))
    items = list(items)
    rng.shuffle(items)
    total_count = len(items)
    if total_count < 2:
        return items, items
    train_size = int(total_count * (1.0 - float(eval_ratio)))
    train_size = max(1, min(train_size, total_count - 1))
    return items[:train_size], items[train_size:]


def _instance_to_panel_item(traj, name=None):
    instance = _extract_instance_array(traj)
    if instance is None:
        return None
    n_jobs = int(instance.shape[0])
    n_machines = int(instance.shape[1] // 2)
    h = _instance_hash(instance)
    return {
        "name": name or h[:10],
        "scale": _scale_from_shape(n_jobs, n_machines).lower(),
        "n_jobs": n_jobs,
        "n_machines": n_machines,
        "instance_hash": h,
        "current_instance": instance.tolist(),
        "reference_makespan": _extract_reference_makespan(traj),
    }


def _select_rollout_panel_scales(args, available_scales):
    requested = _parse_scale_list(getattr(args, "rollout_panel_scales", "auto"))
    available = {str(s).lower() for s in available_scales}
    if not requested:
        return []
    if requested == ["auto"]:
        preferred = ["10x10", "15x10", "20x10", "15x15", "20x15", "20x20"]
        selected = [s for s in preferred if s in available]
        if selected:
            return selected
        eval_jobs, eval_machines = _resolve_rollout_eval_scale(args)
        scale = _scale_from_shape(eval_jobs, eval_machines).lower()
        return [scale] if scale in available else sorted(available)
    return [s for s in requested if s in available]


def _resolve_rollout_panel_size(args, n_scales):
    raw = int(getattr(args, "rollout_panel_size_per_scale", 0) or 0)
    if raw > 0:
        return raw
    if n_scales <= 1:
        return max(1, int(getattr(args, "rollout_episodes", 1) or 1))
    return 1


def _save_rollout_panels(args, panels):
    out_dir = getattr(args, "output_dir", None)
    if not out_dir or not panels:
        return
    jsonl_path = os.path.join(out_dir, "rollout_eval_panel.jsonl")
    manifest_path = os.path.join(out_dir, "rollout_eval_panel_manifest.json")
    try:
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for scale in sorted(panels.keys()):
                for item in panels[scale]:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        manifest = {
            "source": "eval_split",
            "scales": {
                scale: {
                    "instances": len(items),
                    "with_reference_makespan": int(sum(1 for it in items if it.get("reference_makespan") is not None)),
                }
                for scale, items in sorted(panels.items())
            },
        }
        _write_json(manifest_path, manifest)
        print(f"[RolloutPanel] Saved fixed rollout panel to {jsonl_path}")
    except Exception as exc:
        print(f"[RolloutPanel] Warning: failed to save rollout panel: {exc}")


def _load_rollout_panels_from_jsonl(paths):
    panels = defaultdict(list)
    for path in [p.strip() for p in str(paths or "").split(",") if p.strip()]:
        with open(path, "r", encoding="utf-8") as f:
            for row_idx, line in enumerate(f):
                s = line.strip()
                if not s:
                    continue
                obj = json.loads(s)
                item = _instance_to_panel_item(obj, name=obj.get("name", f"{os.path.basename(path)}:{row_idx}"))
                if item is None:
                    continue
                if obj.get("reference_makespan", None) is not None:
                    item["reference_makespan"] = _finite_float(obj.get("reference_makespan"), item.get("reference_makespan"))
                panels[item["scale"]].append(item)
    return dict(panels)


def _build_rollout_panels_from_items(items, args, source_name="eval_split"):
    by_scale = defaultdict(list)
    seen = set()
    for idx, item in enumerate(items):
        if isinstance(item, dict) and "current_instance" in item:
            panel_item = item
        else:
            panel_item = _instance_to_panel_item(item, name=f"{source_name}_{idx}")
        if panel_item is None:
            continue
        scale = str(panel_item["scale"]).lower()
        h = panel_item.get("instance_hash") or _instance_hash(panel_item.get("current_instance"))
        if not h or h in seen:
            continue
        seen.add(h)
        by_scale[scale].append(panel_item)

    selected_scales = _select_rollout_panel_scales(args, by_scale.keys())
    if not selected_scales:
        return {}
    limit = _resolve_rollout_panel_size(args, len(selected_scales))
    panels = {}
    for scale in selected_scales:
        panels[scale] = by_scale.get(scale, [])[:limit]
        if not panels[scale]:
            panels.pop(scale, None)
    _save_rollout_panels(args, panels)
    if panels:
        summary = ", ".join(f"{scale}:{len(items)}" for scale, items in sorted(panels.items()))
        print(f"[RolloutPanel] Fixed eval-split rollout panel: {summary}")
    return panels


def _build_rollout_panels_from_lazy(path, eval_offsets, args):
    items = []
    with open(path, "r", encoding="utf-8") as fh:
        for idx, offset in enumerate(eval_offsets):
            try:
                fh.seek(offset)
                traj = json.loads(fh.readline())
                item = _instance_to_panel_item(traj, name=f"eval_offset_{offset}")
                if item is not None:
                    items.append(item)
            except Exception:
                continue
    return _build_rollout_panels_from_items(items, args, source_name="lazy_eval")


def load_or_generate_trajectories(args):
    """Load pre-collected trajectories or generate synthetic ones using either random or OR-Tools agent."""
    # 确保os模块在函数作用域内可用
    import os

    # 添加详细的调试信息
    print(f"[DEBUG] 开始处理轨迹路径")
    print(f"[DEBUG] 当前工作目录: {os.getcwd()}")

    # 处理轨迹路径，确保路径被正确解析
    trajectories_path = args.trajectories_path
    if trajectories_path:
        # 将路径标准化，去除前后空格
        trajectories_path = trajectories_path.strip()
        # 打印原始路径
        print(f"[DEBUG] 原始轨迹路径: {trajectories_path}")
        # 如果是相对路径，转换为绝对路径
        if not os.path.isabs(trajectories_path):
            trajectories_path = os.path.abspath(trajectories_path)
            print(f"[DEBUG] 转换为绝对路径: {trajectories_path}")
        else:
            print(f"[DEBUG] 已经是绝对路径: {trajectories_path}")

        # 检查路径的各个组成部分
        dir_path = os.path.dirname(trajectories_path)
        filename = os.path.basename(trajectories_path)
        print(f"[DEBUG] 目录路径: {dir_path}")
        print(f"[DEBUG] 文件名: {filename}")

        # 检查目录是否存在
        if os.path.exists(dir_path):
            print(f"[DEBUG] 目录存在")
            # 列出目录中的文件
            try:
                files = os.listdir(dir_path)
                print(f"[DEBUG] 目录中的文件: {files}")
                # 检查是否有类似名称的文件
                similar_files = [f for f in files if 'expert' in f.lower() or 'trajector' in f.lower()]
                if similar_files:
                    print(f"[DEBUG] 发现类似文件: {similar_files}")
            except Exception as e:
                print(f"[DEBUG] 无法列出目录内容: {e}")
        else:
            print(f"[DEBUG] 目录不存在")
    else:
        print(f"[DEBUG] 未提供轨迹路径")

    # 自动处理轨迹路径和生成设置
    file_exists = False
    if trajectories_path:
        file_exists = os.path.exists(trajectories_path)
        print(f"[DEBUG] 文件存在检查: {file_exists}")

    if trajectories_path and file_exists:
        try:
            file_size_mb = os.path.getsize(trajectories_path) / (1024.0 * 1024.0)
            if (
                bool(getattr(args, "lazy_trajectory_loading", True))
                and str(trajectories_path).lower().endswith(".jsonl")
                and file_size_mb >= float(getattr(args, "lazy_trajectory_threshold_mb", 2048.0) or 0.0)
            ):
                print(
                    f"[LazyTrajectoryLoading] Using lazy jsonl loading for {trajectories_path} "
                    f"({file_size_mb:.1f} MB). The file will not be fully parsed into RAM."
                )
                return {"__lazy_jsonl_path__": trajectories_path, "__file_size_mb__": file_size_mb}

            # Load pre-collected trajectories
            with open(trajectories_path, 'r', encoding='utf-8') as f:
                first_non_ws = ''
                while True:
                    ch = f.read(1)
                    if not ch:
                        break
                    if not ch.isspace():
                        first_non_ws = ch
                        break
                f.seek(0)
                if first_non_ws == '[':
                    trajectories = json.load(f)
                else:
                    trajectories = []
                    for line in f:
                        s = line.strip()
                        if not s:
                            continue
                        trajectories.append(json.loads(s))
            print(f"Loaded {len(trajectories)} trajectories from {trajectories_path}")

            def _normalize_action_list(action_list):
                if not isinstance(action_list, list) or len(action_list) == 0:
                    return action_list
                if not all(isinstance(a, (int, float, np.integer, np.floating)) for a in action_list):
                    return action_list
                max_a = int(max(action_list))
                if max_a < int(args.n_jobs):
                    return [int(a) for a in action_list]
                nm = int(getattr(args, "n_machines", 0) or 0)
                nj = int(getattr(args, "n_jobs", 0) or 0)
                if nm > 0 and nj > 0 and max_a < nj * nm:
                    return [int(a) // nm for a in action_list]
                return [int(a) for a in action_list]

            for traj in trajectories:
                if isinstance(traj, dict):
                    if 'actions' in traj:
                        traj['actions'] = _normalize_action_list(traj.get('actions'))
                    if 'expert_actions' in traj:
                        traj['expert_actions'] = _normalize_action_list(traj.get('expert_actions'))

            # 清洗轨迹数据
            print("Cleaning trajectories...")
            trajectories = [clean_trajectory(traj) for traj in trajectories]
            print("Trajectory cleaning completed")

            return trajectories
        except Exception as e:
            print(f"Error loading trajectories from {trajectories_path}: {e}")
            print(f"Falling back to generating new trajectories...")
            args.generate_trajectories = True

    # 当轨迹路径不存在或加载失败时，生成新的轨迹
    args.generate_trajectories = True

    # 执行轨迹生成，无论是因为没有提供路径还是路径不存在
    print(f"Generating new trajectories: {args.generate_trajectories}")

    # 执行轨迹生成
    # Import JSSP environment with enhanced error handling
    import sys
    import os
    JSSPEnv = None

    # 打印当前环境信息以帮助调试
    print(f"[DEBUG] 当前文件路径: {os.path.abspath(__file__)}")
    print(f"[DEBUG] 当前工作目录: {os.getcwd()}")

    # 强制将 dt_train.py 所在目录加入 sys.path 最前面（无论从哪个目录运行）
    _project_root = os.path.dirname(os.path.abspath(__file__))
    for _p in [_project_root, os.getcwd()]:
        if _p not in sys.path:
            sys.path.insert(0, _p)
    print(f"[DEBUG] sys.path[0:3]: {sys.path[:3]}")

    # 尝试多种导入方式
    imported = False
    JSSPEnv = None
    _jssp_env_file = os.path.join(_project_root, 'Envs', 'JSSPEnv', 'JSSPEnv.py')
    print(f"[DEBUG] JSSPEnv.py 是否存在: {os.path.exists(_jssp_env_file)}")

    # 方案1：直接使用文件顶层已成功导入的 JSSPEnv（最可靠）
    try:
        if _TopLevelJSSPEnv is not None:
            JSSPEnv = _TopLevelJSSPEnv
            print(f"[DEBUG] 导入成功: 使用顶层 _TopLevelJSSPEnv")
            imported = True
    except Exception as _e0:
        print(f"[DEBUG] 方案1失败: {_e0}")

    # 方案2：标准导入
    if not imported:
        try:
            from Envs.JSSPEnv.JSSPEnv import JSSPEnv
            print("[DEBUG] 导入成功: from Envs.JSSPEnv.JSSPEnv import JSSPEnv")
            imported = True
        except ImportError as e1:
            # 方案3：importlib 直接从文件加载
            try:
                import importlib.util
                _spec = importlib.util.spec_from_file_location("JSSPEnv_mod", _jssp_env_file)
                _mod = importlib.util.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)
                JSSPEnv = _mod.JSSPEnv
                print(f"[DEBUG] 导入成功: importlib直接加载 {_jssp_env_file}")
                imported = True
            except Exception as e2:
                print(f"[DEBUG] 所有导入尝试失败:")
                print(f"[DEBUG] 尝试1 ImportError: {e1}")
                print(f"[DEBUG] 尝试2 importlib错误: {e2}")
                envs_dir = os.path.join(_project_root, 'Envs')
                print(f"[DEBUG] Envs 目录是否存在: {os.path.exists(envs_dir)}")
                if os.path.exists(envs_dir):
                    print(f"[DEBUG] Envs 目录内容: {os.listdir(envs_dir)}")

    print(f"[DEBUG] 是否成功导入 JSSPEnv: {imported}")

    try:
        if args.use_ortools_agent and imported and JSSPEnv is not None:
            print(f"Generating {args.n_trajectories} optimized JSSP trajectories using OR-Tools agent...")
            if args.ortools_time_limit is not None:
                os.environ["JSSP_ORTOOLS_TIME_LIMIT"] = str(float(args.ortools_time_limit))
                print(f"[DEBUG] OR-Tools time limit: {float(args.ortools_time_limit):.1f}s")

            # Create environment with correct parameters (n_job_range and n_mac_range as lists)
            # Use same value for min and max to create fixed-size instances
            # IMPORTANT: Don't pass seed here - we want each trajectory to have different random seed
            env = JSSPEnv(n_job_range=[args.n_jobs, args.n_jobs], n_mac_range=[args.n_machines, args.n_machines], seed=None)

            # Collect trajectories using OR-Tools agent
            # Use output_dir if provided, otherwise use 'data' directory
            output_dir = args.output_dir if args.output_dir else 'data'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Use custom filename if provided, otherwise generate a unique filename with timestamp
            import time
            if args.trajectory_filename:
                save_path = os.path.join(output_dir, args.trajectory_filename)
            else:
                timestamp = int(time.time())
                if args.mixed_scale_training:
                    save_path = os.path.join(output_dir, f'mixed_scale_jssp_trajectories_{timestamp}.jsonl')
                else:
                    save_path = os.path.join(output_dir, f'ortools_jssp_trajectories_{args.n_jobs}x{args.n_machines}_{timestamp}.jsonl')

            # Collect trajectories using OR-Tools agent or mixed scale training
            if args.mixed_scale_training:
                print(f"Generating {args.n_trajectories} mixed scale JSSP trajectories ({args.mixed_scale_config})...")
                trajectories = collect_mixed_scale_trajectories(
                    env=env,
                    num_trajectories=args.n_trajectories,
                    max_steps=args.max_steps_per_trajectory,
                    output_path=save_path,
                    pure_expert_ratio=args.ortools_pure_expert_ratio,
                    epsilon_min=args.ortools_epsilon_min,
                    epsilon_max=args.ortools_epsilon_max,
                    processing_time_mutation_ratio=args.processing_time_mutation_ratio,
                    processing_time_mutation_pct=args.processing_time_mutation_pct,
                    mixed_scale_config=args.mixed_scale_config,
                    save_light_trajectories=args.save_light_trajectories,
                    validate_light_trajectories=args.validate_light_trajectories,
                )
            else:
                trajectories = collect_ortools_trajectories(
                    env=env,
                    num_trajectories=args.n_trajectories,
                    max_steps=args.max_steps_per_trajectory,
                    output_path=save_path,
                    pure_expert_ratio=args.ortools_pure_expert_ratio,
                    epsilon_min=args.ortools_epsilon_min,
                    epsilon_max=args.ortools_epsilon_max,
                    processing_time_mutation_ratio=args.processing_time_mutation_ratio,
                    processing_time_mutation_pct=args.processing_time_mutation_pct,
                    use_sequence_swap=args.use_sequence_swap,  # 使用新的序列交换方法
                    save_light_trajectories=args.save_light_trajectories,
                    validate_light_trajectories=args.validate_light_trajectories,
                )
            print(f"Generated {len(trajectories)} trajectories")

            # 清洗轨迹数据
            print("Cleaning trajectories...")
            trajectories = [clean_trajectory(traj) for traj in trajectories]
            print("Trajectory cleaning completed")
        else:
            # Generate synthetic trajectories using default method
            print(f"Generating {args.n_trajectories} synthetic JSSP trajectories...")
            if not os.path.exists('data'):
                os.makedirs('data')

            # Use custom filename if provided, otherwise generate a unique filename with timestamp
            import time
            if args.trajectory_filename:
                save_path = os.path.join('data', args.trajectory_filename)
            else:
                timestamp = int(time.time())
                save_path = os.path.join('data', f'synthetic_jssp_trajectories_{args.n_jobs}x{args.n_machines}_{timestamp}.json')

            trajectories = generate_synthetic_jssp_trajectories(
                n_trajectories=args.n_trajectories,
                n_jobs=args.n_jobs,
                n_machines=args.n_machines,
                max_steps=args.max_steps_per_trajectory,
                output_path=save_path
            )
            print(f"Generated {len(trajectories)} synthetic trajectories")

            # 清洗轨迹数据
            print("Cleaning trajectories...")
            trajectories = [clean_trajectory(traj) for traj in trajectories]
            print("Trajectory cleaning completed")

            # Trajectories are already saved incrementally during collection/generation
            print(f"Total trajectories saved to {save_path}: {len(trajectories)}")

    except Exception as e:
        print(f"Error generating trajectories: {e}")
        # Fallback to original generation method if OR-Tools generation fails
        print("Falling back to default synthetic trajectory generation...")
        trajectories = generate_synthetic_jssp_trajectories(
            n_trajectories=args.n_trajectories,
            n_jobs=args.n_jobs,
            n_machines=args.n_machines,
            max_steps=args.max_steps_per_trajectory
        )

        # 清洗轨迹数据
        print("Cleaning trajectories...")
        trajectories = [clean_trajectory(traj) for traj in trajectories]
        print("Trajectory cleaning completed")

        # Save fallback trajectories
        if not os.path.exists('data'):
            os.makedirs('data')

        # Use custom filename for fallback trajectories if provided
        import time
        if args.trajectory_filename:
            save_path = os.path.join('data', args.trajectory_filename)
        else:
            timestamp = int(time.time())
            save_path = os.path.join('data', f'fallback_jssp_trajectories_{args.n_jobs}x{args.n_machines}_{timestamp}.json')

        serializable_trajectories = []
        for traj in trajectories:
            serializable_traj = {}
            for key, value in traj.items():
                if isinstance(value, np.ndarray):
                    serializable_traj[key] = value.tolist()
                elif isinstance(value, list) and value and isinstance(value[0], np.ndarray):
                    serializable_traj[key] = [item.tolist() for item in value]
                else:
                    serializable_traj[key] = value
            serializable_trajectories.append(serializable_traj)

        with open(save_path, 'w') as f:
            json.dump(serializable_trajectories, f)
        print(f"Saved fallback trajectories to {save_path}")

    # 确保我们有轨迹可以返回
    if 'trajectories' not in locals():
        raise ValueError("Failed to generate trajectories. Please check your environment and parameters.")

    return trajectories


def create_dt_datasets(trajectories, args):
    """Create training and evaluation datasets for Decision Transformer."""
    if _is_lazy_trajectory_source(trajectories):
        model_max_jobs, model_max_machines = _resolve_model_capacity(args)
        path = trajectories["__lazy_jsonl_path__"]
        print(f"[LazyTrajectoryLoading] Building line-offset index for {path}")
        offsets = build_jsonl_offsets(path)
        if not offsets:
            raise ValueError(f"No trajectories found in lazy jsonl file: {path}")

        offsets = list(offsets)

        split_seed = int(getattr(args, "seed", 42))
        eval_ratio = float(getattr(args, "eval_ratio", 0.2) or 0.2)
        if bool(getattr(args, "instance_hash_split", True)):
            print("[InstanceHashSplit] Scanning lazy jsonl metadata for grouped scale-stratified split...")
            records = _read_lazy_metadata(path, offsets)
            train_records, eval_records, split_summary = _split_records_by_instance(
                records,
                eval_ratio=eval_ratio,
                seed=split_seed,
            )
            train_offsets = [int(r["offset"]) for r in train_records]
            eval_offsets = [int(r["offset"]) for r in eval_records]
            _write_split_manifest(args, split_summary)
            if split_summary.get("cross_split_instances", 0) != 0:
                print(f"[InstanceHashSplit] WARNING: cross split instances = {split_summary['cross_split_instances']}")
            else:
                print("[InstanceHashSplit] Verified: 0 instance hashes shared by train/eval")
        else:
            train_offsets, eval_offsets = _legacy_row_split(offsets, eval_ratio=eval_ratio, seed=split_seed)

        max_actions = int(getattr(args, "n_jobs", 0) or model_max_jobs or 0)
        if max_actions <= 0:
            max_actions = 10

        augmentation_mode = getattr(args, 'augmentation_mode', 'offline')
        enable_online_aug = bool(getattr(args, 'use_data_augmentation', False)) and augmentation_mode in ('online', 'both')
        if bool(getattr(args, 'use_data_augmentation', False)) and augmentation_mode in ('offline', 'both'):
            print("[LazyTrajectoryLoading] Offline augmentation is skipped for lazy datasets to keep memory bounded.")
        if bool(getattr(args, "use_curriculum_learning", False)):
            print("[LazyTrajectoryLoading] Curriculum learning is disabled for lazy datasets.")
            args.use_curriculum_learning = False

        train_dataset = LazyJSSPDecisionTransformerDataset(
            path,
            train_offsets,
            state_dim=args.state_dim,
            act_dim=max_actions,
            max_length=args.max_length,
            max_jobs=model_max_jobs,
            max_machines=model_max_machines,
            enable_augmentation=enable_online_aug,
            augmentation_ratio=getattr(args, 'augmentation_ratio', 0.3),
            state_noise_std=getattr(args, 'state_noise_std', 0.01),
            rtg_scale=getattr(args, 'rtg_scale', 1.1),
            rtg_mode=args.training_rtg_mode,
            use_gap_weighted_loss=args.use_gap_weighted_loss,
            gap_weight_alpha=args.gap_weight_alpha,
            gap_weight_power=args.gap_weight_power,
            quality_weight_mix=args.quality_weight_mix,
            min_sample_weight=args.min_sample_weight,
            max_sample_weight=args.max_sample_weight,
            state_tensorization_mode=args.state_tensorization_mode,
        )
        eval_dataset = LazyJSSPDecisionTransformerDataset(
            path,
            eval_offsets,
            state_dim=args.state_dim,
            act_dim=max_actions,
            max_length=args.max_length,
            max_jobs=model_max_jobs,
            max_machines=model_max_machines,
            enable_augmentation=False,
            rtg_scale=getattr(args, 'rtg_scale', 1.0) if bool(getattr(args, 'use_data_augmentation', False)) else 1.0,
            rtg_mode=args.training_rtg_mode,
            use_gap_weighted_loss=args.use_gap_weighted_loss,
            gap_weight_alpha=args.gap_weight_alpha,
            gap_weight_power=args.gap_weight_power,
            quality_weight_mix=args.quality_weight_mix,
            min_sample_weight=args.min_sample_weight,
            max_sample_weight=args.max_sample_weight,
            state_tensorization_mode=args.state_tensorization_mode,
        )
        train_dataset.scale_labels = [str(r.get("scale", "unknown")).lower() for r in train_records]
        eval_dataset.scale_labels = [str(r.get("scale", "unknown")).lower() for r in eval_records]

        print(f"Detected max action space: {max_actions}")
        print(f"Training on {len(train_dataset)} lazy trajectories, evaluating on {len(eval_dataset)}")
        if getattr(args, "rollout_panel_jsonl", None):
            args._rollout_eval_panels = _load_rollout_panels_from_jsonl(args.rollout_panel_jsonl)
        else:
            args._rollout_eval_panels = _build_rollout_panels_from_lazy(path, eval_offsets, args)
        return train_dataset, eval_dataset, train_offsets, eval_offsets, max_actions

    if trajectories is None or len(trajectories) == 0:
        raise ValueError("No trajectories available for dataset creation.")
    model_max_jobs, model_max_machines = _resolve_model_capacity(args)

    # 动态计算最大动作数
    max_actions = 0
    for traj in trajectories:
        max_actions = max(max_actions, max(traj['actions']) + 1)  # +1 因为动作从 0 开始
    print(f"Detected max action space: {max_actions}")

    split_seed = int(getattr(args, "seed", 42))
    eval_ratio = float(getattr(args, "eval_ratio", 0.2) or 0.2)
    if bool(getattr(args, "instance_hash_split", True)):
        records = []
        for idx, traj in enumerate(trajectories):
            meta = _trajectory_metadata(traj, row_idx=idx, offset=None)
            records.append(meta)
        train_records, eval_records, split_summary = _split_records_by_instance(
            records,
            eval_ratio=eval_ratio,
            seed=split_seed,
        )
        train_trajectories = [trajectories[int(r["row_idx"])] for r in train_records]
        eval_trajectories = [trajectories[int(r["row_idx"])] for r in eval_records]
        _write_split_manifest(args, split_summary)
        if split_summary.get("cross_split_instances", 0) != 0:
            print(f"[InstanceHashSplit] WARNING: cross split instances = {split_summary['cross_split_instances']}")
        else:
            print("[InstanceHashSplit] Verified: 0 instance hashes shared by train/eval")
    else:
        train_trajectories, eval_trajectories = _legacy_row_split(
            trajectories,
            eval_ratio=eval_ratio,
            seed=split_seed,
        )

    print(f"Training on {len(train_trajectories)} trajectories, evaluating on {len(eval_trajectories)}")

    augmentation_mode = getattr(args, 'augmentation_mode', 'offline')
    enable_offline_aug = bool(getattr(args, 'use_data_augmentation', False)) and augmentation_mode in ('offline', 'both')
    enable_online_aug = bool(getattr(args, 'use_data_augmentation', False)) and augmentation_mode in ('online', 'both')

    # 警告：online aug 会导致训练/eval 数据分布不一致
    # 训练 batch 每次动态加噪声，eval 是干净数据，模型实际在噪声数据上过拟合
    # 强烈建议使用 offline 模式（一次性扩充数据集）
    if enable_online_aug:
        print("[WARNING] augmentation_mode=online/both 检测到！训练时动态加噪声会导致"
              "训练与eval分布不一致（eval acc 虚低），建议改用 --augmentation_mode offline")

    # 数据增强（离线扩充）
    if enable_offline_aug:
        print("\n🔧 应用数据增强...")
        try:
            from dataset_optimization import DatasetOptimizer
            
            optimizer = DatasetOptimizer(train_trajectories)
            augmented_trajectories = optimizer.apply_all_augmentations(
                augmentation_ratio=getattr(args, 'augmentation_ratio', 0.3),
                noise_std=getattr(args, 'state_noise_std', 0.01),
                rtg_scale=getattr(args, 'rtg_scale', 1.1)
            )
            
            stats = optimizer.get_statistics()
            print(f"  原始轨迹: {len(train_trajectories)}")
            print(f"  增强后轨迹: {stats['n_trajectories']}")
            print(f"  增强轨迹数: {stats['augmentation_count']}")
            print(f"  平均效率: {stats['avg_efficiency']:.3f}")
            
            train_trajectories = augmented_trajectories
        except ImportError:
            print("  ⚠️ 无法导入 dataset_optimization 模块，跳过数据增强")
        except Exception as e:
            print(f"  ⚠️ 数据增强失败: {e}")

    # Process trajectories for Decision Transformer
    train_dataset = process_jssp_trajectories_for_dt(
        train_trajectories,
        state_dim=args.state_dim,
        act_dim=max_actions,  # 使用动态计算的最大动作数
        max_length=args.max_length,
        enable_augmentation=enable_online_aug,
        augmentation_ratio=getattr(args, 'augmentation_ratio', 0.3),
        state_noise_std=getattr(args, 'state_noise_std', 0.01),
        rtg_scale=getattr(args, 'rtg_scale', 1.1),
        max_jobs=model_max_jobs,
        max_machines=model_max_machines,
        rtg_mode=args.training_rtg_mode,
        use_gap_weighted_loss=args.use_gap_weighted_loss,
        gap_weight_alpha=args.gap_weight_alpha,
        gap_weight_power=args.gap_weight_power,
        quality_weight_mix=args.quality_weight_mix,
        min_sample_weight=args.min_sample_weight,
        max_sample_weight=args.max_sample_weight,
        state_tensorization_mode=args.state_tensorization_mode,
    )

    # eval_dataset 不做随机增强（保持确定性），但 rtg_scale 须与训练集一致
    # 否则模型在训练时见到的 RTG 是放大后的值，推理/eval 时 RTG 偏小，行为模式不一致
    eval_rtg_scale = getattr(args, 'rtg_scale', 1.0) if bool(getattr(args, 'use_data_augmentation', False)) else 1.0
    eval_dataset = process_jssp_trajectories_for_dt(
        eval_trajectories,
        state_dim=args.state_dim,
        act_dim=max_actions,  # 使用动态计算的最大动作数
        max_length=args.max_length,
        enable_augmentation=False,
        rtg_scale=eval_rtg_scale,  # 与训练集 RTG 保持同一尺度，消除分布偏移
        max_jobs=model_max_jobs,
        max_machines=model_max_machines,
        rtg_mode=args.training_rtg_mode,
        use_gap_weighted_loss=args.use_gap_weighted_loss,
        gap_weight_alpha=args.gap_weight_alpha,
        gap_weight_power=args.gap_weight_power,
        quality_weight_mix=args.quality_weight_mix,
        min_sample_weight=args.min_sample_weight,
        max_sample_weight=args.max_sample_weight,
        state_tensorization_mode=args.state_tensorization_mode,
    )
    if bool(getattr(args, "instance_hash_split", True)):
        train_dataset.scale_labels = [str(r.get("scale", "unknown")).lower() for r in train_records]
        eval_dataset.scale_labels = [str(r.get("scale", "unknown")).lower() for r in eval_records]

    if bool(getattr(args, "use_gap_weighted_loss", False)):
        try:
            train_w = np.array(getattr(train_dataset, "sample_weights", []), dtype=float)
            if train_w.size > 0:
                print(
                    f"[GapWeightedLoss] train sample weights: "
                    f"mean={train_w.mean():.3f}, min={train_w.min():.3f}, max={train_w.max():.3f}, p90={np.percentile(train_w, 90):.3f}"
                )
        except Exception as e:
            print(f"[GapWeightedLoss] Warning: failed to summarize sample weights: {e}")

    if getattr(args, "rollout_panel_jsonl", None):
        args._rollout_eval_panels = _load_rollout_panels_from_jsonl(args.rollout_panel_jsonl)
    else:
        args._rollout_eval_panels = _build_rollout_panels_from_items(eval_trajectories, args, source_name="eval")

    return train_dataset, eval_dataset, train_trajectories, eval_trajectories, max_actions


def _trajectory_difficulty(traj):
    metrics = traj.get("metrics", {}) if isinstance(traj, dict) else {}
    n_jobs = metrics.get("n_jobs", None)
    n_machines = metrics.get("n_machines", None)
    scale = metrics.get("scale", None)
    if (n_jobs is None or n_machines is None) and isinstance(scale, str) and "x" in scale:
        parts = scale.lower().split("x")
        if len(parts) == 2:
            try:
                n_jobs = int(parts[0])
                n_machines = int(parts[1])
            except Exception:
                n_jobs = None
                n_machines = None
    if n_jobs is None or n_machines is None:
        states = traj.get("states", []) if isinstance(traj, dict) else []
        if isinstance(states, list) and len(states) > 0 and isinstance(states[0], dict):
            s0 = states[0]
            try:
                n_jobs = int(len(s0.get("job_next_op", [])))
            except Exception:
                n_jobs = None
            try:
                n_machines = int(len(s0.get("normalized_machine_availability", [])))
            except Exception:
                n_machines = None
    if n_jobs is None:
        n_jobs = 0
    if n_machines is None:
        n_machines = 0
    eff = float(metrics.get("efficiency_ratio", 0.0) or 0.0)
    return (int(n_jobs) * int(n_machines), -eff)


def build_curriculum_loader(train_trajectories, args, max_actions, epoch_idx, total_epochs):
    """
    改进版课程学习数据加载器。

    策略：
    - Stage 1（前1/3 epoch）：只使用「简单」轨迹（10x5规模 + 高效率），让模型先学会基础调度规律。
    - Stage 2（中1/3 epoch）：引入全部10x5 + 少量10x10（按难度排序前70%），混合比例7:3与数据集一致。
    - Stage 3（后1/3 epoch）：使用全量轨迹，同时对10x10轨迹上采样到与10x5等比，强迫模型关注大规模。

    难度排序指标：规模（n_jobs * n_machines，小规模优先）+ 效率比（高效率优先）。
    """
    stages = max(1, int(getattr(args, "curriculum_stages", 3) or 3))
    cur_stage = min(stages - 1, int((epoch_idx * stages) / max(1, total_epochs)))
    ratio = float(cur_stage + 1) / float(stages)

    # 按难度排序（小规模、高效率排在前面）
    ordered = sorted(list(train_trajectories), key=_trajectory_difficulty)

    if stages >= 3:
        if cur_stage == 0:
            # Stage 1：只取前40%（最简单的轨迹，以10x5为主）
            keep_n = max(1, int(len(ordered) * 0.4))
            selected = ordered[:keep_n]
            print(f"[Curriculum] Stage 1/{stages} (epoch {epoch_idx+1}): "
                  f"使用最简单的 {len(selected)}/{len(ordered)} 条轨迹（10x5为主）")
        elif cur_stage == 1:
            # Stage 2：取前70%，同时确保10x10轨迹不超过30%
            keep_n = max(1, int(len(ordered) * 0.7))
            selected_all = ordered[:keep_n]
            # 分离不同规模
            small_scale = [t for t in selected_all if _trajectory_difficulty(t)[0] <= 50]  # <=10x5
            large_scale = [t for t in selected_all if _trajectory_difficulty(t)[0] > 50]   # >10x5
            # 控制大规模比例不超过30%
            max_large = max(1, int(len(selected_all) * 0.3))
            large_scale = large_scale[:max_large]
            selected = small_scale + large_scale
            print(f"[Curriculum] Stage 2/{stages} (epoch {epoch_idx+1}): "
                  f"小规模 {len(small_scale)} + 大规模 {len(large_scale)} = {len(selected)} 条轨迹")
        else:
            # Stage 3：全量轨迹，对大规模轨迹上采样使其比例提升到50%
            small_scale = [t for t in ordered if _trajectory_difficulty(t)[0] <= 50]
            large_scale = [t for t in ordered if _trajectory_difficulty(t)[0] > 50]
            # 上采样大规模轨迹
            if large_scale and small_scale:
                target_large = len(small_scale)  # 让大规模与小规模数量相当
                import random
                if len(large_scale) < target_large:
                    # 重复采样补足
                    extra = random.choices(large_scale, k=target_large - len(large_scale))
                    large_scale = large_scale + extra
                selected = small_scale + large_scale
            else:
                selected = ordered
            print(f"[Curriculum] Stage 3/{stages} (epoch {epoch_idx+1}): "
                  f"全量 + 大规模上采样，共 {len(selected)} 条轨迹")
    else:
        # 两阶段或单阶段：简单按比例选取
        keep_n = max(1, int(len(ordered) * ratio))
        selected = ordered[:keep_n]
        print(f"[Curriculum] Stage {cur_stage+1}/{stages} (epoch {epoch_idx+1}): "
              f"使用 {len(selected)}/{len(ordered)} 条轨迹 (ratio={ratio:.2f})")

    train_dataset = process_jssp_trajectories_for_dt(
        selected,
        state_dim=args.state_dim,
        act_dim=max_actions,
        max_length=args.max_length,
        enable_augmentation=bool(getattr(args, 'use_data_augmentation', False)) and getattr(args, 'augmentation_mode', 'offline') in ('online', 'both'),
        augmentation_ratio=getattr(args, 'augmentation_ratio', 0.3),
        state_noise_std=getattr(args, 'state_noise_std', 0.01),
        rtg_scale=getattr(args, 'rtg_scale', 1.1),
        max_jobs=int(getattr(args, "n_jobs", 0) or 0),
        max_machines=int(getattr(args, "n_machines", 0) or 0),
        rtg_mode=getattr(args, "training_rtg_mode", "normalized_budget"),
        use_gap_weighted_loss=args.use_gap_weighted_loss,
        gap_weight_alpha=args.gap_weight_alpha,
        gap_weight_power=args.gap_weight_power,
        quality_weight_mix=args.quality_weight_mix,
        min_sample_weight=args.min_sample_weight,
        max_sample_weight=args.max_sample_weight,
        state_tensorization_mode=args.state_tensorization_mode,
    )
    return create_data_loader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        num_workers=0
    )


def process_state_for_batch(
    states_history,
    device,
    max_jobs=None,
    max_machines=None,
    state_tensorization_mode=None,
):
    """
    将状态历史转换为模型可以接受的批处理格式。
    支持 JSSP 的 2D 状态 (n_jobs, n_machines) 和展平状态 (state_dim)
    """
    import torch
    import numpy as np

    if not states_history:
        return None

    first_state = states_history[0]

    # 处理 JSSP 2D 状态 (n_jobs, n_machines)
    if isinstance(first_state, np.ndarray) and first_state.ndim == 2:
        # 转换为张量并添加 batch/sequence 维度
        tensor_list = []
        for state in states_history:
            if isinstance(state, np.ndarray):
                state_tensor = torch.tensor(state, dtype=torch.float32)
            elif isinstance(state, torch.Tensor):
                state_tensor = state.clone()
            else:
                # 处理其他类型（list等）
                state_tensor = torch.tensor(state, dtype=torch.float32)
            # 添加 batch 和 sequence 维度 -> (1, 1, n_jobs, n_machines)
            state_tensor = state_tensor.unsqueeze(0).unsqueeze(0)
            tensor_list.append(state_tensor)

        # 拼接所有时间步 -> (1, T, n_jobs, n_machines)
        model_states = torch.cat(tensor_list, dim=1)
        return model_states.to(device)

    # 处理展平状态 (state_dim)
    elif isinstance(first_state, (np.ndarray, torch.Tensor)) and first_state.ndim == 1:
        if isinstance(first_state, np.ndarray):
            states_tensor = torch.stack([
                torch.tensor(s, dtype=torch.float32) for s in states_history
            ])
        elif isinstance(first_state, torch.Tensor):
            states_tensor = torch.stack(states_history)
        else:
            # 处理其他类型（int, float, list等）
            states_tensor = torch.stack([
                torch.tensor(s, dtype=torch.float32) for s in states_history
            ])

        # 确保是 3D (B, T, D)
        if states_tensor.dim() == 2:
            states_tensor = states_tensor.unsqueeze(0)  # (1, T, D)
        return states_tensor.to(device)

    # Graph dictionary states must use exactly the same tensorization as the
    # dataset and external inference paths.
    elif isinstance(first_state, dict):
        actual_jobs = int(len(first_state.get("job_next_op", [])) or 0)
        actual_machines = int(
            len(first_state.get("normalized_machine_availability", [])) or 0
        )
        capacity_jobs = int(max_jobs or actual_jobs)
        capacity_machines = int(max_machines or actual_machines)
        history = tensorize_graph_state_history(
            states_history,
            actual_n_jobs=actual_jobs,
            actual_n_machines=actual_machines,
            max_jobs=capacity_jobs,
            max_machines=capacity_machines,
            keys=GRAPH_STATE_KEYS,
            mode=state_tensorization_mode or get_state_tensorization_mode(),
        )
        return {
            key: value.unsqueeze(0).to(device)
            for key, value in history.items()
        }

    else:
        raise ValueError(f"Unsupported state type: {type(first_state)}")


def evaluate_rollout(
    dt_model,
    env,
    num_episodes=10,
    target_return=1.0,
    rtg_mode="normalized_budget",
    debug_prints=False,
    rollout_top_k=1,
    rollout_temperature=0.0,
    rollout_best_of_n=1,
    quiet_summary=False,
    ortools_makespan_override=None,
    rollout_crop_history=True,
    rollout_eval_max_length=64,
):
    """
    使用当前模型进行 rollout，返回平均相对gap（模型makespan与OR-Tools最优makespan的相对差值）。

    Args:
        dt_model: 训练中的 DecisionTransformer 模型
        env: JSSPEnv 环境（支持 reset() 和 step(action)）
        num_episodes: 测试多少个不同的实例
        target_return: RTG 条件强度。constant 模式下表示固定负 RTG 的绝对值；
                       budget / normalized_budget 模式下表示目标 makespan 的缩放系数。

    Returns:
        avg_gap: 平均相对gap（(模型makespan - OR-Tools最优makespan) / OR-Tools最优makespan * 100），单位：%
    """
    import torch
    import builtins
    from Envs.JSSPEnv.JSSPEnv import solve_jssp
    import numpy as np

    if int(rollout_best_of_n or 1) > 1:
        return evaluate_rollout_best_of_n(
            dt_model=dt_model,
            env=env,
            num_episodes=num_episodes,
            target_return=target_return,
            rtg_mode=rtg_mode,
            debug_prints=debug_prints,
            rollout_top_k=rollout_top_k,
            rollout_temperature=rollout_temperature,
            rollout_best_of_n=rollout_best_of_n,
            rollout_crop_history=rollout_crop_history,
            rollout_eval_max_length=rollout_eval_max_length,
        )

    rtg_mode = str(rtg_mode).lower().strip()
    if rtg_mode not in {"constant", "budget", "normalized_budget"}:
        raise ValueError(f"Unknown rtg_mode={rtg_mode}. Use 'constant', 'budget' or 'normalized_budget'.")

    if not debug_prints:
        def print(*args, **kwargs):
            return None

    # 验证 state_encoder 是否存在
    print(f"[DEBUG] Model has state_encoder: {hasattr(dt_model, 'state_encoder') and dt_model.state_encoder is not None}")
    print(f"[DEBUG] state_encoder type: {type(dt_model.state_encoder) if hasattr(dt_model, 'state_encoder') else 'N/A'}")
    print(f"[DEBUG] state_encoder is None: {dt_model.state_encoder is None if hasattr(dt_model, 'state_encoder') else 'N/A'}")

    if not (hasattr(dt_model, 'state_encoder') and dt_model.state_encoder is not None):
        raise RuntimeError("state_encoder is not set! Please use create_decision_transformer_for_jssp to create the model.")

    # 验证state_encoder是否是nn.Module
    if hasattr(dt_model, 'state_encoder') and dt_model.state_encoder is not None:
        print(f"[DEBUG] state_encoder is nn.Module: {isinstance(dt_model.state_encoder, torch.nn.Module)}")
        if isinstance(dt_model.state_encoder, torch.nn.Module):
            print(f"[DEBUG] state_encoder parameters count: {sum(p.numel() for p in dt_model.state_encoder.parameters())}")

    device = next(dt_model.parameters()).device
    dt_model.eval()
    total_gap = 0.0
    valid_episodes = 0  # 新增：记录有效实例数量

    print(f"\n[DEBUG] Starting rollout evaluation with {num_episodes} episodes...")
    target_valid_episodes = int(num_episodes)
    max_attempts = max(1, target_valid_episodes * 10)
    attempts = 0
    skip_reasons = {
        "ortools_infeasible": 0,
        "ortools_failed": 0,
        "not_completed": 0,
        "action_prediction_failed": 0,
        "env_step_failed": 0,
    }

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with torch.inference_mode():
        while valid_episodes < target_valid_episodes and attempts < max_attempts:
            attempts += 1
            print(f"\n[DEBUG] Episode {valid_episodes+1}/{target_valid_episodes} (attempt {attempts}/{max_attempts}):")

            state = env.reset()
            done = False
            total_ops = int(env.n_jobs) * int(env.n_machines)
            env.max_steps = max(int(getattr(env, "max_steps", 0) or 0), total_ops + 5)

            # 获取当前实例，用于OR-Tools求解
            current_instance = env.current_instance.copy()
            lower_bound = _jssp_makespan_lower_bound(current_instance)
            print(f"[DEBUG] Current instance shape: {current_instance.shape}")

            # 使用OR-Tools求解最优makespan
            try:
                if ortools_makespan_override is not None:
                    ortools_makespan = float(ortools_makespan_override)
                else:
                    print(f"[DEBUG] Calling solve_jssp...")
                    ortools_result = solve_jssp(current_instance)
                    if isinstance(ortools_result, str) and ortools_result == "infeasible":
                        print(f"[DEBUG] OR-Tools returned infeasible, skipping")
                        skip_reasons["ortools_infeasible"] += 1
                        continue  # 跳过不可行实例
                    else:
                        # 处理返回值，可能是单个值或元组
                        if isinstance(ortools_result, tuple):
                            ortools_makespan = float(ortools_result[0])
                        else:
                            ortools_makespan = float(ortools_result)
                print(f"[DEBUG] OR-Tools makespan: {ortools_makespan}, lower_bound: {lower_bound}")
            except Exception as e:
                print(f"[DEBUG] OR-Tools求解失败: {e}")
                skip_reasons["ortools_failed"] += 1
                continue

            conditioned_budget = None
            if rtg_mode in {"budget", "normalized_budget"}:
                budget_scale = float(target_return) if target_return is not None else 1.0
                conditioned_budget = max(1.0, float(ortools_makespan) * budget_scale)

            if rtg_mode == "constant":
                # constant 模式保持旧版语义，输入固定负 RTG 条件。
                current_rtg = -float(target_return) if target_return is not None else -1.0
            elif rtg_mode == "budget":
                current_rtg = max(0.0, conditioned_budget - float(getattr(env, "current_time", 0.0)))
            else:
                remaining_budget = max(0.0, conditioned_budget - float(getattr(env, "current_time", 0.0)))
                current_rtg = -remaining_budget / conditioned_budget

            t = 0
            max_steps = max(200, total_ops * 3)  # 包含等待推进步，避免20x10在默认100/200处被截断

            # --- 新增：初始化推理缓存 ---
            states_history = [state]  # 初始状态(dict)，长度为1
            # 推理时使用模型动作空间末尾的 Start/Padding Token 作为初始动作占位符。
            # 小规模 env 也必须使用 dt_model.n_jobs，否则会被模型误读成一个真实 job id。
            padding_action = int(getattr(dt_model, "n_jobs", env.n_jobs))
            actions_history = [padding_action]
            rtgs_history = [current_rtg]  # 使用 current_rtg，长度为1
            timesteps_history = [0]  # 初始时间步，长度为1
            # --- 初始化结束 ---

            job_completed = False  # 初始化job_completed变量，解决作用域问题

            while t < max_steps:
                # --- 准备模型输入 ---
                # 使用当前历史（不包括即将添加的新状态）
                current_seq_len = len(states_history)
                print(f"[DEBUG] Step {t}, current_seq_len={current_seq_len}, states_hist_len={len(states_history)}, acts_hist_len={len(actions_history)}, rtgs_hist_len={len(rtgs_history)}, timesteps_hist_len={len(timesteps_history)}")

                # === 关键修复：确保所有历史列表长度一致 ===
                # 调整 actions_history 长度以匹配 states_history
                if len(actions_history) > current_seq_len:
                    actions_history = actions_history[:current_seq_len]
                elif len(actions_history) < current_seq_len:
                    # 用 Start Token 占位，避免把合法动作0误当成历史动作
                    actions_history.extend([padding_action] * (current_seq_len - len(actions_history)))

                # 调整 rtgs_history 长度以匹配 states_history
                if len(rtgs_history) > current_seq_len:
                    rtgs_history = rtgs_history[:current_seq_len]
                elif len(rtgs_history) < current_seq_len:
                    # 用最后一个值补齐
                    rtgs_history.extend([rtgs_history[-1]] * (current_seq_len - len(rtgs_history)))

                # 调整 timesteps_history 长度以匹配 states_history
                if len(timesteps_history) > current_seq_len:
                    timesteps_history = timesteps_history[:current_seq_len]
                elif len(timesteps_history) < current_seq_len:
                    # 用最后一个值+1的序列补齐
                    last_timestep = timesteps_history[-1]
                    timesteps_history.extend([last_timestep + i for i in range(1, current_seq_len - len(timesteps_history) + 1)])

                # 验证所有历史列表长度一致
                assert len(states_history) == len(actions_history) == len(rtgs_history) == len(timesteps_history), \
                    f"History lengths mismatch: states={len(states_history)}, actions={len(actions_history)}, rtgs={len(rtgs_history)}, timesteps={len(timesteps_history)}"
                print(f"[DEBUG] After history adjustment - states: {len(states_history)}, actions: {len(actions_history)}, rtgs: {len(rtgs_history)}, timesteps: {len(timesteps_history)}")

                if rollout_crop_history:
                    override_limit = int(rollout_eval_max_length or 0)
                    history_limit = override_limit if override_limit > 0 else int(getattr(dt_model, "max_length", 0) or 0)
                    if history_limit > 0 and len(states_history) > history_limit:
                        states_view = states_history[-history_limit:]
                        actions_view = actions_history[-history_limit:]
                        rtgs_view = rtgs_history[-history_limit:]
                        timesteps_view = timesteps_history[-history_limit:]
                    else:
                        states_view = states_history
                        actions_view = actions_history
                        rtgs_view = rtgs_history
                        timesteps_view = timesteps_history
                else:
                    states_view = states_history
                    actions_view = actions_history
                    rtgs_view = rtgs_history
                    timesteps_view = timesteps_history

                # 准备模型输入
                encoder = getattr(dt_model, "state_encoder", None)
                model_states = process_state_for_batch(
                    states_view,
                    device,
                    max_jobs=getattr(encoder, "max_jobs", None),
                    max_machines=getattr(encoder, "max_machines", None),
                    state_tensorization_mode=get_state_tensorization_mode(),
                )

                # 获取模型数据类型并转换
                model_dtype = next(dt_model.parameters()).dtype

                # 转换数据类型
                if isinstance(model_states, dict):
                    model_states = {k: v.to(dtype=model_dtype) for k, v in model_states.items()}
                else:
                    model_states = model_states.to(dtype=model_dtype)

                # 调试打印
                print(f"[DEBUG] Processed state shape: {model_states.shape if not isinstance(model_states, dict) else {k: v.shape for k, v in model_states.items()}}")
                print(f"[DEBUG] Model dtype: {model_dtype}")

                # 验证所有状态组件都是 3D
                if isinstance(model_states, dict):
                    for k, v in model_states.items():
                        assert v.dim() == 3, f"Component '{k}' has shape {v.shape}, expected 3D!"
                    print("[DEBUG] All state components are 3D. Safe to proceed.")

                # 使用完整的 actions_history 来准备模型输入
                model_actions = torch.tensor(actions_view, dtype=torch.long, device=device).unsqueeze(0)
                print(f"[DEBUG] model_actions shape: {model_actions.shape}")

                # 将RTG和时间步转换为tensor
                model_rtgs = torch.tensor(rtgs_view, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)
                model_timesteps = torch.tensor(timesteps_view, dtype=torch.long, device=device).unsqueeze(0)

                # 创建attention mask
                attention_mask = torch.ones_like(model_timesteps)

                # === 关键：验证所有张量的序列长度一致 ===
                seq_len = model_rtgs.shape[1]  # 获取RTG的序列长度

                # 验证所有张量的序列长度都相同
                print(f"[DEBUG] Sequence length check - RTG: {model_rtgs.shape[1]}, Timesteps: {model_timesteps.shape[1]}, Actions: {model_actions.shape[1]}")

                if isinstance(model_states, dict):
                    for k, v in model_states.items():
                        if v.shape[1] != seq_len:
                            print(f"[DEBUG] WARNING: state['{k}'] sequence length {v.shape[1]} != {seq_len}")
                            # 调整状态张量的序列长度
                            if v.shape[1] == 1 and seq_len > 1:
                                # 重复序列维度
                                model_states[k] = v.repeat(1, seq_len, 1)
                            elif v.shape[1] > 1 and seq_len == 1:
                                # 只取第一个时间步
                                model_states[k] = v[:, :1, :]

                # 类型验证和调试打印
                print(f"[DEBUG] model_states type: {type(model_states)}")
                if isinstance(model_states, dict):
                    for k, v in model_states.items():
                        print(f"[DEBUG] model_states['{k}'] type: {type(v)}, shape: {v.shape}")
                        assert isinstance(v, torch.Tensor), f"model_states['{k}'] is not a Tensor!"
                else:
                    print(f"[DEBUG] model_states shape: {model_states.shape}")
                    assert isinstance(model_states, torch.Tensor), "model_states is not a Tensor!"

                print(f"[DEBUG] model_actions type: {type(model_actions)}, shape: {model_actions.shape}")
                assert isinstance(model_actions, torch.Tensor), "model_actions is not a Tensor!"

                print(f"[DEBUG] model_rtgs type: {type(model_rtgs)}, shape: {model_rtgs.shape}")
                assert isinstance(model_rtgs, torch.Tensor), "model_rtgs is not a Tensor!"

                print(f"[DEBUG] model_timesteps type: {type(model_timesteps)}, shape: {model_timesteps.shape}")
                assert isinstance(model_timesteps, torch.Tensor), "model_timesteps is not a Tensor!"

                print(f"[DEBUG] attention_mask type: {type(attention_mask)}, shape: {attention_mask.shape}")
                assert isinstance(attention_mask, torch.Tensor), "attention_mask is not a Tensor!"

                # Predict action using the current model with full history
                try:
                    # 获取环境的有效动作空间
                    action_space = env.action_space

                    # 创建正确的动作掩码：只有action_space中的动作是有效的
                    action_mask = torch.zeros(env.n_jobs, dtype=torch.bool)
                    for valid_action in action_space:
                        if 0 <= valid_action < env.n_jobs:
                            action_mask[valid_action] = True
                    model_action_mask = normalize_action_mask_for_n_jobs(
                        action_mask,
                        int(getattr(dt_model, "n_jobs", action_mask.numel())),
                        device=device,
                    )

                    if action_mask.sum() == 0:
                        print(f"[DEBUG] Step {t}: No valid actions in action_space, advancing time")
                        step_result = env.step()
                        if isinstance(step_result, tuple) and len(step_result) == 4:
                            next_state, reward, done, info = step_result
                        elif isinstance(step_result, tuple) and len(step_result) == 3:
                            next_state, done, info = step_result
                            reward = float(info.get("reward", 0.0)) if isinstance(info, dict) else 0.0
                        else:
                            raise ValueError(f"Unexpected env.step return: {step_result}")

                        action_idx = padding_action
                        # 无动作时 RTG 保持不变（时间推进但无奖励）
                        if rtg_mode == "constant":
                            current_rtg = rtgs_history[-1] if rtgs_history else (-float(target_return) if target_return is not None else -1.0)
                        elif rtg_mode == "budget":
                            current_rtg = max(0.0, conditioned_budget - float(getattr(env, "current_time", 0.0)))
                        else:
                            remaining_budget = max(0.0, conditioned_budget - float(getattr(env, "current_time", 0.0)))
                            current_rtg = -remaining_budget / conditioned_budget

                        states_history.append(next_state)
                        actions_history.append(action_idx)
                        rtgs_history.append(current_rtg)
                        timesteps_history.append(t + 1)

                        state = next_state
                        t += 1
                        episode_done = bool(done) if not isinstance(done, dict) else bool(done.get("done", False))
                        if episode_done:
                            job_completed = True
                            break
                        continue

                    # 直接调用forward方法，传递完整历史序列
                    outputs = dt_model(
                        states=model_states,
                        actions=model_actions,
                        returns_to_go=model_rtgs,
                        timesteps=model_timesteps,
                        attention_mask=attention_mask,
                        action_mask=action_mask
                    )

                    # 显式应用有效动作掩码，确保只在合法动作中决策
                    logits = outputs.logits[:, -1, :].to(torch.float32)  # (1, n_jobs)
                    model_action_mask = normalize_action_mask_for_n_jobs(
                        model_action_mask,
                        logits.shape[-1],
                        device=logits.device,
                    )
                    masked_logits = logits.masked_fill(~model_action_mask.unsqueeze(0), -1e9)
                    # 添加详细调试信息
                    print(f"[DEBUG] Step {t}: Action space: {action_space}")
                    print(f"[DEBUG] Step {t}: Action mask created from action_space: {action_mask.tolist()}")
                    print(f"[DEBUG] Step {t}: Raw logits: {logits.detach().cpu().numpy().tolist()}")
                    print(f"[DEBUG] Step {t}: Masked logits max: {masked_logits.max().item()}, argmax: {torch.argmax(masked_logits, dim=-1).item()}")

                    temp = float(rollout_temperature or 0.0)
                    top_k = int(rollout_top_k or 1)
                    if temp > 0.0 and top_k > 1:
                        legal_count = int(model_action_mask.sum().item())
                        k = max(1, min(top_k, legal_count))
                        top_vals, top_idx = torch.topk(masked_logits, k=k, dim=-1)
                        probs = torch.softmax(top_vals / temp, dim=-1)
                        sampled_pos = torch.multinomial(probs, num_samples=1)
                        action_idx = int(top_idx.gather(-1, sampled_pos).item())
                    elif temp > 0.0:
                        probs = torch.softmax(masked_logits / temp, dim=-1)
                        action_idx = int(torch.multinomial(probs, num_samples=1).item())
                    else:
                        # 选择概率最高的合法动作（基于 masked_logits）
                        action_idx = int(torch.argmax(masked_logits, dim=-1).item())

                    # 验证action_idx是否在有效范围内
                    if action_idx < 0 or action_idx >= env.n_jobs:
                        print(f"[DEBUG] Step {t}: Invalid action index {action_idx} (out of bounds), skipping")
                        break

                    # 验证选择的动作是否在有效动作空间内
                    if action_idx not in action_space:
                        print(f"[DEBUG] Step {t}: WARNING - Selected action {action_idx} is not in action_space {action_space}")
                    else:
                        print(f"[DEBUG] Step {t}: Selected action {action_idx} is valid (in action_space {action_space})")

                    print(f"[DEBUG] Step {t}: Predicted action: {action_idx}")
                except Exception as e:
                    print(f"[DEBUG] Action prediction failed: {e}")
                    print(f"[DEBUG] Exception type: {type(e).__name__}")
                    import traceback
                    traceback.print_exc()
                    if torch.cuda.is_available() and isinstance(e, torch.cuda.OutOfMemoryError):
                        torch.cuda.empty_cache()
                    skip_reasons["action_prediction_failed"] += 1
                    break

                # Step environment
                try:
                    if action_idx == -1:
                        # No valid actions available, advance time automatically
                        print(f"[DEBUG] Step {t}: No valid actions, advancing time")
                        step_result = env.step()
                    else:
                        print(f"[DEBUG] Step {t}: Executing action {action_idx}")
                        step_result = env.step(action=action_idx)

                    # Handle step result - JSSPEnv.step returns 4 values
                    if len(step_result) == 4:
                        next_state, reward, done, info = step_result
                    else:
                        # Fallback for legacy environments that return 3 values
                        next_state, done, info = step_result
                        reward = float(info.get("reward", 0.0)) if isinstance(info, dict) else 0.0

                    print(f"[DEBUG] Step {t}: Step successful, done={done}")
                except Exception as e:
                    print(f"[DEBUG] Environment step failed: {e}")
                    skip_reasons["env_step_failed"] += 1
                    break

                if rtg_mode == "constant":
                    current_rtg = rtgs_history[-1] if rtgs_history else (-float(target_return) if target_return is not None else -1.0)
                elif rtg_mode == "budget":
                    current_rtg = max(0.0, conditioned_budget - float(getattr(env, "current_time", 0.0)))
                else:
                    remaining_budget = max(0.0, conditioned_budget - float(getattr(env, "current_time", 0.0)))
                    current_rtg = -remaining_budget / conditioned_budget
                print(f"[DEBUG] Step {t}: Before state update, current_time={env.current_time}")

                # --- 统一更新所有历史缓存 ---
                # 在环境交互之后，下一次循环之前更新所有历史列表
                # 这样所有历史列表的长度始终保持一致
                states_history.append(next_state)  # 添加新状态(dict)
                actions_history.append(action_idx)  # 添加执行的动作（无动作时使用 padding_action 作为占位）
                rtgs_history.append(current_rtg)  # 添加新的RTG
                timesteps_history.append(t + 1)  # 添加新的时间步
                # --- 更新结束 ---

                # 更新状态和时间步
                state = next_state
                t += 1
                print(f"[DEBUG] Step {t}: After state update, new current_time={env.current_time}")
                print(f"[DEBUG] Updated history lengths - states: {len(states_history)}, actions: {len(actions_history)}, rtgs: {len(rtgs_history)}, timesteps: {len(timesteps_history)}")
                # --- 状态更新验证结束 ---

                episode_done = bool(done) if not isinstance(done, dict) else bool(done.get("done", False))
                job_completed = episode_done

            completed_ok = _jssp_all_operations_completed(env)
            if job_completed and completed_ok:
                # 计算相对gap - 使用论文中常见的相对差距公式
                # 使用get_makespan()方法获取正确的makespan，而不是max(env.machine_status)
                model_makespan = env.get_makespan()
                if model_makespan + 1e-8 < lower_bound:
                    print(
                        f"[DEBUG] Invalid rollout: model_makespan {model_makespan} "
                        f"< lower_bound {lower_bound}; job_status={getattr(env, 'job_status', None)}"
                    )
                    skip_reasons["not_completed"] += 1
                    continue
                # relative_gap = (makespan_model - makespan_or) / makespan_or * 100  # 单位：%
                gap = (model_makespan - ortools_makespan) / ortools_makespan * 100
                print(f"[DEBUG] Episode attempt {attempts} completed - OR-Tools makespan: {ortools_makespan}, Model makespan: {model_makespan}, Gap: {gap}%")
                total_gap += gap
                valid_episodes += 1  # 新增：有效实例计数+1
            elif job_completed and not completed_ok:
                print(
                    f"[DEBUG] Episode attempt {attempts} got done=True before all operations completed; "
                    f"job_status={getattr(env, 'job_status', None)}, max_steps={getattr(env, 'max_steps', None)}"
                )
                skip_reasons["not_completed"] += 1
            elif t >= max_steps:
                print(f"[DEBUG] Episode attempt {attempts} terminated due to max_steps limit (t={t}, max_steps={max_steps})")
                print(f"[DEBUG] Final state: {state if isinstance(done, dict) and 'state' in done else 'N/A'}")
                print(f"[DEBUG] done dict: {done}")
                skip_reasons["not_completed"] += 1
                # 可以选择给一个很大的惩罚 makespan
                # model_makespan = float('inf')
                # total_gap += ... (处理无效 episode)
            else:
                print(f"[DEBUG] Episode attempt {attempts} did not complete (t={t}, max_steps={max_steps})")
                print(f"[DEBUG] Final state: {state if isinstance(done, dict) and 'state' in done else 'N/A'}")
                print(f"[DEBUG] done dict: {done}")
                skip_reasons["not_completed"] += 1
                # 任务未完成，继续循环
                pass

    if not quiet_summary:
        builtins.print(f"\n[DEBUG] Rollout evaluation completed - {valid_episodes}/{target_valid_episodes} valid episodes")
        builtins.print(f"[DEBUG] Rollout attempts: {attempts}/{max_attempts}, skipped: {attempts - valid_episodes}")
        builtins.print(f"[DEBUG] Skip reasons: {skip_reasons}")
    avg_gap = total_gap / valid_episodes if valid_episodes > 0 else np.nan  # 新增：使用有效实例数量计算平均gap，无效时返回NaN
    if not quiet_summary:
        builtins.print(f"[DEBUG] Average gap: {avg_gap:.2f}%")
    return avg_gap


def evaluate_rollout_best_of_n(
    dt_model,
    env,
    num_episodes=10,
    target_return=1.0,
    rtg_mode="normalized_budget",
    debug_prints=False,
    rollout_top_k=3,
    rollout_temperature=0.8,
    rollout_best_of_n=8,
    rollout_crop_history=True,
    rollout_eval_max_length=64,
):
    """Evaluate by sampling multiple schedules per same instance and keeping the best gap."""
    import numpy as np
    import builtins
    from Envs.JSSPEnv.JSSPEnv import JSSPEnv, solve_jssp

    best_of_n = max(1, int(rollout_best_of_n or 1))
    target_valid_episodes = int(num_episodes)
    max_attempts = max(1, target_valid_episodes * 10)
    valid_episodes = 0
    attempts = 0
    total_gap = 0.0

    while valid_episodes < target_valid_episodes and attempts < max_attempts:
        attempts += 1
        env.reset()
        base_instance = env.current_instance.copy()
        base_n_jobs = int(env.n_jobs)
        base_n_machines = int(env.n_machines)

        try:
            ortools_result = solve_jssp(base_instance)
            if isinstance(ortools_result, str) and ortools_result == "infeasible":
                attempts += 0
                continue
            if isinstance(ortools_result, tuple):
                base_ortools_makespan = float(ortools_result[0])
            else:
                base_ortools_makespan = float(ortools_result)
        except Exception:
            continue

        trial_gaps = []
        for trial_idx in range(best_of_n):
            trial_env = JSSPEnv(
                n_job_range=[base_n_jobs, base_n_jobs],
                n_mac_range=[base_n_machines, base_n_machines],
                external_instance=base_instance.copy(),
                n_jobs=base_n_jobs,
                n_machines=base_n_machines,
            )
            gap = evaluate_rollout(
                dt_model=dt_model,
                env=trial_env,
                num_episodes=1,
                target_return=target_return,
                rtg_mode=rtg_mode,
                debug_prints=debug_prints,
                rollout_top_k=rollout_top_k,
                rollout_temperature=rollout_temperature,
                rollout_best_of_n=1,
                quiet_summary=True,
                ortools_makespan_override=base_ortools_makespan,
                rollout_crop_history=rollout_crop_history,
                rollout_eval_max_length=rollout_eval_max_length,
            )
            if not np.isnan(gap):
                trial_gaps.append(float(gap))

        if trial_gaps:
            total_gap += min(trial_gaps)
            valid_episodes += 1

    avg_gap = total_gap / valid_episodes if valid_episodes > 0 else np.nan
    builtins.print(
        f"\n[DEBUG] Best-of-{best_of_n} rollout completed - "
        f"{valid_episodes}/{target_valid_episodes} valid episodes, "
        f"attempts={attempts}/{max_attempts}, avg_best_gap={avg_gap:.2f}%"
    )
    return avg_gap


def evaluate_rollout_multi_target(
    dt_model,
    env,
    num_episodes=10,
    target_returns=(0.8, 0.9, 1.0, 1.1),
    rtg_mode="normalized_budget",
    debug_prints=False,
    rollout_top_k=1,
    rollout_temperature=0.0,
    rollout_best_of_n=1,
    rollout_crop_history=True,
    rollout_eval_max_length=64,
):
    """在多个 target_return 上评估 rollout，返回最优（最小）gap。"""
    import numpy as np
    import builtins

    gaps = []
    for tr in target_returns:
        gap = evaluate_rollout(
            dt_model=dt_model,
            env=env,
            num_episodes=num_episodes,
            target_return=float(tr),
            rtg_mode=rtg_mode,
            debug_prints=debug_prints,
            rollout_top_k=rollout_top_k,
            rollout_temperature=rollout_temperature,
            rollout_best_of_n=rollout_best_of_n,
            rollout_crop_history=rollout_crop_history,
            rollout_eval_max_length=rollout_eval_max_length,
        )
        if not np.isnan(gap):
            gaps.append((float(tr), float(gap)))

    if not gaps:
        return np.nan

    best_tr, best_gap = min(gaps, key=lambda x: x[1])
    builtins.print(f"[DEBUG] Multi-target rollout: {gaps}, best_target_return={best_tr}, best_gap={best_gap:.2f}%")
    return best_gap


def evaluate_rollout_panel(
    dt_model,
    panels,
    target_return=1.0,
    rtg_mode="normalized_budget",
    debug_prints=False,
    rollout_top_k=1,
    rollout_temperature=0.0,
    rollout_best_of_n=1,
    rollout_crop_history=True,
    rollout_eval_max_length=64,
    scale_weights=None,
):
    import numpy as np
    from Envs.JSSPEnv.JSSPEnv import JSSPEnv

    scale_weights = scale_weights or {}
    per_scale = {}
    flat_gaps = []
    best_of_n = max(1, int(rollout_best_of_n or 1))

    for scale in sorted((panels or {}).keys()):
        items = list(panels.get(scale, []) or [])
        gaps = []
        for item in items:
            instance = _normalise_instance_array(item.get("current_instance", item.get("instance", None)))
            if instance is None:
                continue
            n_jobs = int(instance.shape[0])
            n_machines = int(instance.shape[1] // 2)
            ref_makespan = _finite_float(item.get("reference_makespan", None), None)
            trial_gaps = []
            for _ in range(best_of_n):
                env = JSSPEnv(
                    n_job_range=[n_jobs, n_jobs],
                    n_mac_range=[n_machines, n_machines],
                    external_instance=instance.copy(),
                    n_jobs=n_jobs,
                    n_machines=n_machines,
                )
                gap = evaluate_rollout(
                    dt_model=dt_model,
                    env=env,
                    num_episodes=1,
                    target_return=target_return,
                    rtg_mode=rtg_mode,
                    debug_prints=debug_prints,
                    rollout_top_k=rollout_top_k,
                    rollout_temperature=rollout_temperature,
                    rollout_best_of_n=1,
                    quiet_summary=True,
                    ortools_makespan_override=ref_makespan,
                    rollout_crop_history=rollout_crop_history,
                    rollout_eval_max_length=rollout_eval_max_length,
                )
                if not np.isnan(gap):
                    trial_gaps.append(float(gap))
            if trial_gaps:
                gap = float(min(trial_gaps))
                gaps.append(gap)
                flat_gaps.append(gap)

        per_scale[scale] = {
            "mean_gap": float(np.mean(gaps)) if gaps else float("nan"),
            "median_gap": float(np.median(gaps)) if gaps else float("nan"),
            "valid_instances": int(len(gaps)),
            "total_instances": int(len(items)),
        }

    valid_scale_items = [
        (scale, stats["mean_gap"])
        for scale, stats in per_scale.items()
        if np.isfinite(stats.get("mean_gap", float("nan")))
    ]
    if not valid_scale_items:
        score = float("nan")
    else:
        weighted_sum = 0.0
        weight_sum = 0.0
        for scale, gap in valid_scale_items:
            weight = float(scale_weights.get(scale, 1.0))
            weighted_sum += weight * float(gap)
            weight_sum += weight
        score = weighted_sum / max(weight_sum, 1e-8)

    return {
        "score": float(score),
        "mean_gap_all_instances": float(np.mean(flat_gaps)) if flat_gaps else float("nan"),
        "target_return": float(target_return),
        "per_scale": per_scale,
    }


def evaluate_rollout_panel_multi_target(
    dt_model,
    panels,
    target_returns=(1.0,),
    rtg_mode="normalized_budget",
    debug_prints=False,
    rollout_top_k=1,
    rollout_temperature=0.0,
    rollout_best_of_n=1,
    rollout_crop_history=True,
    rollout_eval_max_length=64,
    scale_weights=None,
):
    import numpy as np
    import builtins

    summaries = []
    for tr in target_returns:
        summary = evaluate_rollout_panel(
            dt_model=dt_model,
            panels=panels,
            target_return=float(tr),
            rtg_mode=rtg_mode,
            debug_prints=debug_prints,
            rollout_top_k=rollout_top_k,
            rollout_temperature=rollout_temperature,
            rollout_best_of_n=rollout_best_of_n,
            rollout_crop_history=rollout_crop_history,
            rollout_eval_max_length=rollout_eval_max_length,
            scale_weights=scale_weights,
        )
        if np.isfinite(summary.get("score", float("nan"))):
            summaries.append(summary)

    if not summaries:
        return {
            "score": float("nan"),
            "mean_gap_all_instances": float("nan"),
            "target_return": None,
            "per_scale": {},
            "all_targets": [],
        }

    best = min(summaries, key=lambda x: x["score"])
    builtins.print(
        f"[RolloutPanel] best_target_return={best['target_return']}, "
        f"score={best['score']:.2f}%, all_target_scores="
        f"{[(s['target_return'], round(s['score'], 3)) for s in summaries]}"
    )
    for scale, stats in sorted(best.get("per_scale", {}).items()):
        mean_gap = stats.get("mean_gap", float("nan"))
        valid = stats.get("valid_instances", 0)
        total = stats.get("total_instances", 0)
        if np.isfinite(mean_gap):
            builtins.print(f"[RolloutPanel] {scale}: gap={mean_gap:.2f}% ({valid}/{total} valid)")
        else:
            builtins.print(f"[RolloutPanel] {scale}: gap=nan ({valid}/{total} valid)")
    best["all_targets"] = summaries
    return best


def _do_save_plots(dir_out, train_loss_history, eval_loss_history,
                   eval_gap_history, eval_gap_steps_history,
                   global_steps_history, eval_steps_history,
                   rollout_gap_records=None):
    """保存训练loss和gap曲线图（可在早停或正常结束时调用）。"""
    import json, os
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # 保存历史数据到JSON
    loss_data = {
        'train_loss': train_loss_history,
        'eval_loss': eval_loss_history,
        'eval_gap': eval_gap_history,
        'eval_gap_steps': eval_gap_steps_history,
        'global_steps': global_steps_history,
        'eval_steps': eval_steps_history,
    }
    loss_data_path = os.path.join(dir_out, 'loss_history.json')
    with open(loss_data_path, 'w') as f:
        json.dump(loss_data, f)
    print(f"Saved loss and gap history data to {loss_data_path}")

    # 1. Training Loss
    plt.figure(figsize=(12, 6))
    if train_loss_history and global_steps_history:
        plt.plot(global_steps_history[:len(train_loss_history)], train_loss_history,
                 'b-', label='Training Loss')
        plt.title('Decision Transformer Training Loss')
        plt.xlabel('Global Steps')
        plt.ylabel('Training Loss')
        plt.grid(True)
        plt.legend()
        loss_plot_path = os.path.join(dir_out, 'training_loss.png')
        plt.savefig(loss_plot_path, bbox_inches='tight')
        print(f"Saved training loss visualization to {loss_plot_path}")
    plt.close()

    # 2. Evaluation Relative Gap
    plt.figure(figsize=(12, 6))
    if eval_gap_history and eval_gap_steps_history:
        gap_steps = np.array(eval_gap_steps_history)
        gap_values = np.array(eval_gap_history, dtype=float)
        valid_mask = ~np.isnan(gap_values)
        if valid_mask.any():
            plt.plot(gap_steps[valid_mask], gap_values[valid_mask],
                     'r-', alpha=0.7, label='Evaluation Relative Gap (%)')
            plt.scatter(gap_steps[valid_mask], gap_values[valid_mask],
                        c='red', s=100, alpha=0.9, edgecolors='k', zorder=5)
            plt.title('Decision Transformer Evaluation Relative Gap')
            plt.xlabel('Global Steps')
            plt.ylabel('Evaluation Relative Gap (%)')
            plt.grid(True)
            plt.legend()
            gap_plot_path = os.path.join(dir_out, 'evaluation_gap.png')
            plt.savefig(gap_plot_path, bbox_inches='tight')
            print(f"Saved evaluation gap visualization to {gap_plot_path}")
        else:
            print("Warning: No valid gap values to plot")
    plt.close()

    rollout_gap_records = list(rollout_gap_records or [])
    if rollout_gap_records:
        rollout_gap_path = os.path.join(dir_out, 'validation_rollout_gap_history.json')
        with open(rollout_gap_path, 'w') as f:
            json.dump(rollout_gap_records, f)
        print(f"Saved validation rollout gap history to {rollout_gap_path}")

        steps = np.array([int(r.get("step", 0)) for r in rollout_gap_records], dtype=int)
        score_values = np.array([float(r.get("score", np.nan)) for r in rollout_gap_records], dtype=float)
        scale_names = sorted({
            str(scale)
            for r in rollout_gap_records
            for scale in (r.get("per_scale", {}) or {}).keys()
        })

        plt.figure(figsize=(12, 7))
        valid_score = ~np.isnan(score_values)
        if valid_score.any():
            plt.plot(steps[valid_score], score_values[valid_score],
                     marker='o', linewidth=2.2, label='Panel Score')

        for scale in scale_names:
            values = []
            for r in rollout_gap_records:
                stats = (r.get("per_scale", {}) or {}).get(scale, {})
                if isinstance(stats, dict):
                    values.append(float(stats.get("mean_gap", np.nan)))
                else:
                    values.append(np.nan)
            values = np.array(values, dtype=float)
            valid = ~np.isnan(values)
            if valid.any():
                plt.plot(steps[valid], values[valid], marker='.', linewidth=1.4, alpha=0.85, label=scale)

        plt.title('Validation Rollout Gap by Evaluation Step')
        plt.xlabel('Global Steps')
        plt.ylabel('Relative Gap (%)')
        plt.grid(True)
        plt.legend()
        rollout_gap_plot_path = os.path.join(dir_out, 'validation_rollout_gap_by_step.png')
        plt.savefig(rollout_gap_plot_path, bbox_inches='tight')
        plt.close()
        print(f"Saved validation rollout gap visualization to {rollout_gap_plot_path}")


def train_decision_transformer(args):
    """Train a Decision Transformer model for JSSP."""
    # Re-import necessary modules to ensure they're available in this function scope
    import os
    import numpy as np
    set_state_tensorization_mode(args.state_tensorization_mode)
    print(f"State tensorization mode: {get_state_tensorization_mode()}")
    model_max_jobs, model_max_machines = _resolve_model_capacity(args)
    rollout_eval_n_jobs, rollout_eval_n_machines = _resolve_rollout_eval_scale(args)
    rollout_target_returns = _parse_rollout_target_returns(args)
    if getattr(args, "ortools_time_limit", None) is not None:
        os.environ["JSSP_ORTOOLS_TIME_LIMIT"] = str(float(args.ortools_time_limit))
        print(f"[DEBUG] OR-Tools time limit: {float(args.ortools_time_limit):.1f}s")
    # Set output directory
    if args.output_dir is None:
        dir_out = (f"output_dt_jssp_qwen2.5_7b_" 
                   f"r{args.lora_r}_alpha{args.lora_alpha}_" 
                   f"s{args.state_dim}_j{args.n_jobs}_" 
                   f"b{args.per_device_train_batch_size}_" 
                   f"ep{args.num_train_epochs}")
    else:
        dir_out = args.output_dir

    # Create output directory if it doesn't exist
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    # Setup WandB or dummy logger
    wandb = setup_wandb(args, dir_out)

    # Set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load or generate trajectories
    trajectories = load_or_generate_trajectories(args)

    # 如果只需要生成轨迹，生成完成后直接返回
    if args.only_generate_trajectories:
        print("✅ 轨迹生成完成，已跳过训练步骤")
        return None

    if _is_lazy_trajectory_source(trajectories):
        print(f"Detected lazy trajectory source: {trajectories['__lazy_jsonl_path__']}")
        print(f"Keeping args.state_dim={args.state_dim}; states will be parsed per batch")
    elif trajectories and len(trajectories) > 0:
        sample_traj = trajectories[0]
        first_state = sample_traj['states'][0]
        if isinstance(first_state, dict):
            print(f"Detected dict-based states; keeping args.state_dim={args.state_dim}")
        else:
            actual_state_dim = len(first_state)
            print(f"Auto-detected state dimension: {actual_state_dim}")
            args.state_dim = actual_state_dim
    else:
        print("Warning: No trajectories loaded, keeping default state_dim")

    # Create datasets
    train_dataset, eval_dataset, train_trajectories, eval_trajectories, max_actions = create_dt_datasets(trajectories, args)
    rollout_eval_panels = getattr(args, "_rollout_eval_panels", {}) or {}
    rollout_panel_weights = _parse_scale_weights(getattr(args, "rollout_panel_score_weights", ""))
    if rollout_eval_panels:
        panel_summary = ", ".join(f"{scale}:{len(items)}" for scale, items in sorted(rollout_eval_panels.items()))
        print(f"Rollout fixed panel enabled: {panel_summary}")
        if rollout_panel_weights:
            print(f"Rollout panel score weights: {rollout_panel_weights}")
    else:
        print("Rollout fixed panel disabled or empty; falling back to random rollout eval")

    # Create data loaders
    train_loader = create_train_data_loader(train_dataset, args)

    eval_loader = create_data_loader(
        eval_dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        num_workers=0
    )

    # Create Decision Transformer model
    print(f"Creating Decision Transformer model with {args.model_name}...")
    print(f"Model capacity: {model_max_jobs} jobs x {model_max_machines} machines")
    print(f"Rollout evaluation scale: {rollout_eval_n_jobs} jobs x {rollout_eval_n_machines} machines")
    print(f"Rollout target returns: {rollout_target_returns}")
    print(f"Rollout RTG mode: {args.rollout_rtg_mode}")
    print(
        f"Rollout decoding: top_k={args.rollout_top_k}, "
        f"temperature={args.rollout_temperature}, best_of_n={args.rollout_best_of_n}"
    )
    rollout_eval_max_length = int(getattr(args, "rollout_eval_max_length", 64) or 0)
    print(
        f"Rollout history crop: enabled={bool(getattr(args, 'rollout_crop_history', True))}, "
        f"eval_max_length={rollout_eval_max_length if rollout_eval_max_length > 0 else 'model max_length'}"
    )
    dt_model = create_decision_transformer_for_jssp(
        model_name=args.model_name,
        state_dim=args.state_dim,
        n_jobs=model_max_jobs,
        n_machines=model_max_machines,
        max_length=args.max_length,
        max_ep_len=args.max_ep_len,
        use_peft=False if args.init_model_dir else True,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        model_source=args.model_source,
        load_in_4bit=args.load_in_4bit,  # 传递4bit量化参数
        dtype=args.dtype,  # 传递dtype参数
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        state_encoder_type=args.state_encoder_type,
        graph_use_queue_pos=not args.graph_disable_queue_pos,
        use_candidate_action_head=args.use_candidate_action_head,
        use_candidate_harmony_mask=args.use_candidate_harmony_mask,
        candidate_harmony_group_by=args.candidate_harmony_group_by,
        use_scale_adapter_expert=args.use_scale_adapter_expert,
        use_scale_context_adapter=(args.use_scale_context_adapter or args.use_scale_adapter_expert),
        use_scale_candidate_residual=(args.use_scale_candidate_residual or args.use_scale_adapter_expert),
        use_scale_adapter_harmony_mask=args.use_scale_adapter_harmony_mask,
        scale_adapter_group_by=args.scale_adapter_group_by,
        scale_adapter_bottleneck=args.scale_adapter_bottleneck,
        scale_adapter_alpha_init=args.scale_adapter_alpha_init,
        use_scale_embedding=args.use_scale_embedding,
        scale_alpha_init=args.scale_alpha_init,
        scale_embedding_init_std=args.scale_embedding_init_std,
        use_scale_param_mask=args.use_scale_param_mask,
        mask_gnn_encoder=args.mask_gnn_encoder,
        mask_state_projection=args.mask_state_projection,
        mask_lora=args.mask_lora,
        mask_candidate_head=args.mask_candidate_head,
        mask_scale_adapter=args.mask_scale_adapter,
        scale_param_mask_group_by=args.scale_param_mask_group_by,
        input_grad_clip=args.input_grad_clip,
    )
    dt_model.debug_prints = bool(getattr(args, "debug_prints", False))
    if hasattr(dt_model, "state_encoder") and dt_model.state_encoder is not None:
        dt_model.state_encoder.debug_prints = dt_model.debug_prints

    # Move model to GPU if available and not using 4-bit quantization
    # 4-bit models already have device and dtype set correctly during loading
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not args.load_in_4bit:
        dt_model = dt_model.to(device)

    if args.init_model_dir:
        dt_model = load_adapter_checkpoint_into_dt_model(
            dt_model,
            args.init_model_dir,
            custom_modules_path=args.init_custom_modules,
            is_trainable=True,
        )
        if hasattr(dt_model, "register_scale_param_lora_hooks"):
            dt_model.register_scale_param_lora_hooks()

        # ── 迁移学习：分阶段冻结策略 ──────────────────────────────────────────
        # 从单规模(10x5)预训练模型迁移到混合规模训练时：
        # Phase 1（前N/3 epoch）：冻结LLM主干的LoRA层，只训练GNN encoder + 投影头
        #   → 让新GNN编码器快速适配混合规模特征，避免破坏已学到的调度策略
        # Phase 2（后2N/3 epoch）：解冻全部参数，以更小学习率联合微调
        # 通过 args.transfer_freeze_epochs 控制冻结的epoch数，默认=总epoch数的1/3
        transfer_freeze_raw = getattr(args, "transfer_freeze_epochs", None)
        transfer_freeze_epochs = int(transfer_freeze_raw) if transfer_freeze_raw is not None else max(1, args.num_train_epochs // 3)
        if transfer_freeze_epochs > 0:
            print(f"[Transfer Learning] 预训练模型已加载，Phase 1 冻结LLM主干 {transfer_freeze_epochs} 个epoch")
            print(f"[Transfer Learning] Phase 2 将在第 {transfer_freeze_epochs+1} epoch 起解冻全部参数联合微调")

            # 冻结LLM主干（LoRA参数），只训练自定义模块
            for name, param in dt_model.base_model.named_parameters():
                param.requires_grad = False
            print("[Transfer Learning] LLM主干LoRA参数已冻结")

            # 将冻结epoch数记录到args，以便训练循环中判断是否解冻
            args._transfer_freeze_epochs = transfer_freeze_epochs
            args._transfer_phase = 1
        else:
            print("[Transfer Learning] 预训练模型已加载，transfer_freeze_epochs=0，立即联合微调")
            args._transfer_freeze_epochs = 0
            args._transfer_phase = 2
    else:
        args._transfer_freeze_epochs = 0
        args._transfer_phase = 0

    teacher_model = None
    if bool(getattr(args, "distill_old_model_logits", False)):
        if not args.teacher_model_dir:
            raise ValueError("--distill_old_model_logits requires --teacher_model_dir")
        print(f"[Distill] Loading frozen teacher from {args.teacher_model_dir}")
        teacher_max_jobs = int(getattr(args, "teacher_model_max_jobs", 0) or model_max_jobs)
        teacher_default_machines = int(getattr(args, "distill_max_machines", 0) or model_max_machines)
        teacher_max_machines = int(getattr(args, "teacher_model_max_machines", 0) or teacher_default_machines)
        print(f"[Distill] Teacher capacity: {teacher_max_jobs} jobs x {teacher_max_machines} machines")
        teacher_model = create_decision_transformer_for_jssp(
            model_name=args.model_name,
            state_dim=args.state_dim,
            n_jobs=teacher_max_jobs,
            n_machines=teacher_max_machines,
            max_length=args.max_length,
            max_ep_len=args.max_ep_len,
            use_peft=False,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            model_source=args.model_source,
            load_in_4bit=bool(getattr(args, "teacher_load_in_4bit", False)),
            dtype=args.dtype,
            use_gradient_checkpointing=False,
            state_encoder_type=args.state_encoder_type,
            graph_use_queue_pos=not args.graph_disable_queue_pos,
            use_candidate_action_head=args.use_candidate_action_head,
            use_candidate_harmony_mask=False,
            use_scale_adapter_expert=False,
            use_scale_context_adapter=False,
            use_scale_candidate_residual=False,
            use_scale_adapter_harmony_mask=False,
            use_scale_embedding=False,
            input_grad_clip=args.input_grad_clip,
        )
        teacher_model.debug_prints = False
        if hasattr(teacher_model, "state_encoder") and teacher_model.state_encoder is not None:
            teacher_model.state_encoder.debug_prints = False
        if not bool(getattr(args, "teacher_load_in_4bit", False)):
            teacher_model = teacher_model.to(device)
        teacher_model = load_adapter_checkpoint_into_dt_model(
            teacher_model,
            args.teacher_model_dir,
            custom_modules_path=args.teacher_custom_modules,
            is_trainable=False,
        )
        if not bool(getattr(args, "teacher_load_in_4bit", False)):
            teacher_model = teacher_model.to(device)
        else:
            teacher_model = move_dt_custom_modules_to_device(teacher_model, device)
        for p in teacher_model.parameters():
            p.requires_grad = False
        teacher_model.eval()
        print(
            f"[Distill] Enabled old-scale logits distillation: weight={float(args.distill_weight):.3f}, "
            f"T={float(args.distill_temperature):.2f}, max_machines={int(args.distill_max_machines)}"
        )

    custom_trainable_modules = [
        "state_encoder", "rtg_proj", "action_proj", "action_head",
        "candidate_job_proj", "candidate_context_proj", "candidate_fuse_norm", "candidate_action_head", "value_head",
        "scale_context_adapter_norm", "scale_context_adapter_down", "scale_context_adapter_up", "scale_context_adapter_alpha",
        "scale_candidate_residual_norm", "scale_candidate_residual_down", "scale_candidate_residual_up", "scale_candidate_residual_alpha",
        "timestep_embedding", "type_embedding", "scale_embedding", "scale_alpha", "state_proj", "state_proj_norm",
    ]

    def _is_lora_param_name(name):
        return ("lora_A" in name) or ("lora_B" in name) or (".lora_" in name)

    def _set_trainable_parameters(enable_lora):
        for _, param in dt_model.named_parameters():
            param.requires_grad = False

        for module_name in custom_trainable_modules:
            module = getattr(dt_model, module_name, None)
            if module is None:
                continue
            if isinstance(module, torch.nn.Parameter):
                module.requires_grad = True
            elif hasattr(module, "parameters"):
                for p in module.parameters():
                    p.requires_grad = True

        if enable_lora:
            for name, param in dt_model.base_model.named_parameters():
                if _is_lora_param_name(name):
                    param.requires_grad = True

        _apply_graph_encoder_layer_freeze()

    def _freeze_module_params(module):
        if module is None:
            return 0
        count = 0
        for p in module.parameters():
            if p.requires_grad:
                p.requires_grad = False
                count += p.numel()
        return count

    def _apply_graph_encoder_layer_freeze():
        n_freeze = int(getattr(args, "freeze_graph_encoder_layers", 0) or 0)
        encoder = getattr(dt_model, "state_encoder", None)
        if n_freeze <= 0 or encoder is None or not hasattr(encoder, "num_layers"):
            return
        n = max(0, min(n_freeze, int(getattr(encoder, "num_layers", 0) or 0)))
        frozen = 0
        layer_module_names = [
            "pre_msg", "disj_msg", "scheduled_msg",
            "pre_gate_vec", "disj_gate_vec", "scheduled_gate_vec",
            "gru_cells", "norms",
        ]
        for idx in range(n):
            for module_name in layer_module_names:
                modules = getattr(encoder, module_name, None)
                if modules is not None and idx < len(modules):
                    frozen += _freeze_module_params(modules[idx])
        if frozen > 0 and not getattr(args, "_printed_graph_freeze", False):
            print(f"[GraphFreeze] Frozen first {n}/{encoder.num_layers} GraphJSSPStateEncoder message layers ({frozen} params)")
            args._printed_graph_freeze = True

    def _cast_trainable_parameters_to_fp32():
        trainable_cast_count = 0
        for _, param in dt_model.named_parameters():
            if param.requires_grad and param.dtype != torch.float32:
                param.data = param.data.to(dtype=torch.float32)
                trainable_cast_count += 1
        if trainable_cast_count > 0:
            print(f"Casted {trainable_cast_count} trainable parameter tensors to float32 for optimizer stability")

    def _log_trainable_parameters():
        trainable_params = [(n, p.numel()) for n, p in dt_model.named_parameters() if p.requires_grad]
        total_trainable = sum(n for _, n in trainable_params)
        total_params = sum(p.numel() for p in dt_model.parameters())
        print(f"Trainable parameters: {total_trainable} / {total_params} ({100.0 * total_trainable / max(1, total_params):.4f}%)")
        for name, n_params in trainable_params:
            print(f"  [T] {name}: {n_params}")

    amp_enabled = torch.cuda.is_available() and (args.dtype != 'float32')
    amp_dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(torch.cuda.is_available() and args.dtype == 'float16'))

    # Total steps calculation - 考虑梯度累积
    total_steps = (len(train_loader) * args.num_train_epochs) // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)

    def _build_optimizer_and_scheduler(enable_lora):
        custom_params = [p for n, p in dt_model.named_parameters() if p.requires_grad and any(x in n for x in custom_trainable_modules)]
        lora_params = [p for n, p in dt_model.named_parameters() if p.requires_grad and _is_lora_param_name(n)]
        param_groups = []
        if custom_params:
            param_groups.append({
                "params": custom_params,
                "lr": args.learning_rate,
                "weight_decay": args.weight_decay,
            })
        if enable_lora and lora_params:
            param_groups.append({
                "params": lora_params,
                "lr": args.learning_rate * float(getattr(args, "lora_lr_scale", 0.1)),
                "weight_decay": args.weight_decay,
            })

        optim_name = str(getattr(args, "optim", "adamw_8bit") or "adamw_8bit").lower().strip()
        print(
            f"[Optimizer] building optimizer={optim_name}, "
            f"groups={len(param_groups)}, custom_tensors={len(custom_params)}, lora_tensors={len(lora_params)}",
            flush=True,
        )
        if optim_name in {"adamw_8bit", "paged_adamw_8bit", "bnb_adamw_8bit"}:
            try:
                import bitsandbytes as bnb
            except Exception as exc:
                print(f"[Optimizer] bitsandbytes unavailable ({exc}); falling back to torch.optim.AdamW", flush=True)
                optimizer = torch.optim.AdamW(
                    param_groups,
                    betas=(0.9, 0.95),
                    eps=1e-8,
                )
            else:
                optimizer_cls = bnb.optim.PagedAdamW8bit if optim_name == "paged_adamw_8bit" else bnb.optim.AdamW8bit
                optimizer = optimizer_cls(
                    param_groups,
                    betas=(0.9, 0.95),
                    eps=1e-8,
                )
        else:
            optimizer = torch.optim.AdamW(
                param_groups,
                betas=(0.9, 0.95),
                eps=1e-8,
            )
        print(f"[Optimizer] built {optimizer.__class__.__name__}", flush=True)
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=1e-6
        )
        if warmup_steps > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_steps]
            )
        else:
            scheduler = main_scheduler
        return optimizer, scheduler

    use_lora_warmup = bool(hasattr(dt_model.base_model, "peft_config")) and not bool(args.init_model_dir)
    args._freeze_lora_epochs = int(getattr(args, "freeze_lora_epochs", 1) or 0) if use_lora_warmup else 0
    args._lora_phase = 1 if args._freeze_lora_epochs > 0 else 2
    if args._lora_phase == 1:
        print(f"[LoRA Warmup] Phase 1: freeze LoRA for the first {args._freeze_lora_epochs} epoch(s)")
        print(f"[LoRA Warmup] Phase 2 will unfreeze LoRA with lr scale={float(getattr(args, 'lora_lr_scale', 0.1))}")

    enable_lora_now = not (args._lora_phase == 1 or getattr(args, "_transfer_phase", 0) == 1)
    _set_trainable_parameters(enable_lora=enable_lora_now)
    _cast_trainable_parameters_to_fp32()
    _log_trainable_parameters()
    optimizer, scheduler = _build_optimizer_and_scheduler(enable_lora=enable_lora_now)

    candidate_harmony_group_by = str(getattr(args, "candidate_harmony_group_by", "machines") or "machines").lower().strip()
    if candidate_harmony_group_by != "scale":
        candidate_harmony_group_by = "machines"

    def _candidate_harmony_scale_id(n_jobs_value, n_machines_value):
        jobs = int(round(float(n_jobs_value)))
        machines = int(round(float(n_machines_value)))
        if jobs < 0 or machines < 0 or jobs > int(model_max_jobs) or machines > int(model_max_machines):
            return None
        return jobs * (int(model_max_machines) + 1) + machines

    def _candidate_harmony_group_label(group):
        group = int(group)
        if candidate_harmony_group_by == "scale":
            machines_base = int(model_max_machines) + 1
            jobs = group // machines_base
            machines = group % machines_base
            return f"{jobs}x{machines}"
        return f"M{group}"

    def _size_from_sample(sample):
        if not isinstance(sample, dict):
            return None, None
        states = sample.get("states", None)
        if not isinstance(states, dict):
            return None, None

        def _read_scalar(key):
            value = states.get(key, None)
            if value is None:
                return None
            try:
                if torch.is_tensor(value):
                    flat = value.detach().cpu().reshape(-1)
                    flat = flat[flat > 0]
                    if flat.numel() > 0:
                        return int(round(float(flat[0].item())))
            except Exception:
                return None
            try:
                arr = np.asarray(value).reshape(-1)
                arr = arr[arr > 0]
                if arr.size > 0:
                    return int(round(float(arr[0])))
            except Exception:
                return None
            return None

        n_jobs_value = _read_scalar("problem_n_jobs")
        n_machines_value = _read_scalar("problem_n_machines")
        if n_jobs_value is None:
            value = states.get("job_completion_mask", None)
            if value is None:
                value = states.get("job_next_op", None)
            if value is not None:
                try:
                    n_jobs_value = int(value.shape[-1] if torch.is_tensor(value) else np.asarray(value).shape[-1])
                except Exception:
                    pass
        if n_machines_value is None:
            for key in ("normalized_machine_availability", "machine_busy_mask", "machine_status"):
                value = states.get(key, None)
                if value is None:
                    continue
                try:
                    n_machines_value = int(value.shape[-1] if torch.is_tensor(value) else np.asarray(value).shape[-1])
                    break
                except Exception:
                    continue
        return n_jobs_value, n_machines_value

    def _candidate_harmony_group_from_sample(sample):
        n_jobs_value, n_machines_value = _size_from_sample(sample)
        if n_machines_value is None:
            return None
        if candidate_harmony_group_by == "scale":
            if n_jobs_value is None:
                return None
            return _candidate_harmony_scale_id(n_jobs_value, n_machines_value)
        machines = int(round(float(n_machines_value)))
        if 0 <= machines <= int(model_max_machines):
            return machines
        return None

    def _collect_candidate_harmony_groups_from_dataset():
        groups = []
        seen = set()
        if len(train_dataset) <= 0:
            return groups
        for idx in range(len(train_dataset)):
            try:
                group = _candidate_harmony_group_from_sample(train_dataset[idx])
            except Exception:
                continue
            if group is not None and group not in seen:
                seen.add(group)
                groups.append(int(group))
        return sorted(groups)

    def _parse_candidate_harmony_groups():
        raw = str(getattr(args, "candidate_harmony_groups", "") or "").strip().lower()
        if raw in {"", "auto"}:
            return _collect_candidate_harmony_groups_from_dataset()
        groups = []
        for part in raw.split(","):
            part = part.strip().lower()
            if not part:
                continue
            val = None
            if candidate_harmony_group_by == "scale":
                if "x" in part:
                    left, right = part.split("x", 1)
                    try:
                        val = _candidate_harmony_scale_id(int(round(float(left))), int(round(float(right))))
                    except Exception:
                        val = None
                else:
                    try:
                        val = int(round(float(part)))
                    except Exception:
                        val = None
                    max_group = (int(model_max_jobs) + 1) * (int(model_max_machines) + 1)
                    if val is not None and not (0 <= val < max_group):
                        val = None
            else:
                if "x" in part:
                    part = part.split("x", 1)[1]
                try:
                    val = int(round(float(part)))
                except Exception:
                    val = None
                if val is not None and not (0 <= val <= int(model_max_machines)):
                    val = None
            if val is not None and val not in groups:
                groups.append(int(val))
        return groups

    harmony_rng = np.random.default_rng(int(getattr(args, "seed", 42)) + 9109)
    candidate_harmony_groups = (
        _parse_candidate_harmony_groups()
        if bool(getattr(args, "use_candidate_harmony_mask", False))
        else []
    )

    def _sample_candidate_harmony_batch(group):
        batch_size_probe = max(1, int(getattr(args, "candidate_harmony_probe_batch_size", 1) or 1))
        max_attempts = max(batch_size_probe, int(getattr(args, "candidate_harmony_probe_max_attempts", 200) or 200))
        samples = []
        if len(train_dataset) <= 0:
            return None
        for _ in range(max_attempts):
            idx = int(harmony_rng.integers(0, len(train_dataset)))
            try:
                sample = train_dataset[idx]
            except Exception:
                continue
            if _candidate_harmony_group_from_sample(sample) == int(group):
                samples.append(sample)
                if len(samples) >= batch_size_probe:
                    break
        if not samples:
            return None
        return collate_fn(samples)

    def _candidate_harmony_probe_gradient(batch, group):
        if batch is None or getattr(dt_model, "candidate_harmony_mask", None) is None:
            return None
        mask_table = dt_model.candidate_harmony_mask
        if not torch.is_tensor(mask_table) or int(group) >= int(mask_table.shape[0]):
            return None

        was_training = dt_model.training
        dt_model.train()
        optimizer.zero_grad(set_to_none=True)
        if mask_table.grad is not None:
            mask_table.grad = None
        mask_table.requires_grad_(True)

        states = batch["states"]
        if not isinstance(states, dict):
            states = states.to(device)
        actions = batch["actions"].to(device)
        expert_actions = batch.get("expert_actions", None)
        if expert_actions is not None:
            expert_actions = expert_actions.to(device)
        returns_to_go = batch["returns_to_go"].to(device)
        timesteps = batch["timesteps"].to(device)
        batch_mask = batch["mask"].to(device)

        model_input_actions = actions.clone()
        model_input_actions[model_input_actions == -100] = dt_model.n_jobs
        model_actions = build_prev_action_inputs(model_input_actions, dt_model.n_jobs)
        target_actions = expert_actions if expert_actions is not None else actions
        action_mask = None
        if bool(getattr(args, "use_train_action_mask", False)):
            action_mask = build_graph_action_mask(states, dt_model.n_jobs)
            action_mask = include_target_actions_in_mask(action_mask, target_actions, dt_model.n_jobs)
        ce_mask = build_choice_weighted_loss_mask(
            batch_mask,
            action_mask,
            single_action_weight=getattr(args, "single_action_loss_weight", 1.0),
        )

        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            outputs = dt_model(
                states,
                returns_to_go,
                timesteps,
                model_actions,
                attention_mask=batch_mask,
                action_mask=action_mask,
            )
            logits = torch.clamp(outputs.logits, min=-1e4, max=1e4)
            value_preds = getattr(outputs, "value_preds", None)
            if logits.shape[1] != target_actions.shape[1]:
                logits = logits[:, :target_actions.shape[1], :]
                ce_mask = ce_mask[:, :target_actions.shape[1]]
                batch_mask = batch_mask[:, :target_actions.shape[1]]
                if action_mask is not None and action_mask.shape[1] > target_actions.shape[1]:
                    action_mask = action_mask[:, :target_actions.shape[1], :]
            value_targets = None
            if bool(getattr(args, "use_value_aux_loss", False)) and value_preds is not None:
                if returns_to_go.dim() == 3:
                    value_targets = returns_to_go[:, :target_actions.shape[1], 0].to(logits.device, dtype=value_preds.dtype)
                elif returns_to_go.dim() == 2:
                    value_targets = returns_to_go[:, :target_actions.shape[1]].to(logits.device, dtype=value_preds.dtype)
            loss = compute_total_loss(
                logits,
                target_actions.to(logits.device),
                ce_mask.to(logits.device),
                unlikelihood_weight=args.unlikelihood_weight,
                num_negatives=args.unlikelihood_negatives,
                action_valid_mask=action_mask,
                label_smoothing=args.label_smoothing,
                value_preds=value_preds,
                value_targets=value_targets,
                value_loss_weight=args.value_loss_weight if bool(getattr(args, "use_value_aux_loss", False)) else 0.0,
            )

        if torch.isfinite(loss):
            loss.backward()
        grad = None
        if mask_table.grad is not None:
            grad = mask_table.grad[int(group)].detach().to(device="cpu", dtype=torch.float32).clone()
        mask_table.requires_grad_(False)
        mask_table.grad = None
        optimizer.zero_grad(set_to_none=True)
        if not was_training:
            dt_model.eval()
        return grad

    @torch.no_grad()
    def _candidate_harmony_probe_flip_stats(batch, group):
        if batch is None or getattr(dt_model, "candidate_harmony_mask", None) is None:
            return None
        mask_table = dt_model.candidate_harmony_mask
        if not torch.is_tensor(mask_table) or int(group) >= int(mask_table.shape[0]):
            return None

        def _forward_logits():
            outputs = dt_model(
                states,
                returns_to_go,
                timesteps,
                model_actions,
                attention_mask=batch_mask,
                action_mask=action_mask,
            )
            logits = torch.clamp(outputs.logits, min=-1e4, max=1e4)
            if logits.shape[1] != valid_positions.shape[1]:
                logits = logits[:, :valid_positions.shape[1], :]
            return logits

        was_training = dt_model.training
        dt_model.eval()

        states = batch["states"]
        if not isinstance(states, dict):
            states = states.to(device)
        actions = batch["actions"].to(device)
        expert_actions = batch.get("expert_actions", None)
        if expert_actions is not None:
            expert_actions = expert_actions.to(device)
        returns_to_go = batch["returns_to_go"].to(device)
        timesteps = batch["timesteps"].to(device)
        batch_mask = batch["mask"].to(device)

        model_input_actions = actions.clone()
        model_input_actions[model_input_actions == -100] = dt_model.n_jobs
        model_actions = build_prev_action_inputs(model_input_actions, dt_model.n_jobs)
        target_actions = expert_actions if expert_actions is not None else actions
        action_mask = None
        if bool(getattr(args, "use_train_action_mask", False)):
            action_mask = build_graph_action_mask(states, dt_model.n_jobs)
            action_mask = include_target_actions_in_mask(action_mask, target_actions, dt_model.n_jobs)
        valid_positions = (batch_mask > 0)
        if target_actions is not None:
            valid_positions = valid_positions & (target_actions.to(device) >= 0)

        row = mask_table[int(group)]
        saved_row = row.detach().clone()
        try:
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                row.fill_(1.0)
                base_logits = _forward_logits()
                row.copy_(saved_row)
                masked_logits = _forward_logits()
        finally:
            row.copy_(saved_row)
            if was_training:
                dt_model.train()
            else:
                dt_model.eval()

        valid_positions = valid_positions[:, :base_logits.shape[1]].to(device=base_logits.device)
        valid_count = int(valid_positions.sum().item())
        if valid_count <= 0:
            return None

        base_top1 = base_logits.argmax(dim=-1)
        masked_top1 = masked_logits.argmax(dim=-1)
        flip_rate = ((base_top1 != masked_top1) & valid_positions).to(dtype=torch.float32).sum() / float(valid_count)

        base_top2 = torch.topk(base_logits, k=min(2, base_logits.shape[-1]), dim=-1).values
        masked_top2 = torch.topk(masked_logits, k=min(2, masked_logits.shape[-1]), dim=-1).values
        if base_top2.shape[-1] < 2 or masked_top2.shape[-1] < 2:
            return {"flip": float(flip_rate.item()), "dmargin": float("nan")}
        base_margin = (base_top2[..., 0] - base_top2[..., 1])[valid_positions].to(dtype=torch.float32).mean()
        masked_margin = (masked_top2[..., 0] - masked_top2[..., 1])[valid_positions].to(dtype=torch.float32).mean()
        return {
            "flip": float(flip_rate.item()),
            "dmargin": float((masked_margin - base_margin).item()),
        }

    @torch.no_grad()
    def _apply_candidate_harmony_update(group_gradients, global_step_value, group_batches=None):
        mask_table = getattr(dt_model, "candidate_harmony_mask", None)
        if not group_gradients or mask_table is None or not torch.is_tensor(mask_table):
            return
        available = {int(k): v.to(device=mask_table.device, dtype=torch.float32) for k, v in group_gradients.items() if v is not None}
        if not available:
            return
        avg_grad = torch.stack(list(available.values()), dim=0).mean(dim=0)
        width = int(mask_table.shape[1])
        target_zeros = int(round(width * max(0.0, min(0.95, float(getattr(args, "candidate_harmony_sparsity", 0.1) or 0.0)))))
        base_change = int(round(width * max(0.0, float(getattr(args, "candidate_harmony_change_frac", 0.01) or 0.0))))
        change_num = max(1, base_change) if target_zeros > 0 else 0
        gamma = float(getattr(args, "candidate_harmony_gamma", 10.0) or 0.0)
        summaries = []
        for group, grad in sorted(available.items()):
            if group >= int(mask_table.shape[0]) or change_num <= 0:
                continue
            row = mask_table[group]
            active = row > 0.5
            inactive = ~active
            current_zeros = int(inactive.sum().item())
            harmony = grad.to(device=row.device, dtype=torch.float32) * avg_grad.to(device=row.device, dtype=torch.float32)
            importance = grad.to(device=row.device, dtype=torch.float32).pow(2)
            close_n = 0
            open_n = 0
            if current_zeros < target_zeros:
                close_n = min(change_num, target_zeros - current_zeros, int(active.sum().item()))
            elif current_zeros > target_zeros:
                open_n = min(change_num, current_zeros - target_zeros, int(inactive.sum().item()))
            else:
                close_n = min(change_num, int(active.sum().item()))
                open_n = min(close_n, int(inactive.sum().item()))
                close_n = open_n

            if close_n > 0:
                close_score = gamma * harmony + importance
                close_score = close_score.masked_fill(~active, float("inf"))
                close_idx = torch.topk(close_score, k=close_n, largest=False).indices
                row[close_idx] = 0.0
            if open_n > 0:
                open_score = harmony.masked_fill(~inactive, float("-inf"))
                open_idx = torch.topk(open_score, k=open_n, largest=True).indices
                row[open_idx] = 1.0
            density = float((row > 0.5).to(dtype=torch.float32).mean().item())
            summary = f"{_candidate_harmony_group_label(group)}:density={density:.3f},close={close_n},open={open_n}"
            if bool(getattr(args, "candidate_harmony_log_flip", False)) and group_batches is not None:
                flip_stats = _candidate_harmony_probe_flip_stats(group_batches.get(group), group)
                if flip_stats is not None:
                    summary += (
                        f",flip={100.0 * float(flip_stats.get('flip', 0.0)):.2f}%"
                        f",dmargin={float(flip_stats.get('dmargin', float('nan'))):.4f}"
                    )
            summaries.append(summary)
        if summaries:
            overlap_summaries = []
            valid_groups = [g for g in sorted(available.keys()) if g < int(mask_table.shape[0])]
            for left_idx, left_group in enumerate(valid_groups):
                left_closed = mask_table[left_group] <= 0.5
                for right_group in valid_groups[left_idx + 1:]:
                    right_closed = mask_table[right_group] <= 0.5
                    union = (left_closed | right_closed).sum()
                    if int(union.item()) == 0:
                        jaccard = 1.0
                    else:
                        intersection = (left_closed & right_closed).sum()
                        jaccard = float((intersection.to(dtype=torch.float32) / union.to(dtype=torch.float32)).item())
                    overlap_summaries.append(
                        f"{_candidate_harmony_group_label(left_group)}-{_candidate_harmony_group_label(right_group)}:{jaccard:.3f}"
                    )
            msg = f"[CandidateHarmonyMask] step={global_step_value} " + "; ".join(summaries)
            if overlap_summaries:
                msg += " | closed_jaccard=" + ", ".join(overlap_summaries)
            print(msg)

    def _maybe_update_candidate_harmony_mask(global_step_value):
        interval = int(getattr(args, "candidate_harmony_update_interval", 0) or 0)
        if interval <= 0 or global_step_value <= 0 or global_step_value % interval != 0:
            return
        if not bool(getattr(args, "use_candidate_harmony_mask", False)):
            return
        if getattr(dt_model, "candidate_harmony_mask", None) is None:
            return
        group_gradients = {}
        group_batches = {}
        for group in candidate_harmony_groups:
            batch = _sample_candidate_harmony_batch(group)
            if batch is None:
                print(
                    f"[CandidateHarmonyMask] step={global_step_value} "
                    f"skip {_candidate_harmony_group_label(group)}: no probe batch found"
                )
                continue
            grad = _candidate_harmony_probe_gradient(batch, group)
            if grad is not None and torch.isfinite(grad).all():
                group_gradients[group] = grad
                if bool(getattr(args, "candidate_harmony_log_flip", False)):
                    group_batches[group] = batch
        _apply_candidate_harmony_update(group_gradients, global_step_value, group_batches=group_batches)

    if bool(getattr(args, "use_candidate_harmony_mask", False)):
        print(
            "[CandidateHarmonyMask] enabled: "
            f"group_by={candidate_harmony_group_by}, "
            f"groups={[ _candidate_harmony_group_label(g) for g in candidate_harmony_groups ]}, "
            f"update_interval={int(getattr(args, 'candidate_harmony_update_interval', 0) or 0)}, "
            f"sparsity={float(getattr(args, 'candidate_harmony_sparsity', 0.1) or 0.0):.3f}"
        )

    scale_adapter_group_by = str(getattr(args, "scale_adapter_group_by", "scale") or "scale").lower().strip()
    if scale_adapter_group_by != "machines":
        scale_adapter_group_by = "scale"

    def _scale_adapter_group_label(group):
        group = int(group)
        if scale_adapter_group_by == "scale":
            machines_base = int(model_max_machines) + 1
            jobs = group // machines_base
            machines = group % machines_base
            return f"{jobs}x{machines}"
        return f"M{group}"

    def _scale_adapter_group_from_sample(sample):
        n_jobs_value, n_machines_value = _size_from_sample(sample)
        if n_machines_value is None:
            return None
        if scale_adapter_group_by == "scale":
            if n_jobs_value is None:
                return None
            return _candidate_harmony_scale_id(n_jobs_value, n_machines_value)
        machines = int(round(float(n_machines_value)))
        if 0 <= machines <= int(model_max_machines):
            return machines
        return None

    def _collect_scale_adapter_groups_from_dataset():
        groups = []
        seen = set()
        if len(train_dataset) <= 0:
            return groups
        for idx in range(len(train_dataset)):
            try:
                group = _scale_adapter_group_from_sample(train_dataset[idx])
            except Exception:
                continue
            if group is not None and group not in seen:
                seen.add(group)
                groups.append(int(group))
        return sorted(groups)

    def _parse_scale_adapter_groups():
        raw = str(getattr(args, "scale_adapter_groups", "") or "").strip().lower()
        if raw in {"", "auto"}:
            return _collect_scale_adapter_groups_from_dataset()
        groups = []
        for part in raw.split(","):
            part = part.strip().lower()
            if not part:
                continue
            val = None
            if scale_adapter_group_by == "scale":
                if "x" in part:
                    left, right = part.split("x", 1)
                    try:
                        val = _candidate_harmony_scale_id(int(round(float(left))), int(round(float(right))))
                    except Exception:
                        val = None
                else:
                    try:
                        val = int(round(float(part)))
                    except Exception:
                        val = None
                max_group = (int(model_max_jobs) + 1) * (int(model_max_machines) + 1)
                if val is not None and not (0 <= val < max_group):
                    val = None
            else:
                if "x" in part:
                    part = part.split("x", 1)[1]
                try:
                    val = int(round(float(part)))
                except Exception:
                    val = None
                if val is not None and not (0 <= val <= int(model_max_machines)):
                    val = None
            if val is not None and val not in groups:
                groups.append(int(val))
        return groups

    scale_adapter_rng = np.random.default_rng(int(getattr(args, "seed", 42)) + 12031)
    scale_adapter_groups = (
        _parse_scale_adapter_groups()
        if bool(getattr(args, "use_scale_adapter_harmony_mask", False))
        else []
    )

    def _sample_scale_adapter_batch(group):
        batch_size_probe = max(1, int(getattr(args, "scale_adapter_harmony_probe_batch_size", 1) or 1))
        max_attempts = max(batch_size_probe, int(getattr(args, "scale_adapter_harmony_probe_max_attempts", 300) or 300))
        samples = []
        if len(train_dataset) <= 0:
            return None
        for _ in range(max_attempts):
            idx = int(scale_adapter_rng.integers(0, len(train_dataset)))
            try:
                sample = train_dataset[idx]
            except Exception:
                continue
            if _scale_adapter_group_from_sample(sample) == int(group):
                samples.append(sample)
                if len(samples) >= batch_size_probe:
                    break
        if not samples:
            return None
        return collate_fn(samples)

    def _scale_adapter_mask_tables():
        tables = {}
        context_mask = getattr(dt_model, "scale_context_harmony_mask", None)
        candidate_mask = getattr(dt_model, "scale_candidate_harmony_mask", None)
        if torch.is_tensor(context_mask):
            tables["context"] = context_mask
        if torch.is_tensor(candidate_mask):
            tables["candidate"] = candidate_mask
        return tables

    def _scale_adapter_harmony_probe_gradients(batch, group):
        tables = _scale_adapter_mask_tables()
        if batch is None or not tables:
            return None
        for table in tables.values():
            if int(group) >= int(table.shape[0]):
                return None

        was_training = dt_model.training
        dt_model.train()
        optimizer.zero_grad(set_to_none=True)
        for table in tables.values():
            if table.grad is not None:
                table.grad = None
            table.requires_grad_(True)

        states = batch["states"]
        if not isinstance(states, dict):
            states = states.to(device)
        actions = batch["actions"].to(device)
        expert_actions = batch.get("expert_actions", None)
        if expert_actions is not None:
            expert_actions = expert_actions.to(device)
        returns_to_go = batch["returns_to_go"].to(device)
        timesteps = batch["timesteps"].to(device)
        batch_mask = batch["mask"].to(device)

        model_input_actions = actions.clone()
        model_input_actions[model_input_actions == -100] = dt_model.n_jobs
        model_actions = build_prev_action_inputs(model_input_actions, dt_model.n_jobs)
        target_actions = expert_actions if expert_actions is not None else actions
        action_mask = None
        if bool(getattr(args, "use_train_action_mask", False)):
            action_mask = build_graph_action_mask(states, dt_model.n_jobs)
            action_mask = include_target_actions_in_mask(action_mask, target_actions, dt_model.n_jobs)
        ce_mask = build_choice_weighted_loss_mask(
            batch_mask,
            action_mask,
            single_action_weight=getattr(args, "single_action_loss_weight", 1.0),
        )

        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            outputs = dt_model(
                states,
                returns_to_go,
                timesteps,
                model_actions,
                attention_mask=batch_mask,
                action_mask=action_mask,
            )
            logits = torch.clamp(outputs.logits, min=-1e4, max=1e4)
            value_preds = getattr(outputs, "value_preds", None)
            if logits.shape[1] != target_actions.shape[1]:
                logits = logits[:, :target_actions.shape[1], :]
                ce_mask = ce_mask[:, :target_actions.shape[1]]
                batch_mask = batch_mask[:, :target_actions.shape[1]]
                if action_mask is not None and action_mask.shape[1] > target_actions.shape[1]:
                    action_mask = action_mask[:, :target_actions.shape[1], :]
            value_targets = None
            if bool(getattr(args, "use_value_aux_loss", False)) and value_preds is not None:
                if returns_to_go.dim() == 3:
                    value_targets = returns_to_go[:, :target_actions.shape[1], 0].to(logits.device, dtype=value_preds.dtype)
                elif returns_to_go.dim() == 2:
                    value_targets = returns_to_go[:, :target_actions.shape[1]].to(logits.device, dtype=value_preds.dtype)
            loss = compute_total_loss(
                logits,
                target_actions.to(logits.device),
                ce_mask.to(logits.device),
                unlikelihood_weight=args.unlikelihood_weight,
                num_negatives=args.unlikelihood_negatives,
                action_valid_mask=action_mask,
                label_smoothing=args.label_smoothing,
                value_preds=value_preds,
                value_targets=value_targets,
                value_loss_weight=args.value_loss_weight if bool(getattr(args, "use_value_aux_loss", False)) else 0.0,
            )

        if torch.isfinite(loss):
            loss.backward()
        gradients = {}
        for name, table in tables.items():
            if table.grad is not None:
                gradients[name] = table.grad[int(group)].detach().to(device="cpu", dtype=torch.float32).clone()
            table.requires_grad_(False)
            table.grad = None
        optimizer.zero_grad(set_to_none=True)
        if not was_training:
            dt_model.eval()
        return gradients

    @torch.no_grad()
    def _apply_single_scale_adapter_harmony_update(mask_name, mask_table, group_gradients, global_step_value):
        available = {
            int(group): grads[mask_name].to(device=mask_table.device, dtype=torch.float32)
            for group, grads in group_gradients.items()
            if grads is not None and mask_name in grads and grads[mask_name] is not None
        }
        if not available:
            return
        avg_grad = torch.stack(list(available.values()), dim=0).mean(dim=0)
        width = int(mask_table.shape[1])
        target_zeros = int(round(width * max(0.0, min(0.95, float(getattr(args, "scale_adapter_harmony_sparsity", 0.05) or 0.0)))))
        base_change = int(round(width * max(0.0, float(getattr(args, "scale_adapter_harmony_change_frac", 0.005) or 0.0))))
        change_num = max(1, base_change) if target_zeros > 0 else 0
        gamma = float(getattr(args, "scale_adapter_harmony_gamma", 10.0) or 0.0)
        summaries = []
        for group, grad in sorted(available.items()):
            if group >= int(mask_table.shape[0]) or change_num <= 0:
                continue
            row = mask_table[group]
            active = row > 0.5
            inactive = ~active
            current_zeros = int(inactive.sum().item())
            harmony = grad.to(device=row.device, dtype=torch.float32) * avg_grad.to(device=row.device, dtype=torch.float32)
            importance = grad.to(device=row.device, dtype=torch.float32).pow(2)
            close_n = 0
            open_n = 0
            if current_zeros < target_zeros:
                close_n = min(change_num, target_zeros - current_zeros, int(active.sum().item()))
            elif current_zeros > target_zeros:
                open_n = min(change_num, current_zeros - target_zeros, int(inactive.sum().item()))
            else:
                close_n = min(change_num, int(active.sum().item()))
                open_n = min(close_n, int(inactive.sum().item()))
                close_n = open_n

            if close_n > 0:
                close_score = gamma * harmony + importance
                close_score = close_score.masked_fill(~active, float("inf"))
                close_idx = torch.topk(close_score, k=close_n, largest=False).indices
                row[close_idx] = 0.0
            if open_n > 0:
                open_score = harmony.masked_fill(~inactive, float("-inf"))
                open_idx = torch.topk(open_score, k=open_n, largest=True).indices
                row[open_idx] = 1.0
            density = float((row > 0.5).to(dtype=torch.float32).mean().item())
            summaries.append(f"{_scale_adapter_group_label(group)}:density={density:.3f},close={close_n},open={open_n}")
        if not summaries:
            return
        overlap_summaries = []
        valid_groups = [g for g in sorted(available.keys()) if g < int(mask_table.shape[0])]
        for left_idx, left_group in enumerate(valid_groups):
            left_closed = mask_table[left_group] <= 0.5
            for right_group in valid_groups[left_idx + 1:]:
                right_closed = mask_table[right_group] <= 0.5
                union = (left_closed | right_closed).sum()
                if int(union.item()) == 0:
                    jaccard = 1.0
                else:
                    intersection = (left_closed & right_closed).sum()
                    jaccard = float((intersection.to(dtype=torch.float32) / union.to(dtype=torch.float32)).item())
                overlap_summaries.append(f"{_scale_adapter_group_label(left_group)}-{_scale_adapter_group_label(right_group)}:{jaccard:.3f}")
        msg = f"[ScaleAdapterHarmonyMask:{mask_name}] step={global_step_value} " + "; ".join(summaries)
        if overlap_summaries:
            msg += " | closed_jaccard=" + ", ".join(overlap_summaries)
        print(msg)

    def _maybe_update_scale_adapter_harmony_mask(global_step_value):
        interval = int(getattr(args, "scale_adapter_harmony_update_interval", 0) or 0)
        if interval <= 0 or global_step_value <= 0 or global_step_value % interval != 0:
            return
        if not bool(getattr(args, "use_scale_adapter_harmony_mask", False)):
            return
        tables = _scale_adapter_mask_tables()
        if not tables:
            return
        group_gradients = {}
        for group in scale_adapter_groups:
            batch = _sample_scale_adapter_batch(group)
            if batch is None:
                print(
                    f"[ScaleAdapterHarmonyMask] step={global_step_value} "
                    f"skip {_scale_adapter_group_label(group)}: no probe batch found"
                )
                continue
            grads = _scale_adapter_harmony_probe_gradients(batch, group)
            if grads and all(torch.isfinite(g).all() for g in grads.values()):
                group_gradients[group] = grads
        for mask_name, mask_table in tables.items():
            _apply_single_scale_adapter_harmony_update(mask_name, mask_table, group_gradients, global_step_value)

    if bool(getattr(args, "use_scale_adapter_harmony_mask", False)):
        print(
            "[ScaleAdapterHarmonyMask] enabled: "
            f"group_by={scale_adapter_group_by}, "
            f"groups={[ _scale_adapter_group_label(g) for g in scale_adapter_groups ]}, "
            f"update_interval={int(getattr(args, 'scale_adapter_harmony_update_interval', 0) or 0)}, "
            f"sparsity={float(getattr(args, 'scale_adapter_harmony_sparsity', 0.05) or 0.0):.3f}, "
            f"targets={list(_scale_adapter_mask_tables().keys())}"
        )

    scale_param_mask_group_by = str(getattr(args, "scale_param_mask_group_by", "scale") or "scale").lower().strip()
    if scale_param_mask_group_by != "machines":
        scale_param_mask_group_by = "scale"

    def _scale_param_mask_group_label(group):
        group = int(group)
        if scale_param_mask_group_by == "scale":
            machines_base = int(model_max_machines) + 1
            jobs = group // machines_base
            machines = group % machines_base
            return f"{jobs}x{machines}"
        return f"M{group}"

    def _scale_param_mask_group_from_sample(sample):
        n_jobs_value, n_machines_value = _size_from_sample(sample)
        if n_machines_value is None:
            return None
        if scale_param_mask_group_by == "scale":
            if n_jobs_value is None:
                return None
            return _candidate_harmony_scale_id(n_jobs_value, n_machines_value)
        machines = int(round(float(n_machines_value)))
        if 0 <= machines <= int(model_max_machines):
            return machines
        return None

    def _collect_scale_param_mask_groups_from_dataset():
        groups = []
        seen = set()
        if len(train_dataset) <= 0:
            return groups
        for idx in range(len(train_dataset)):
            try:
                group = _scale_param_mask_group_from_sample(train_dataset[idx])
            except Exception:
                continue
            if group is not None and group not in seen:
                seen.add(group)
                groups.append(int(group))
        return sorted(groups)

    def _parse_scale_param_mask_groups():
        raw = str(getattr(args, "scale_param_mask_groups", "") or "").strip().lower()
        if raw in {"", "auto"}:
            return _collect_scale_param_mask_groups_from_dataset()
        groups = []
        for part in raw.split(","):
            part = part.strip().lower()
            if not part:
                continue
            val = None
            if scale_param_mask_group_by == "scale":
                if "x" in part:
                    left, right = part.split("x", 1)
                    try:
                        val = _candidate_harmony_scale_id(int(round(float(left))), int(round(float(right))))
                    except Exception:
                        val = None
                else:
                    try:
                        val = int(round(float(part)))
                    except Exception:
                        val = None
                max_group = (int(model_max_jobs) + 1) * (int(model_max_machines) + 1)
                if val is not None and not (0 <= val < max_group):
                    val = None
            else:
                if "x" in part:
                    part = part.split("x", 1)[1]
                try:
                    val = int(round(float(part)))
                except Exception:
                    val = None
                if val is not None and not (0 <= val <= int(model_max_machines)):
                    val = None
            if val is not None and val not in groups:
                groups.append(int(val))
        return groups

    scale_param_mask_rng = np.random.default_rng(int(getattr(args, "seed", 42)) + 15017)
    scale_param_mask_groups = _parse_scale_param_mask_groups()
    scale_param_mask_conflict_mode = str(
        getattr(args, "scale_param_mask_conflict_mode", "legacy_global_mean") or "legacy_global_mean"
    ).strip().lower()
    ssnc_controller = None
    ssnc_diagnostics_path = None
    if scale_param_mask_conflict_mode == "ssnc_neighbor":
        if scale_param_mask_group_by != "scale":
            raise ValueError("SSNC requires --scale_param_mask_group_by scale so (n_jobs,n_machines) can be decoded.")
        ssnc_controller = SSNCMaskController(
            neighbor_k=max(1, int(getattr(args, "ssnc_neighbor_k", 3) or 3)),
            structure_ema_decay=float(getattr(args, "ssnc_structure_ema_decay", 0.90)),
            grad_norm_ema_decay=float(getattr(args, "ssnc_grad_norm_ema_decay", 0.90)),
            fisher_ema_decay=float(getattr(args, "ssnc_fisher_ema_decay", 0.90)),
            score_ema_decay=float(getattr(args, "ssnc_score_ema_decay", 0.90)),
        )
        loaded_ssnc_state = getattr(dt_model, "_loaded_ssnc_mask_state", None)
        if loaded_ssnc_state is not None:
            ssnc_controller.load_state_dict(loaded_ssnc_state)
            print("[SSNCMask] restored EMA controller state from checkpoint")
        dt_model._ssnc_mask_controller = ssnc_controller
        ssnc_diagnostics_path = os.path.join(
            dir_out,
            str(getattr(args, "ssnc_diagnostics_filename", "ssnc_mask_diagnostics.jsonl")),
        )
    dt_model.scale_param_mask_conflict_mode = scale_param_mask_conflict_mode

    def _append_ssnc_diagnostic(record):
        if not ssnc_diagnostics_path:
            return
        os.makedirs(os.path.dirname(ssnc_diagnostics_path), exist_ok=True)
        with open(ssnc_diagnostics_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def _sample_scale_param_mask_batch(group):
        batch_size_probe = max(1, int(getattr(args, "scale_param_mask_probe_batch_size", 1) or 1))
        max_attempts = max(batch_size_probe, int(getattr(args, "scale_param_mask_probe_max_attempts", 300) or 300))
        samples = []
        if len(train_dataset) <= 0:
            return None
        for _ in range(max_attempts):
            idx = int(scale_param_mask_rng.integers(0, len(train_dataset)))
            try:
                sample = train_dataset[idx]
            except Exception:
                continue
            if _scale_param_mask_group_from_sample(sample) == int(group):
                samples.append(sample)
                if len(samples) >= batch_size_probe:
                    break
        if not samples:
            return None
        return collate_fn(samples)

    def _scale_param_mask_tables():
        tables = {}
        for name, attr in [
            ("gnn", "scale_param_mask_gnn_encoder"),
            ("state", "scale_param_mask_state_projection"),
            ("lora", "scale_param_mask_lora"),
            ("candidate", "scale_param_mask_candidate_head"),
            ("adapter", "scale_param_mask_scale_adapter"),
        ]:
            table = getattr(dt_model, attr, None)
            if torch.is_tensor(table):
                tables[name] = table
        return tables

    @torch.no_grad()
    def _initialize_scale_param_masks_to_fixed_sparsity():
        tables = _scale_param_mask_tables()
        if not tables:
            return
        sparsity = max(0.0, min(0.95, float(getattr(args, "scale_param_mask_sparsity", 0.10) or 0.0)))
        if sparsity <= 0.0:
            return
        base_seed = int(getattr(args, "seed", 42) or 42) + 26017
        summaries = []
        for table_idx, (mask_name, mask_table) in enumerate(sorted(tables.items())):
            width = int(mask_table.shape[1])
            target_zeros = int(round(width * sparsity))
            if target_zeros <= 0:
                continue
            adjusted = 0
            exact = 0
            for group in scale_param_mask_groups:
                group = int(group)
                if group >= int(mask_table.shape[0]):
                    continue
                row = mask_table[group]
                current_zeros = int((row <= 0.5).sum().item())
                if current_zeros == target_zeros:
                    exact += 1
                    continue
                gen = torch.Generator(device="cpu")
                gen.manual_seed(base_seed + table_idx * 1000003 + group * 9176 + width)
                if current_zeros < target_zeros:
                    active_idx = torch.nonzero(row.detach().to(device="cpu") > 0.5, as_tuple=False).flatten()
                    need_close = min(target_zeros - current_zeros, int(active_idx.numel()))
                    if need_close > 0:
                        perm = torch.randperm(int(active_idx.numel()), generator=gen)[:need_close]
                        close_idx = active_idx[perm].to(device=row.device)
                        row[close_idx] = 0.0
                else:
                    inactive_idx = torch.nonzero(row.detach().to(device="cpu") <= 0.5, as_tuple=False).flatten()
                    need_open = min(current_zeros - target_zeros, int(inactive_idx.numel()))
                    if need_open > 0:
                        perm = torch.randperm(int(inactive_idx.numel()), generator=gen)[:need_open]
                        open_idx = inactive_idx[perm].to(device=row.device)
                        row[open_idx] = 1.0
                adjusted += 1
            if adjusted > 0 or exact > 0:
                summaries.append(
                    f"{mask_name}:target_zeros={target_zeros}/{width},"
                    f"adjusted={adjusted},exact={exact}"
                )
        if summaries:
            print("[ScaleParamMask:init_fixed_sparsity] " + "; ".join(summaries))

    def _collect_lora_rank_probe_gradient(width):
        rank_grad = None
        rank_count = None
        for name, param in dt_model.base_model.named_parameters():
            grad = param.grad
            if grad is None or not torch.is_tensor(grad):
                continue
            grad = grad.detach().to(device="cpu", dtype=torch.float32)
            contrib = None
            if "lora_A" in name and grad.dim() >= 2 and int(grad.shape[0]) == int(width):
                reduce_dims = tuple(range(1, grad.dim()))
                contrib = grad.mean(dim=reduce_dims)
            elif "lora_B" in name and grad.dim() >= 2 and int(grad.shape[1]) == int(width):
                reduce_dims = (0,) + tuple(range(2, grad.dim()))
                contrib = grad.mean(dim=reduce_dims)
            if contrib is None:
                continue
            if rank_grad is None:
                rank_grad = torch.zeros_like(contrib)
                rank_count = torch.zeros_like(contrib)
            rank_grad += contrib
            rank_count += 1.0
        if rank_grad is None or rank_count is None:
            return None
        return rank_grad / rank_count.clamp_min(1.0)

    def _scale_param_mask_probe_gradients(batch, group):
        tables = _scale_param_mask_tables()
        if batch is None or not tables:
            return None
        for table in tables.values():
            if int(group) >= int(table.shape[0]):
                return None

        was_training = dt_model.training
        dt_model.train()
        optimizer.zero_grad(set_to_none=True)
        forward_tables = {name: table for name, table in tables.items() if name != "lora"}
        for table in forward_tables.values():
            if table.grad is not None:
                table.grad = None
            table.requires_grad_(True)
        if hasattr(dt_model, "_scale_param_disable_lora_grad_mask"):
            dt_model._scale_param_disable_lora_grad_mask = True

        states = batch["states"]
        if not isinstance(states, dict):
            states = states.to(device)
        actions = batch["actions"].to(device)
        expert_actions = batch.get("expert_actions", None)
        if expert_actions is not None:
            expert_actions = expert_actions.to(device)
        returns_to_go = batch["returns_to_go"].to(device)
        timesteps = batch["timesteps"].to(device)
        batch_mask = batch["mask"].to(device)

        model_input_actions = actions.clone()
        model_input_actions[model_input_actions == -100] = dt_model.n_jobs
        model_actions = build_prev_action_inputs(model_input_actions, dt_model.n_jobs)
        target_actions = expert_actions if expert_actions is not None else actions
        action_mask = None
        if bool(getattr(args, "use_train_action_mask", False)):
            action_mask = build_graph_action_mask(states, dt_model.n_jobs)
            action_mask = include_target_actions_in_mask(action_mask, target_actions, dt_model.n_jobs)
        ce_mask = build_choice_weighted_loss_mask(
            batch_mask,
            action_mask,
            single_action_weight=getattr(args, "single_action_loss_weight", 1.0),
        )

        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            outputs = dt_model(
                states,
                returns_to_go,
                timesteps,
                model_actions,
                attention_mask=batch_mask,
                action_mask=action_mask,
            )
            logits = torch.clamp(outputs.logits, min=-1e4, max=1e4)
            value_preds = getattr(outputs, "value_preds", None)
            if logits.shape[1] != target_actions.shape[1]:
                logits = logits[:, :target_actions.shape[1], :]
                ce_mask = ce_mask[:, :target_actions.shape[1]]
                batch_mask = batch_mask[:, :target_actions.shape[1]]
                if action_mask is not None and action_mask.shape[1] > target_actions.shape[1]:
                    action_mask = action_mask[:, :target_actions.shape[1], :]
            value_targets = None
            if bool(getattr(args, "use_value_aux_loss", False)) and value_preds is not None:
                if returns_to_go.dim() == 3:
                    value_targets = returns_to_go[:, :target_actions.shape[1], 0].to(logits.device, dtype=value_preds.dtype)
                elif returns_to_go.dim() == 2:
                    value_targets = returns_to_go[:, :target_actions.shape[1]].to(logits.device, dtype=value_preds.dtype)
            loss = compute_total_loss(
                logits,
                target_actions.to(logits.device),
                ce_mask.to(logits.device),
                unlikelihood_weight=args.unlikelihood_weight,
                num_negatives=args.unlikelihood_negatives,
                action_valid_mask=action_mask,
                label_smoothing=args.label_smoothing,
                value_preds=value_preds,
                value_targets=value_targets,
                value_loss_weight=args.value_loss_weight if bool(getattr(args, "use_value_aux_loss", False)) else 0.0,
            )

        if torch.isfinite(loss):
            loss.backward()
        gradients = {}
        for name, table in forward_tables.items():
            if table.grad is not None:
                gradients[name] = table.grad[int(group)].detach().to(device="cpu", dtype=torch.float32).clone()
            table.requires_grad_(False)
            table.grad = None
        if "lora" in tables:
            lora_grad = _collect_lora_rank_probe_gradient(int(tables["lora"].shape[1]))
            if lora_grad is not None:
                gradients["lora"] = lora_grad
        if hasattr(dt_model, "_scale_param_disable_lora_grad_mask"):
            dt_model._scale_param_disable_lora_grad_mask = False
        optimizer.zero_grad(set_to_none=True)
        if not was_training:
            dt_model.eval()
        return gradients

    @torch.no_grad()
    def _apply_single_scale_param_mask_update(mask_name, mask_table, group_gradients, global_step_value):
        available = {
            int(group): grads[mask_name].to(device=mask_table.device, dtype=torch.float32)
            for group, grads in group_gradients.items()
            if grads is not None and mask_name in grads and grads[mask_name] is not None
        }
        if not available:
            return
        avg_grad = None
        ssnc_scores = None
        ssnc_diagnostic = None
        if scale_param_mask_conflict_mode == "ssnc_neighbor":
            ssnc_scores, ssnc_diagnostic = ssnc_controller.compute_scores(
                mask_name=mask_name,
                gradients={group: grad.detach().cpu() for group, grad in available.items()},
                gamma=float(getattr(args, "scale_param_mask_gamma", 10.0) or 0.0),
            )
        else:
            # Exact legacy behavior: every scale is compared with the unnormalized
            # mean gradient over all available scales.
            avg_grad = torch.stack(list(available.values()), dim=0).mean(dim=0)
        width = int(mask_table.shape[1])
        target_zeros = int(round(width * max(0.0, min(0.95, float(getattr(args, "scale_param_mask_sparsity", 0.10) or 0.0)))))
        base_change = int(round(width * max(0.0, float(getattr(args, "scale_param_mask_change_frac", 0.005) or 0.0))))
        min_change = max(1, int(getattr(args, "scale_param_mask_min_change", 1) or 1))
        change_num = max(min_change, base_change) if target_zeros > 0 else 0
        change_num = min(change_num, target_zeros, width - target_zeros)
        gamma = float(getattr(args, "scale_param_mask_gamma", 10.0) or 0.0)
        summaries = []
        for group, grad in sorted(available.items()):
            if group >= int(mask_table.shape[0]) or change_num <= 0:
                continue
            row = mask_table[group]
            active_before = row > 0.5
            inactive_before = ~active_before
            current_zeros = int(inactive_before.sum().item())
            if scale_param_mask_conflict_mode == "ssnc_neighbor":
                ranking_score = ssnc_scores[group].to(device=row.device, dtype=torch.float32)
            else:
                harmony = grad.to(device=row.device, dtype=torch.float32) * avg_grad.to(device=row.device, dtype=torch.float32)
                importance = grad.to(device=row.device, dtype=torch.float32).pow(2)
            close_n = 0
            open_n = 0
            if current_zeros < target_zeros:
                close_n = min(change_num, target_zeros - current_zeros, int(active_before.sum().item()))
            elif current_zeros > target_zeros:
                open_n = min(change_num, current_zeros - target_zeros, int(inactive_before.sum().item()))
            else:
                close_n = min(change_num, int(active_before.sum().item()))
                open_n = min(close_n, int(inactive_before.sum().item()))
                close_n = open_n

            if close_n > 0:
                close_score = ranking_score if scale_param_mask_conflict_mode == "ssnc_neighbor" else gamma * harmony + importance
                close_score = close_score.masked_fill(~active_before, float("inf"))
                close_idx = torch.topk(close_score, k=close_n, largest=False).indices
                row[close_idx] = 0.0
            if open_n > 0:
                open_score = ranking_score if scale_param_mask_conflict_mode == "ssnc_neighbor" else harmony
                open_score = open_score.masked_fill(~inactive_before, float("-inf"))
                open_idx = torch.topk(open_score, k=open_n, largest=True).indices
                row[open_idx] = 1.0
            density = float((row > 0.5).to(dtype=torch.float32).mean().item())
            summaries.append(f"{_scale_param_mask_group_label(group)}:density={density:.3f},close={close_n},open={open_n}")
            if ssnc_diagnostic is not None and group in ssnc_diagnostic.get("groups", {}):
                ssnc_diagnostic["groups"][group].update(
                    density=density,
                    close_count=int(close_n),
                    open_count=int(open_n),
                    flip_rate=float(close_n + open_n) / float(max(1, width)),
                )
        if not summaries:
            return

        jaccards = []
        valid_groups = [g for g in sorted(available.keys()) if g < int(mask_table.shape[0])]
        for left_idx, left_group in enumerate(valid_groups):
            left_closed = mask_table[left_group] <= 0.5
            for right_group in valid_groups[left_idx + 1:]:
                right_closed = mask_table[right_group] <= 0.5
                union = (left_closed | right_closed).sum()
                if int(union.item()) == 0:
                    jaccards.append(1.0)
                else:
                    intersection = (left_closed & right_closed).sum()
                    jaccards.append(float((intersection.to(dtype=torch.float32) / union.to(dtype=torch.float32)).item()))
        log_family = "SSNCMask" if scale_param_mask_conflict_mode == "ssnc_neighbor" else "ScaleParamMask"
        msg = f"[{log_family}:{mask_name}] step={global_step_value} " + "; ".join(summaries)
        if jaccards:
            msg += (
                f" | closed_jaccard_mean={float(np.mean(jaccards)):.3f}"
                f",min={float(np.min(jaccards)):.3f},max={float(np.max(jaccards)):.3f}"
            )
        print(msg)
        if ssnc_diagnostic is not None:
            _append_ssnc_diagnostic(
                {
                    "step": int(global_step_value),
                    "mask": str(mask_name),
                    "mode": "ssnc_neighbor",
                    "structure_vectors": {
                        str(group): vector.tolist()
                        for group, vector in sorted(ssnc_controller.structure_ema.items())
                    },
                    **ssnc_diagnostic,
                }
            )

    def _maybe_update_scale_param_mask(global_step_value):
        interval = int(getattr(args, "scale_param_mask_update_interval", 0) or 0)
        warmup = int(getattr(args, "scale_param_mask_warmup_steps", 0) or 0)
        if interval <= 0 or global_step_value <= 0 or global_step_value % interval != 0:
            return
        if global_step_value < warmup:
            return
        if not bool(getattr(args, "use_scale_param_mask", False)):
            return
        tables = _scale_param_mask_tables()
        if not tables:
            return
        group_gradients = {}
        group_structures = {}
        for group in scale_param_mask_groups:
            batch = _sample_scale_param_mask_batch(group)
            if batch is None:
                print(
                    f"[ScaleParamMask] step={global_step_value} "
                    f"skip {_scale_param_mask_group_label(group)}: no probe batch found"
                )
                continue
            if scale_param_mask_conflict_mode == "ssnc_neighbor":
                group_structures[int(group)] = extract_probe_structure_features(
                    batch=batch,
                    group=int(group),
                    max_jobs=int(model_max_jobs),
                    max_machines=int(model_max_machines),
                )
            grads = _scale_param_mask_probe_gradients(batch, group)
            if grads and all(torch.isfinite(g).all() for g in grads.values()):
                group_gradients[group] = grads
        if scale_param_mask_conflict_mode == "ssnc_neighbor":
            ssnc_controller.update_structures(group_structures)
        for mask_name, mask_table in tables.items():
            _apply_single_scale_param_mask_update(mask_name, mask_table, group_gradients, global_step_value)

    if bool(getattr(args, "use_scale_param_mask", False)):
        _initialize_scale_param_masks_to_fixed_sparsity()
        print(
            "[ScaleParamMask] enabled: "
            f"group_by={scale_param_mask_group_by}, "
            f"groups={[ _scale_param_mask_group_label(g) for g in scale_param_mask_groups ]}, "
            f"update_interval={int(getattr(args, 'scale_param_mask_update_interval', 0) or 0)}, "
            f"warmup_steps={int(getattr(args, 'scale_param_mask_warmup_steps', 0) or 0)}, "
            f"sparsity={float(getattr(args, 'scale_param_mask_sparsity', 0.10) or 0.0):.3f}, "
            f"change_frac={float(getattr(args, 'scale_param_mask_change_frac', 0.005) or 0.0):.4f}, "
            f"conflict_mode={scale_param_mask_conflict_mode}, "
            f"targets={list(_scale_param_mask_tables().keys())}"
        )

    # Training loop
    global_step = 0
    best_eval_loss = float('inf')
    best_eval_loss_step = 0          # 记录最佳验证loss对应的step
    best_eval_gap = float('inf')
    best_eval_gap_step = 0
    early_stop_patience = int(getattr(args, 'early_stop_patience', 8))  # 连续N次eval不改善则早停
    early_stop_counter = 0           # 当前连续未改善次数
    best_model_path = None           # 最佳模型保存路径
    best_gap_model_path = None       # 最佳gap模型保存路径

    # Lists to store loss and gap history for visualization
    train_loss_history = []
    eval_loss_history = []
    eval_gap_history = []  # 新增：保存gap历史记录
    eval_gap_steps_history = []  # 新增：为gap单独维护step list
    global_steps_history = []
    eval_steps_history = []
    rollout_gap_records = []
    accum_metric_buffer = []
    training_metrics_jsonl = os.path.join(dir_out, "training_metrics.jsonl")

    print(f"Starting training for {args.num_train_epochs} epochs...")
    print(f"Debug - Device: {device}, Load in 4-bit: {args.load_in_4bit}")
    print(f"Debug - Train loader size: {len(train_loader)}")
    print(f"Debug - Gradient accumulation steps: {args.gradient_accumulation_steps}")

    for epoch in range(args.num_train_epochs):
        print(f"Epoch {epoch + 1}/{args.num_train_epochs}")

        # ── 迁移学习：Phase 2 解冻判断 ──────────────────────────────────────
        if getattr(args, "_transfer_freeze_epochs", 0) > 0:
            if epoch == args._transfer_freeze_epochs and getattr(args, "_transfer_phase", 0) == 1:
                args._transfer_phase = 2
                enable_lora_now = True
                _set_trainable_parameters(enable_lora=enable_lora_now)
                _cast_trainable_parameters_to_fp32()
                _log_trainable_parameters()
                optimizer, scheduler = _build_optimizer_and_scheduler(enable_lora=enable_lora_now)
                optimizer.zero_grad()
                print(f"[Transfer Learning] Phase 2 启动（epoch {epoch+1}）：解冻 LoRA，联合微调")

        if getattr(args, "_freeze_lora_epochs", 0) > 0:
            if epoch == args._freeze_lora_epochs and getattr(args, "_lora_phase", 0) == 1:
                args._lora_phase = 2
                enable_lora_now = True
                _set_trainable_parameters(enable_lora=enable_lora_now)
                _cast_trainable_parameters_to_fp32()
                _log_trainable_parameters()
                optimizer, scheduler = _build_optimizer_and_scheduler(enable_lora=enable_lora_now)
                optimizer.zero_grad()
                print(f"[LoRA Warmup] Phase 2 启动（epoch {epoch+1}）：LoRA 解冻，开始联合微调")

        if bool(getattr(args, "use_curriculum_learning", False)):
            train_loader = build_curriculum_loader(
                train_trajectories=train_trajectories,
                args=args,
                max_actions=max_actions,
                epoch_idx=epoch,
                total_epochs=args.num_train_epochs,
            )
            # 同步重建 eval_loader，使其难度分布与当前 curriculum stage 一致
            # 避免 stage3（全量最难数据）训练时用 stage1 的 eval 数据评估，导致 loss 虚高
            stages = max(1, int(getattr(args, 'curriculum_stages', 3) or 3))
            cur_stage = min(stages - 1, int((epoch * stages) / max(1, args.num_train_epochs)))
            ratio = float(cur_stage + 1) / float(stages)
            ordered_eval = sorted(list(eval_trajectories), key=_trajectory_difficulty)
            keep_eval = max(1, int(len(ordered_eval) * ratio))
            stage_eval_trajectories = ordered_eval[:keep_eval]
            stage_eval_dataset = process_jssp_trajectories_for_dt(
                stage_eval_trajectories,
                state_dim=args.state_dim,
                act_dim=max_actions,
                max_length=args.max_length,
                enable_augmentation=False,
                rtg_scale=getattr(args, 'rtg_scale', 1.0) if bool(getattr(args, 'use_data_augmentation', False)) else 1.0,
                max_jobs=model_max_jobs,
                max_machines=model_max_machines,
                rtg_mode=getattr(args, "training_rtg_mode", "normalized_budget"),
                use_gap_weighted_loss=args.use_gap_weighted_loss,
                gap_weight_alpha=args.gap_weight_alpha,
                gap_weight_power=args.gap_weight_power,
                quality_weight_mix=args.quality_weight_mix,
                min_sample_weight=args.min_sample_weight,
                max_sample_weight=args.max_sample_weight,
                state_tensorization_mode=args.state_tensorization_mode,
            )
            eval_loader = create_data_loader(
                stage_eval_dataset,
                batch_size=args.per_device_eval_batch_size,
                shuffle=False,
                num_workers=0
            )
            print(f"[Curriculum] Eval loader 同步更新：stage {cur_stage+1}/{stages}，使用 {len(stage_eval_trajectories)}/{len(eval_trajectories)} 条评估轨迹")

        # Set model to training mode
        dt_model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(train_loader):
            max_optimizer_steps = int(getattr(args, "max_optimizer_steps", 0) or 0)
            if max_optimizer_steps > 0 and global_step >= max_optimizer_steps:
                break
            if args.debug_prints:
                print(f"  Step {step+1} - Processing batch...")
            # Move batch to device
            states = batch['states']
            if not isinstance(states, dict):
                states = states.to(device)
            actions = batch['actions'].to(device)
            expert_actions = batch.get('expert_actions', None)
            if expert_actions is not None:
                expert_actions = expert_actions.to(device)
            sample_weights = batch.get('sample_weight', None)
            returns_to_go = batch['returns_to_go'].to(device)
            timesteps = batch['timesteps'].to(device)
            batch_mask = batch['mask'].to(device)

            # Decision Transformer training: predict the current action at S_t.
            # Action tokens are previous actions aligned with states:
            #   inputs  = [dummy, A0, A1, ...]
            #   targets = [A0,    A1, A2, ...]
            input_states = states
            input_returns_to_go = returns_to_go
            input_actions = actions
            model_input_actions = input_actions.clone()
            model_input_actions[model_input_actions == -100] = dt_model.n_jobs

            model_states = input_states
            model_returns_to_go = input_returns_to_go
            model_timesteps = timesteps
            model_actions = build_prev_action_inputs(model_input_actions, dt_model.n_jobs)
            target_actions = expert_actions if expert_actions is not None else input_actions
            attn_mask = batch_mask
            loss_mask = batch_mask
            model_action_mask = None
            if bool(getattr(args, "use_train_action_mask", False)):
                model_action_mask = build_graph_action_mask(model_states, dt_model.n_jobs)
                model_action_mask = include_target_actions_in_mask(model_action_mask, target_actions, dt_model.n_jobs)
            action_mask_stats = compute_action_mask_stats(model_action_mask, loss_mask, target_actions)
            ce_loss_mask = build_choice_weighted_loss_mask(
                loss_mask,
                model_action_mask,
                single_action_weight=getattr(args, "single_action_loss_weight", 1.0),
            )

            with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp_enabled):
                outputs = dt_model(
                    model_states,
                    model_returns_to_go,
                    model_timesteps,
                    model_actions,
                    attention_mask=attn_mask,
                    action_mask=model_action_mask
                )

                logits = outputs.logits
                value_preds = getattr(outputs, "value_preds", None)

            if args.debug_prints:
                print(f"\n=== Debug Shapes at Step {step} ===")
                print(f"actions: {actions.shape}")
                print(f"model_actions: {model_actions.shape}")
                print(f"target_actions: {target_actions.shape}")
                print(f"timesteps: {timesteps.shape}")
                print(f"model_timesteps: {model_timesteps.shape}")
                print(f"loss_mask (before reshape): {loss_mask.shape}")
                print(f"logits shape: {logits.shape}")
                print(f"target_actions (after reshape): {target_actions.reshape(-1).shape}")
                print(f"loss_mask (after reshape): {loss_mask.reshape(-1).shape}")
                print("==================================\n")

            # Ensure logits and targets match in shape
            if logits.shape[1] != target_actions.shape[1]:
                logits = logits[:, :target_actions.shape[1], :]

                # 同时截断mask以匹配target_actions的长度
                if loss_mask.shape[1] > target_actions.shape[1]:
                    loss_mask = loss_mask[:, :target_actions.shape[1]]
                    ce_loss_mask = ce_loss_mask[:, :target_actions.shape[1]]
                if model_action_mask is not None and model_action_mask.shape[1] > target_actions.shape[1]:
                    model_action_mask = model_action_mask[:, :target_actions.shape[1], :]

            # Calculate loss using cross entropy between logits and target actions
            # 保留必要的检查
            batch_size, seq_length, num_actions = logits.shape

            # 确保至少有一个有效位置
            if loss_mask.sum() == 0:
                optimizer.zero_grad()
                continue  # 跳过这个batch
            # 确保 target_actions 和 loss_mask 与 logits 在同一设备（4bit量化时必须）
            target_actions = target_actions.to(logits.device)
            loss_mask = loss_mask.to(logits.device)
            ce_loss_mask = ce_loss_mask.to(logits.device)
            if sample_weights is not None:
                sample_weights = sample_weights.to(logits.device)
            value_targets = None
            if bool(getattr(args, "use_value_aux_loss", False)) and value_preds is not None:
                if model_returns_to_go.dim() == 3:
                    if target_actions.shape[1] > 0 and model_returns_to_go.shape[1] > target_actions.shape[1]:
                        value_targets = model_returns_to_go[:, 1:1 + target_actions.shape[1], 0]
                    else:
                        value_targets = model_returns_to_go[:, :target_actions.shape[1], 0]
                elif model_returns_to_go.dim() == 2:
                    if target_actions.shape[1] > 0 and model_returns_to_go.shape[1] > target_actions.shape[1]:
                        value_targets = model_returns_to_go[:, 1:1 + target_actions.shape[1]]
                    else:
                        value_targets = model_returns_to_go[:, :target_actions.shape[1]]
                if value_targets is not None:
                    value_targets = value_targets.to(logits.device, dtype=value_preds.dtype)
            with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp_enabled):
                logits = torch.clamp(logits, min=-1e4, max=1e4)
                loss = compute_total_loss(
                    logits,
                    target_actions,
                    ce_loss_mask,
                    unlikelihood_weight=args.unlikelihood_weight,
                    num_negatives=args.unlikelihood_negatives,
                    action_valid_mask=model_action_mask,
                    label_smoothing=args.label_smoothing,
                    sample_weights=sample_weights,
                    value_preds=value_preds,
                    value_targets=value_targets,
                    value_loss_weight=args.value_loss_weight if bool(getattr(args, "use_value_aux_loss", False)) else 0.0,
                )
                distill_loss = None
                if teacher_model is not None and float(getattr(args, "distill_weight", 0.0) or 0.0) > 0.0:
                    distill_mask = build_old_scale_distill_mask(
                        model_states,
                        loss_mask,
                        max_machines=int(getattr(args, "distill_max_machines", 10) or 0),
                    )
                    if distill_mask is not None and float(distill_mask.sum().detach().item()) > 0.0:
                        with torch.no_grad():
                            teacher_outputs = teacher_model(
                                model_states,
                                model_returns_to_go,
                                model_timesteps,
                                model_actions,
                                attention_mask=attn_mask,
                                action_mask=model_action_mask,
                            )
                            teacher_logits = torch.clamp(teacher_outputs.logits, min=-1e4, max=1e4)
                        distill_loss = compute_logits_distillation_loss(
                            logits,
                            teacher_logits,
                            distill_mask,
                            temperature=float(getattr(args, "distill_temperature", 2.0) or 2.0),
                            action_valid_mask=model_action_mask,
                        )
                        if distill_loss is not None:
                            loss = loss + torch.tensor(float(args.distill_weight), device=loss.device, dtype=loss.dtype) * distill_loss

            acc1 = compute_accuracy(logits, target_actions, loss_mask)
            acc3 = compute_topk_accuracy(logits, target_actions, loss_mask, k=3)
            valid_acc = compute_valid_action_accuracy(logits, target_actions, loss_mask, action_valid_mask=model_action_mask)
            non_single_metrics = compute_non_single_accuracy(logits, target_actions, loss_mask, action_valid_mask=model_action_mask)
            loss_mask_stats = compute_action_mask_stats(model_action_mask, ce_loss_mask, target_actions)

            if args.debug_prints:
                print(f"Debug - Step {step+1}/{len(train_loader)} - Loss: {loss.item()}, Is finite: {torch.isfinite(loss).item()}")
                print(f"Debug - Step {step+1}/{len(train_loader)} - Logits min: {logits.min().item()}, Logits max: {logits.max().item()}, Has NaN: {torch.isnan(logits).any().item()}")
                print(f"Debug - Step {step+1}/{len(train_loader)} - Target actions min: {target_actions.min().item()}, Target actions max: {target_actions.max().item()}, Has NaN: {torch.isnan(target_actions).any().item()}")

            # 检查损失是否为非有限值（NaN/Inf）
            if not torch.isfinite(loss):
                print(f"Warning: non-finite loss detected at step {step+1}, skipping batch")
                optimizer.zero_grad()
                accum_metric_buffer.clear()
                continue

            batch_metric = {
                "loss": float(loss.detach().item()),
                "acc1": float(acc1),
                "acc3": float(acc3),
            }
            if valid_acc is not None:
                _append_metric_if_finite(batch_metric, "valid_acc", valid_acc)
            if action_mask_stats is not None:
                _append_metric_if_finite(batch_metric, "legal_avg", action_mask_stats.get("avg_legal"))
                _append_metric_if_finite(batch_metric, "single_pct", action_mask_stats.get("single_pct"))
                _append_metric_if_finite(batch_metric, "rand_acc", action_mask_stats.get("uniform_acc"))
            if loss_mask_stats is not None:
                _append_metric_if_finite(batch_metric, "uniform_ce", loss_mask_stats.get("uniform_ce"))
                _append_metric_if_finite(batch_metric, "excess_ce", float(loss.detach().item()) - float(loss_mask_stats.get("uniform_ce", 0.0)))
            if non_single_metrics is not None:
                for key in ("non_single_acc", "non_single_rand", "non_single_gain"):
                    _append_metric_if_finite(batch_metric, key, non_single_metrics.get(key))
            accum_metric_buffer.append(batch_metric)

            backward_loss = loss / float(max(1, args.gradient_accumulation_steps))
            if scaler.is_enabled():
                scaler.scale(backward_loss).backward()
            else:
                backward_loss.backward()

            # Gradient accumulation
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # 增强梯度裁剪
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    dt_model.parameters(),
                    max_norm=args.max_grad_norm,
                    error_if_nonfinite=False,
                )
                if not torch.isfinite(grad_norm):
                    print(f"Warning: non-finite gradient norm detected at step {step+1} (grad_norm={grad_norm}), skipping optimizer step")
                    bad_grad_names = []
                    bad_param_names = []
                    for name, param in dt_model.named_parameters():
                        if param.grad is not None and not torch.isfinite(param.grad).all():
                            grad_min = torch.nan_to_num(param.grad.detach(), nan=0.0, posinf=0.0, neginf=0.0).min().item()
                            grad_max = torch.nan_to_num(param.grad.detach(), nan=0.0, posinf=0.0, neginf=0.0).max().item()
                            bad_grad_names.append((name, grad_min, grad_max))
                        if not torch.isfinite(param.detach()).all():
                            param_min = torch.nan_to_num(param.detach(), nan=0.0, posinf=0.0, neginf=0.0).min().item()
                            param_max = torch.nan_to_num(param.detach(), nan=0.0, posinf=0.0, neginf=0.0).max().item()
                            bad_param_names.append((name, param_min, param_max))
                        if len(bad_grad_names) >= 8 and len(bad_param_names) >= 8:
                            break
                    if bad_grad_names:
                        print("Non-finite gradients first found in:")
                        for name, grad_min, grad_max in bad_grad_names[:8]:
                            print(f"  [grad] {name}: finite_min={grad_min:.6g}, finite_max={grad_max:.6g}")
                    else:
                        print("No individual parameter.grad showed NaN/Inf despite non-finite total grad norm")
                    if bad_param_names:
                        print("Non-finite parameter values found in:")
                        for name, param_min, param_max in bad_param_names[:8]:
                            print(f"  [param] {name}: finite_min={param_min:.6g}, finite_max={param_max:.6g}")
                    optimizer.zero_grad()
                    accum_metric_buffer.clear()
                    if scaler.is_enabled():
                        scaler.update()
                    continue

                # Optimizer step
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                if args.debug_prints:
                    has_nan_inf = False
                    grad_nan_count = 0
                    grad_inf_count = 0
                    param_nan_count = 0
                    param_inf_count = 0

                    for name, param in dt_model.named_parameters():
                        if torch.isnan(param.data).any():
                            param_nan_count += 1
                            has_nan_inf = True
                            print(f"Warning: Parameter {name} has NaN values")
                        if torch.isinf(param.data).any():
                            param_inf_count += 1
                            has_nan_inf = True
                            print(f"Warning: Parameter {name} has inf values")
                        if param.grad is not None:
                            if torch.isnan(param.grad).any():
                                grad_nan_count += 1
                                has_nan_inf = True
                                print(f"Warning: Gradient of {name} has NaN values")
                            if torch.isinf(param.grad).any():
                                grad_inf_count += 1
                                has_nan_inf = True
                                print(f"Warning: Gradient of {name} has inf values")

                    print(f"Debug - Step {step+1}/{len(train_loader)} - Gradient NaN count: {grad_nan_count}, Gradient inf count: {grad_inf_count}")
                    print(f"Debug - Step {step+1}/{len(train_loader)} - Parameter NaN count: {param_nan_count}, Parameter inf count: {param_inf_count}")

                    if has_nan_inf:
                        print("检测到NaN或inf值，停止训练")
                        return train_loss_history, eval_loss_history, global_steps_history

                # Update learning rate using combined scheduler
                scheduler.step()

                # Zero gradients
                optimizer.zero_grad()

                global_step += 1
                _maybe_update_candidate_harmony_mask(global_step)
                _maybe_update_scale_adapter_harmony_mask(global_step)
                _maybe_update_scale_param_mask(global_step)

                # Log the accumulated optimizer step instead of only the final micro-batch.
                current_loss = _mean_buffer_metric(accum_metric_buffer, "loss")
                if current_loss is None:
                    current_loss = float(loss.item())
                raw_last_loss = float(loss.item())
                current_acc1 = _mean_buffer_metric(accum_metric_buffer, "acc1")
                current_acc3 = _mean_buffer_metric(accum_metric_buffer, "acc3")
                current_valid_acc = _mean_buffer_metric(accum_metric_buffer, "valid_acc")
                current_acc1 = float(acc1) if current_acc1 is None else current_acc1
                current_acc3 = float(acc3) if current_acc3 is None else current_acc3
                recent_window = max(1, int(getattr(args, "train_loss_ma_window", 100) or 100))
                loss_for_ma = train_loss_history[-(recent_window - 1):] + [current_loss] if recent_window > 1 else [current_loss]
                train_loss_ma = float(np.mean(loss_for_ma))
                current_distill_loss = float(distill_loss.detach().item()) if distill_loss is not None else None
                lr = optimizer.param_groups[0]['lr']

                # Log training metrics
                if global_step % args.logging_steps == 0:
                    extra = ""
                    legal_avg = _mean_buffer_metric(accum_metric_buffer, "legal_avg")
                    single_pct = _mean_buffer_metric(accum_metric_buffer, "single_pct")
                    rand_acc = _mean_buffer_metric(accum_metric_buffer, "rand_acc")
                    uniform_ce = _mean_buffer_metric(accum_metric_buffer, "uniform_ce")
                    excess_ce = _mean_buffer_metric(accum_metric_buffer, "excess_ce")
                    non_single_acc = _mean_buffer_metric(accum_metric_buffer, "non_single_acc")
                    non_single_rand = _mean_buffer_metric(accum_metric_buffer, "non_single_rand")
                    non_single_gain = _mean_buffer_metric(accum_metric_buffer, "non_single_gain")
                    if legal_avg is not None:
                        extra = (
                            f", LastLoss: {raw_last_loss:.6f}"
                            f", LossMA{recent_window}: {train_loss_ma:.6f}"
                            f", LegalAvg: {legal_avg:.2f}"
                            f", Single: {single_pct:.2f}"
                            f", RandAcc: {rand_acc:.3f}"
                            f", AccGain: {current_acc1 - rand_acc:.3f}"
                        )
                    else:
                        extra = f", LastLoss: {raw_last_loss:.6f}, LossMA{recent_window}: {train_loss_ma:.6f}"
                    if uniform_ce is not None:
                        extra += f", UniformCE: {uniform_ce:.3f}, ExcessCE: {excess_ce:.3f}"
                    if non_single_acc is not None:
                        extra += (
                            f", NonSingleAcc: {non_single_acc:.3f}"
                            f", NonSingleRand: {non_single_rand:.3f}"
                            f", NonSingleGain: {non_single_gain:.3f}"
                        )
                    if current_distill_loss is not None:
                        extra += f", Distill: {current_distill_loss:.4f}"
                    if current_valid_acc is None:
                        print(f"Global Step {global_step} - Train Loss: {current_loss:.6f}, Acc@1: {current_acc1:.3f}, Acc@3: {current_acc3:.3f}{extra}, LR: {lr:.8f}")
                    else:
                        print(f"Global Step {global_step} - Train Loss: {current_loss:.6f}, Acc@1: {current_acc1:.3f}, Acc@3: {current_acc3:.3f}, ValidAcc: {current_valid_acc:.3f}{extra}, LR: {lr:.8f}")

                # Log to WandB
                wandb_metrics = {
                    'train_loss': current_loss,
                    'train_loss_raw_last_microbatch': raw_last_loss,
                    'train_loss_ma': train_loss_ma,
                    'train_accuracy': current_acc1,
                    'train_top3_accuracy': current_acc3,
                    'learning_rate': lr,
                    'epoch': epoch,
                    'global_step': global_step
                }
                for key in (
                    "legal_avg", "single_pct", "rand_acc", "uniform_ce", "excess_ce",
                    "non_single_acc", "non_single_rand", "non_single_gain",
                ):
                    value = _mean_buffer_metric(accum_metric_buffer, key)
                    if value is not None:
                        wandb_metrics[f"train_{key}"] = value
                wandb.log(wandb_metrics)

                # Persist every optimizer-step metric independently from console
                # logging frequency. This provides an auditable source for paper
                # plots and avoids reconstructing metrics from mixed text logs.
                rand_acc_value = wandb_metrics.get("train_rand_acc", None)
                uniform_ce_value = wandb_metrics.get("train_uniform_ce", None)
                metric_record = dict(wandb_metrics)
                if rand_acc_value is not None:
                    metric_record["train_acc_gain"] = float(current_acc1) - float(rand_acc_value)
                if uniform_ce_value is not None and float(uniform_ce_value) > 1e-12:
                    metric_record["train_ce_gain"] = float(uniform_ce_value) - float(current_loss)
                    metric_record["train_normalized_ce_gain"] = (
                        float(uniform_ce_value) - float(current_loss)
                    ) / float(uniform_ce_value)
                with open(training_metrics_jsonl, "a", encoding="utf-8") as metric_handle:
                    metric_handle.write(json.dumps(metric_record, ensure_ascii=False, sort_keys=True) + "\n")

                # Save loss history
                train_loss_history.append(current_loss)
                global_steps_history.append(global_step)
                accum_metric_buffer.clear()

                # Evaluation based on global step (handle negative eval_steps by checking if it's a multiple)
                if (args.eval_steps > 0 and global_step % args.eval_steps == 0) or (args.eval_steps < 0 and global_step % abs(args.eval_steps) == 0):
                    dt_model.eval()
                    eval_loss = 0.0
                    eval_acc1 = 0.0
                    eval_acc3 = 0.0
                    eval_mask_stats = []
                    eval_metric_buffer = []
                    eval_batches_seen = 0
                    max_eval_batches = int(getattr(args, "eval_max_batches", 0) or 0)

                    with torch.no_grad():
                        for batch_idx, batch in enumerate(eval_loader):
                            if max_eval_batches > 0 and batch_idx >= max_eval_batches:
                                break
                            # Move batch to device
                            states = batch['states']
                            if not isinstance(states, dict):
                                states = states.to(device)
                            actions = batch['actions'].to(device)
                            sample_weights = batch.get('sample_weight', None)
                            returns_to_go = batch['returns_to_go'].to(device)
                            timesteps = batch['timesteps'].to(device)
                            batch_mask = batch['mask'].to(device)

                            # Decision Transformer evaluation: match training alignment.
                            # Previous-action inputs predict current actions at the same state index.
                            input_states = states
                            input_returns_to_go = returns_to_go
                            input_actions = actions
                            model_input_actions = input_actions.clone()
                            model_input_actions[model_input_actions == -100] = dt_model.n_jobs

                            model_states = input_states
                            model_returns_to_go = input_returns_to_go
                            model_timesteps = timesteps
                            model_actions = build_prev_action_inputs(model_input_actions, dt_model.n_jobs)
                            target_actions = actions
                            attn_mask = batch_mask
                            loss_mask = batch_mask
                            model_action_mask = None
                            if bool(getattr(args, "use_train_action_mask", False)):
                                model_action_mask = build_graph_action_mask(model_states, dt_model.n_jobs)
                                model_action_mask = include_target_actions_in_mask(model_action_mask, target_actions, dt_model.n_jobs)
                            action_mask_stats = compute_action_mask_stats(model_action_mask, loss_mask, target_actions)
                            if action_mask_stats is not None:
                                eval_mask_stats.append(action_mask_stats)
                            ce_loss_mask = build_choice_weighted_loss_mask(
                                loss_mask,
                                model_action_mask,
                                single_action_weight=getattr(args, "single_action_loss_weight", 1.0),
                            )

                            # Forward pass with causal inputs matching training
                            outputs = dt_model(model_states, model_returns_to_go, model_timesteps, model_actions, attention_mask=attn_mask, action_mask=model_action_mask)

                            # Get logits from model outputs
                            logits = outputs.logits
                            value_preds = getattr(outputs, "value_preds", None)

                            # Ensure logits and targets match in shape (same as training)
                            if logits.shape[1] != target_actions.shape[1]:
                                logits = logits[:, :target_actions.shape[1], :]

                                # 同时截断mask以匹配target_actions的长度
                                if loss_mask.shape[1] > target_actions.shape[1]:
                                    loss_mask = loss_mask[:, :target_actions.shape[1]]
                                    ce_loss_mask = ce_loss_mask[:, :target_actions.shape[1]]
                                if model_action_mask is not None and model_action_mask.shape[1] > target_actions.shape[1]:
                                    model_action_mask = model_action_mask[:, :target_actions.shape[1], :]

                            # Calculate loss using cross entropy between logits and target actions
                            # 使用新的loss计算函数，仅在action token位置计算loss
                            batch_size, seq_length, num_actions = logits.shape

                            # 添加数值稳定性处理
                            logits = torch.clamp(logits, min=-1e4, max=1e4)

                            # 使用新的loss计算函数
                            if sample_weights is not None:
                                sample_weights = sample_weights.to(logits.device)
                            value_targets = None
                            if bool(getattr(args, "use_value_aux_loss", False)) and value_preds is not None:
                                if model_returns_to_go.dim() == 3:
                                    if target_actions.shape[1] > 0 and model_returns_to_go.shape[1] > target_actions.shape[1]:
                                        value_targets = model_returns_to_go[:, 1:1 + target_actions.shape[1], 0]
                                    else:
                                        value_targets = model_returns_to_go[:, :target_actions.shape[1], 0]
                                elif model_returns_to_go.dim() == 2:
                                    if target_actions.shape[1] > 0 and model_returns_to_go.shape[1] > target_actions.shape[1]:
                                        value_targets = model_returns_to_go[:, 1:1 + target_actions.shape[1]]
                                    else:
                                        value_targets = model_returns_to_go[:, :target_actions.shape[1]]
                                if value_targets is not None:
                                    value_targets = value_targets.to(logits.device, dtype=value_preds.dtype)
                            loss = compute_total_loss(
                                logits,
                                target_actions,
                                ce_loss_mask,
                                action_valid_mask=model_action_mask,
                                label_smoothing=args.label_smoothing,
                                sample_weights=sample_weights,
                                value_preds=value_preds,
                                value_targets=value_targets,
                                value_loss_weight=args.value_loss_weight if bool(getattr(args, "use_value_aux_loss", False)) else 0.0,
                            )
                            eval_loss += loss.item()

                            acc1 = compute_accuracy(logits, target_actions, loss_mask)
                            acc3 = compute_topk_accuracy(logits, target_actions, loss_mask, k=3)
                            loss_mask_stats = compute_action_mask_stats(model_action_mask, ce_loss_mask, target_actions)
                            non_single_metrics = compute_non_single_accuracy(logits, target_actions, loss_mask, action_valid_mask=model_action_mask)
                            eval_metric = {"loss": float(loss.item()), "acc1": float(acc1), "acc3": float(acc3)}
                            if action_mask_stats is not None:
                                _append_metric_if_finite(eval_metric, "legal_avg", action_mask_stats.get("avg_legal"))
                                _append_metric_if_finite(eval_metric, "single_pct", action_mask_stats.get("single_pct"))
                                _append_metric_if_finite(eval_metric, "rand_acc", action_mask_stats.get("uniform_acc"))
                            if loss_mask_stats is not None:
                                _append_metric_if_finite(eval_metric, "uniform_ce", loss_mask_stats.get("uniform_ce"))
                                _append_metric_if_finite(eval_metric, "excess_ce", float(loss.item()) - float(loss_mask_stats.get("uniform_ce", 0.0)))
                            if non_single_metrics is not None:
                                for key in ("non_single_acc", "non_single_rand", "non_single_gain"):
                                    _append_metric_if_finite(eval_metric, key, non_single_metrics.get(key))
                            eval_metric_buffer.append(eval_metric)
                            eval_acc1 += acc1
                            eval_acc3 += acc3
                            eval_batches_seen += 1

                    # Average evaluation loss and accuracy
                    eval_loader_len = max(1, eval_batches_seen)
                    avg_eval_loss = eval_loss / eval_loader_len
                    avg_eval_acc1 = eval_acc1 / eval_loader_len
                    avg_eval_acc3 = eval_acc3 / eval_loader_len
                    eval_extra = ""
                    if eval_metric_buffer:
                        mean_uniform_acc = _mean_buffer_metric(eval_metric_buffer, "rand_acc")
                        if mean_uniform_acc is not None:
                            eval_extra = (
                                f", LegalAvg: {_fmt_float(_mean_buffer_metric(eval_metric_buffer, 'legal_avg'), 2)}"
                                f", Single: {_fmt_float(_mean_buffer_metric(eval_metric_buffer, 'single_pct'), 2)}"
                                f", RandAcc: {_fmt_float(mean_uniform_acc, 3)}"
                                f", AccGain: {_fmt_float(avg_eval_acc1 - mean_uniform_acc, 3)}"
                            )
                        uniform_ce = _mean_buffer_metric(eval_metric_buffer, "uniform_ce")
                        excess_ce = _mean_buffer_metric(eval_metric_buffer, "excess_ce")
                        non_single_acc = _mean_buffer_metric(eval_metric_buffer, "non_single_acc")
                        if uniform_ce is not None:
                            eval_extra += f", UniformCE: {_fmt_float(uniform_ce, 3)}, ExcessCE: {_fmt_float(excess_ce, 3)}"
                        if non_single_acc is not None:
                            eval_extra += (
                                f", NonSingleAcc: {_fmt_float(non_single_acc, 3)}"
                                f", NonSingleRand: {_fmt_float(_mean_buffer_metric(eval_metric_buffer, 'non_single_rand'), 3)}"
                                f", NonSingleGain: {_fmt_float(_mean_buffer_metric(eval_metric_buffer, 'non_single_gain'), 3)}"
                            )
                    print(f"Global Step {global_step} - Eval Loss: {avg_eval_loss:.6f}, Eval Acc@1: {avg_eval_acc1:.3f}, Eval Acc@3: {avg_eval_acc3:.3f}{eval_extra}")

                    # Log to WandB
                    eval_log = {
                        'eval_loss': avg_eval_loss,
                        'eval_accuracy': avg_eval_acc1,
                        'eval_top3_accuracy': avg_eval_acc3,
                        'epoch': epoch,
                        'global_step': global_step
                    }
                    for key in (
                        "legal_avg", "single_pct", "rand_acc", "uniform_ce", "excess_ce",
                        "non_single_acc", "non_single_rand", "non_single_gain",
                    ):
                        value = _mean_buffer_metric(eval_metric_buffer, key)
                        if value is not None:
                            eval_log[f"eval_{key}"] = value
                    wandb.log(eval_log)

                    # Save evaluation loss history
                    eval_loss_history.append(avg_eval_loss)
                    eval_steps_history.append(global_step)

                    # Save model checkpoint if it's the best so far
                    if avg_eval_loss < best_eval_loss:
                        best_eval_loss = avg_eval_loss
                        best_eval_loss_step = global_step
                        early_stop_counter = 0  # 重置早停计数器
                        save_path = os.path.join(dir_out, f"best_model_step_{global_step}")
                        best_model_path = save_path
                        # 使用统一的模型保存函数
                        save_peft_model(dt_model, save_path)
                        print(f"[EarlyStopping] 验证loss改善到 {best_eval_loss:.6f}，已保存最佳模型")
                    else:
                        early_stop_counter += 1
                        print(f"[EarlyStopping] 验证loss未改善（{early_stop_counter}/{early_stop_patience}），当前={avg_eval_loss:.6f}，最佳={best_eval_loss:.6f} @ step {best_eval_loss_step}")
                        if early_stop_counter >= early_stop_patience:
                            print(f"[EarlyStopping] 连续 {early_stop_patience} 次eval未改善，提前停止训练！最佳验证loss={best_eval_loss:.6f} @ step {best_eval_loss_step}")
                            # 恢复最佳模型权重（自定义模块部分）
                            if best_model_path and os.path.exists(os.path.join(best_model_path, 'dt_custom_modules.pt')):
                                try:
                                    best_custom = torch.load(os.path.join(best_model_path, 'dt_custom_modules.pt'), map_location=device)
                                    for k in ['rtg_proj', 'state_proj', 'state_proj_norm', 'state_encoder', 'action_proj', 'action_head',
                                              'candidate_job_proj', 'candidate_context_proj', 'candidate_fuse_norm', 'candidate_action_head',
                                              'scale_context_adapter_norm', 'scale_context_adapter_down', 'scale_context_adapter_up',
                                              'scale_candidate_residual_norm', 'scale_candidate_residual_down', 'scale_candidate_residual_up',
                                              'value_head', 'timestep_embedding', 'type_embedding', 'scale_embedding']:
                                        sd = best_custom.get(k, None)
                                        m = getattr(dt_model, k, None)
                                        if sd is not None and m is not None:
                                            m.load_state_dict(sd, strict=False)
                                    if getattr(dt_model, "scale_alpha", None) is not None and best_custom.get("scale_alpha", None) is not None:
                                        with torch.no_grad():
                                            dt_model.scale_alpha.copy_(best_custom["scale_alpha"].to(device=dt_model.scale_alpha.device, dtype=dt_model.scale_alpha.dtype))
                                    for param_name in ["scale_context_adapter_alpha", "scale_candidate_residual_alpha"]:
                                        param = getattr(dt_model, param_name, None)
                                        src = best_custom.get(param_name, None)
                                        if param is not None and src is not None:
                                            with torch.no_grad():
                                                rows = min(int(src.shape[0]), int(param.shape[0]))
                                                cols = min(int(src.shape[1]), int(param.shape[1]))
                                                param[:rows, :cols].copy_(src[:rows, :cols].to(device=param.device, dtype=param.dtype))
                                    for mask_name in [
                                        "scale_context_harmony_mask", "scale_candidate_harmony_mask",
                                        "scale_param_mask_gnn_encoder", "scale_param_mask_state_projection",
                                        "scale_param_mask_lora", "scale_param_mask_candidate_head",
                                        "scale_param_mask_scale_adapter",
                                    ]:
                                        dst = getattr(dt_model, mask_name, None)
                                        src = best_custom.get(mask_name, None)
                                        if torch.is_tensor(dst) and torch.is_tensor(src):
                                            with torch.no_grad():
                                                rows = min(int(src.shape[0]), int(dst.shape[0]))
                                                cols = min(int(src.shape[1]), int(dst.shape[1]))
                                                dst[:rows, :cols].copy_(src[:rows, :cols].to(device=dst.device, dtype=dst.dtype))
                                    print(f"[EarlyStopping] 已恢复最佳模型自定义模块权重")
                                except Exception as e:
                                    print(f"[EarlyStopping] 恢复模型权重失败: {e}")
                            # 结束训练前保存图表
                            _do_save_plots(
                                dir_out=dir_out,
                                train_loss_history=train_loss_history,
                                eval_loss_history=eval_loss_history,
                                eval_gap_history=eval_gap_history,
                                eval_gap_steps_history=eval_gap_steps_history,
                                global_steps_history=global_steps_history,
                                eval_steps_history=eval_steps_history,
                                rollout_gap_records=rollout_gap_records,
                            )
                            wandb.finish()
                            return train_loss_history, eval_loss_history, global_steps_history

                    # ✅ 新增：Rollout Evaluation
                    try:
                        # 创建一个新的测试环境（避免污染训练数据分布）
                        import sys
                        import os
                        # 添加项目根目录到路径中
                        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                        from Envs.JSSPEnv.JSSPEnv import JSSPEnv
                        test_env = JSSPEnv(
                            n_job_range=[rollout_eval_n_jobs, rollout_eval_n_jobs],
                            n_mac_range=[rollout_eval_n_machines, rollout_eval_n_machines],
                            seed=42,
                        )

                        with _offload_optimizer_state_for_rollout(
                            optimizer,
                            enabled=bool(getattr(args, "rollout_offload_optimizer_state", False)),
                            restore_device=device,
                        ):
                            rollout_log = {}
                            rollout_record = {
                                "step": int(global_step),
                                "score": float("nan"),
                                "mean_gap_all_instances": float("nan"),
                                "target_return": None,
                                "per_scale": {},
                            }
                            if rollout_eval_panels:
                                rollout_summary = evaluate_rollout_panel_multi_target(
                                    dt_model=dt_model,
                                    panels=rollout_eval_panels,
                                    target_returns=rollout_target_returns,
                                    rtg_mode=args.rollout_rtg_mode,
                                    debug_prints=args.debug_prints,
                                    rollout_top_k=args.rollout_top_k,
                                    rollout_temperature=args.rollout_temperature,
                                    rollout_best_of_n=args.rollout_best_of_n,
                                    rollout_crop_history=bool(getattr(args, "rollout_crop_history", True)),
                                    rollout_eval_max_length=rollout_eval_max_length,
                                    scale_weights=rollout_panel_weights,
                                )
                                avg_gap = float(rollout_summary.get("score", np.nan))
                                rollout_log["eval_relative_gap_panel_score"] = avg_gap
                                rollout_log["eval_relative_gap_all_instances"] = float(rollout_summary.get("mean_gap_all_instances", np.nan))
                                if rollout_summary.get("target_return", None) is not None:
                                    rollout_log["eval_rollout_best_target_return"] = float(rollout_summary["target_return"])
                                rollout_record["score"] = avg_gap
                                rollout_record["mean_gap_all_instances"] = float(rollout_summary.get("mean_gap_all_instances", np.nan))
                                rollout_record["target_return"] = (
                                    float(rollout_summary["target_return"])
                                    if rollout_summary.get("target_return", None) is not None
                                    else None
                                )
                                for scale, stats in rollout_summary.get("per_scale", {}).items():
                                    mean_gap = float(stats.get("mean_gap", np.nan))
                                    valid_instances = int(stats.get("valid_instances", 0))
                                    total_instances = int(stats.get("total_instances", 0))
                                    rollout_log[f"eval_gap_{scale}"] = mean_gap
                                    rollout_log[f"eval_gap_valid_{scale}"] = valid_instances
                                    rollout_record["per_scale"][str(scale)] = {
                                        "mean_gap": mean_gap,
                                        "valid_instances": valid_instances,
                                        "total_instances": total_instances,
                                    }
                            else:
                                avg_gap = evaluate_rollout_multi_target(
                                    dt_model=dt_model,
                                    env=test_env,
                                    num_episodes=args.rollout_episodes,
                                    target_returns=rollout_target_returns,
                                    rtg_mode=args.rollout_rtg_mode,
                                    debug_prints=args.debug_prints,
                                    rollout_top_k=args.rollout_top_k,
                                    rollout_temperature=args.rollout_temperature,
                                    rollout_best_of_n=args.rollout_best_of_n,
                                    rollout_crop_history=bool(getattr(args, "rollout_crop_history", True)),
                                    rollout_eval_max_length=rollout_eval_max_length,
                                )
                                rollout_log["eval_relative_gap"] = avg_gap
                                rollout_record["score"] = float(avg_gap)

                        print(f"Step {global_step} - Evaluation Relative Gap Score: {avg_gap:.2f}%")

                        # Log to wandb (if you restore it later)
                        wandb.log(rollout_log)

                        # 保存gap历史记录和对应的global_step，使用单独的step list
                        eval_gap_history.append(avg_gap)
                        eval_gap_steps_history.append(global_step)
                        rollout_gap_records.append(rollout_record)

                        if not np.isnan(avg_gap) and avg_gap < best_eval_gap:
                            best_eval_gap = float(avg_gap)
                            best_eval_gap_step = global_step
                            save_path = os.path.join(dir_out, f"best_gap_model_step_{global_step}")
                            best_gap_model_path = save_path
                            save_peft_model(dt_model, save_path)
                            print(f"[BestGap] 验证gap改善到 {best_eval_gap:.4f}% ，已保存最佳gap模型")

                    except Exception as e:
                        print(f"Rollout evaluation failed: {e}")
                        # 保存NaN作为默认值，以便在图中区分
                        import numpy as np
                        eval_gap_history.append(np.nan)
                        eval_gap_steps_history.append(global_step)
                        rollout_gap_records.append({
                            "step": int(global_step),
                            "score": float("nan"),
                            "mean_gap_all_instances": float("nan"),
                            "target_return": None,
                            "per_scale": {},
                            "error": str(e),
                        })

                    dt_model.train()

            epoch_loss += loss.item()

        max_optimizer_steps = int(getattr(args, "max_optimizer_steps", 0) or 0)
        if max_optimizer_steps > 0 and global_step >= max_optimizer_steps:
            print(f"[SmokeTest] reached max_optimizer_steps={max_optimizer_steps}; skipping epoch-end eval/rollout.")
            break

        # Average loss for the epoch
        train_loader_len = max(1, len(train_loader))
        avg_epoch_loss = epoch_loss / train_loader_len
        print(f"Epoch {epoch + 1} - Average Train Loss: {avg_epoch_loss:.6f}")

        # Final evaluation at the end of each epoch (optional)
        if (epoch + 1) == args.num_train_epochs:
            dt_model.eval()
            eval_loss = 0.0
            eval_acc1 = 0.0
            eval_acc3 = 0.0
            eval_mask_stats = []
            eval_metric_buffer = []
            eval_batches_seen = 0
            max_eval_batches = int(getattr(args, "eval_max_batches", 0) or 0)

            with torch.no_grad():
                for batch_idx, batch in enumerate(eval_loader):
                    if max_eval_batches > 0 and batch_idx >= max_eval_batches:
                        break
                    # Move batch to device
                    states = batch['states']
                    if not isinstance(states, dict):
                        states = states.to(device)
                    actions = batch['actions'].to(device)
                    sample_weights = batch.get('sample_weight', None)
                    returns_to_go = batch['returns_to_go'].to(device)
                    timesteps = batch['timesteps'].to(device)
                    batch_mask = batch['mask'].to(device)

                    # Decision Transformer evaluation: match training alignment.
                    # Previous-action inputs predict current actions at the same state index.
                    input_states = states
                    input_returns_to_go = returns_to_go
                    input_actions = actions
                    model_input_actions = input_actions.clone()
                    model_input_actions[model_input_actions == -100] = dt_model.n_jobs

                    model_states = input_states
                    model_returns_to_go = input_returns_to_go
                    model_timesteps = timesteps
                    model_actions = build_prev_action_inputs(model_input_actions, dt_model.n_jobs)
                    target_actions = actions
                    attn_mask = batch_mask
                    loss_mask = batch_mask
                    model_action_mask = None
                    if bool(getattr(args, "use_train_action_mask", False)):
                        model_action_mask = build_graph_action_mask(model_states, dt_model.n_jobs)
                        model_action_mask = include_target_actions_in_mask(model_action_mask, target_actions, dt_model.n_jobs)
                    action_mask_stats = compute_action_mask_stats(model_action_mask, loss_mask, target_actions)
                    if action_mask_stats is not None:
                        eval_mask_stats.append(action_mask_stats)
                    ce_loss_mask = build_choice_weighted_loss_mask(
                        loss_mask,
                        model_action_mask,
                        single_action_weight=getattr(args, "single_action_loss_weight", 1.0),
                    )

                    # Forward pass with causal inputs matching training
                    outputs = dt_model(model_states, model_returns_to_go, model_timesteps, model_actions, attention_mask=attn_mask, action_mask=model_action_mask)

                    # Get logits from model outputs
                    logits = outputs.logits
                    value_preds = getattr(outputs, "value_preds", None)

                    # Ensure logits and targets match in shape (same as training)
                    if logits.shape[1] != target_actions.shape[1]:
                        logits = logits[:, :target_actions.shape[1], :]

                        # 同时截断mask以匹配target_actions的长度
                        if loss_mask.shape[1] > target_actions.shape[1]:
                            loss_mask = loss_mask[:, :target_actions.shape[1]]
                            ce_loss_mask = ce_loss_mask[:, :target_actions.shape[1]]
                        if model_action_mask is not None and model_action_mask.shape[1] > target_actions.shape[1]:
                            model_action_mask = model_action_mask[:, :target_actions.shape[1], :]

                    # Calculate loss using cross entropy between logits and target actions
                    # 使用新的loss计算函数，仅在action token位置计算loss
                    batch_size, seq_length, num_actions = logits.shape

                    # 添加数值稳定性处理
                    logits = torch.clamp(logits, min=-1e4, max=1e4)

                    # 使用新的loss计算函数，仅在action token位置计算loss
                    if loss_mask.sum() > 0:
                        if sample_weights is not None:
                            sample_weights = sample_weights.to(logits.device)
                        value_targets = None
                        if bool(getattr(args, "use_value_aux_loss", False)) and value_preds is not None:
                            if model_returns_to_go.dim() == 3:
                                if target_actions.shape[1] > 0 and model_returns_to_go.shape[1] > target_actions.shape[1]:
                                    value_targets = model_returns_to_go[:, 1:1 + target_actions.shape[1], 0]
                                else:
                                    value_targets = model_returns_to_go[:, :target_actions.shape[1], 0]
                            elif model_returns_to_go.dim() == 2:
                                if target_actions.shape[1] > 0 and model_returns_to_go.shape[1] > target_actions.shape[1]:
                                    value_targets = model_returns_to_go[:, 1:1 + target_actions.shape[1]]
                                else:
                                    value_targets = model_returns_to_go[:, :target_actions.shape[1]]
                            if value_targets is not None:
                                value_targets = value_targets.to(logits.device, dtype=value_preds.dtype)
                        loss = compute_total_loss(
                            logits,
                            target_actions,
                            ce_loss_mask,
                            action_valid_mask=model_action_mask,
                            label_smoothing=args.label_smoothing,
                            sample_weights=sample_weights,
                            value_preds=value_preds,
                            value_targets=value_targets,
                            value_loss_weight=args.value_loss_weight if bool(getattr(args, "use_value_aux_loss", False)) else 0.0,
                        )
                        eval_loss += loss.item()

                        acc1 = compute_accuracy(logits, target_actions, loss_mask)
                        acc3 = compute_topk_accuracy(logits, target_actions, loss_mask, k=3)
                        loss_mask_stats = compute_action_mask_stats(model_action_mask, ce_loss_mask, target_actions)
                        non_single_metrics = compute_non_single_accuracy(logits, target_actions, loss_mask, action_valid_mask=model_action_mask)
                        eval_metric = {"loss": float(loss.item()), "acc1": float(acc1), "acc3": float(acc3)}
                        if action_mask_stats is not None:
                            _append_metric_if_finite(eval_metric, "legal_avg", action_mask_stats.get("avg_legal"))
                            _append_metric_if_finite(eval_metric, "single_pct", action_mask_stats.get("single_pct"))
                            _append_metric_if_finite(eval_metric, "rand_acc", action_mask_stats.get("uniform_acc"))
                        if loss_mask_stats is not None:
                            _append_metric_if_finite(eval_metric, "uniform_ce", loss_mask_stats.get("uniform_ce"))
                            _append_metric_if_finite(eval_metric, "excess_ce", float(loss.item()) - float(loss_mask_stats.get("uniform_ce", 0.0)))
                        if non_single_metrics is not None:
                            for key in ("non_single_acc", "non_single_rand", "non_single_gain"):
                                _append_metric_if_finite(eval_metric, key, non_single_metrics.get(key))
                        eval_metric_buffer.append(eval_metric)
                        eval_acc1 += acc1
                        eval_acc3 += acc3
                        eval_batches_seen += 1
                    else:
                        # 如果没有有效位置，跳过这个batch的损失计算
                        continue

            # Average evaluation loss and accuracy
            eval_loader_len = max(1, eval_batches_seen)
            avg_eval_loss = eval_loss / eval_loader_len
            avg_eval_acc1 = eval_acc1 / eval_loader_len
            avg_eval_acc3 = eval_acc3 / eval_loader_len
            eval_extra = ""
            if eval_metric_buffer:
                mean_uniform_acc = _mean_buffer_metric(eval_metric_buffer, "rand_acc")
                if mean_uniform_acc is not None:
                    eval_extra = (
                        f", LegalAvg: {_fmt_float(_mean_buffer_metric(eval_metric_buffer, 'legal_avg'), 2)}"
                        f", Single: {_fmt_float(_mean_buffer_metric(eval_metric_buffer, 'single_pct'), 2)}"
                        f", RandAcc: {_fmt_float(mean_uniform_acc, 3)}"
                        f", AccGain: {_fmt_float(avg_eval_acc1 - mean_uniform_acc, 3)}"
                    )
                uniform_ce = _mean_buffer_metric(eval_metric_buffer, "uniform_ce")
                excess_ce = _mean_buffer_metric(eval_metric_buffer, "excess_ce")
                non_single_acc = _mean_buffer_metric(eval_metric_buffer, "non_single_acc")
                if uniform_ce is not None:
                    eval_extra += f", UniformCE: {_fmt_float(uniform_ce, 3)}, ExcessCE: {_fmt_float(excess_ce, 3)}"
                if non_single_acc is not None:
                    eval_extra += (
                        f", NonSingleAcc: {_fmt_float(non_single_acc, 3)}"
                        f", NonSingleRand: {_fmt_float(_mean_buffer_metric(eval_metric_buffer, 'non_single_rand'), 3)}"
                        f", NonSingleGain: {_fmt_float(_mean_buffer_metric(eval_metric_buffer, 'non_single_gain'), 3)}"
                    )
            print(f"Final Epoch {epoch + 1} - Eval Loss: {avg_eval_loss:.6f}, Eval Acc@1: {avg_eval_acc1:.3f}, Eval Acc@3: {avg_eval_acc3:.3f}{eval_extra}")

            # Log to WandB
            eval_log = {
                'eval_loss': avg_eval_loss,
                'eval_accuracy': avg_eval_acc1,
                'eval_top3_accuracy': avg_eval_acc3,
                'epoch': epoch,
                'global_step': global_step
            }
            for key in (
                "legal_avg", "single_pct", "rand_acc", "uniform_ce", "excess_ce",
                "non_single_acc", "non_single_rand", "non_single_gain",
            ):
                value = _mean_buffer_metric(eval_metric_buffer, key)
                if value is not None:
                    eval_log[f"eval_{key}"] = value
            wandb.log(eval_log)

            # Save evaluation loss history
            eval_loss_history.append(avg_eval_loss)
            eval_steps_history.append(global_step)

            # Save model checkpoint if it's the best so far
            if avg_eval_loss < best_eval_loss:
                best_eval_loss = avg_eval_loss
                save_path = os.path.join(dir_out, f"best_model_final_epoch_{epoch + 1}")
                # 使用统一的模型保存函数
                save_peft_model(dt_model, save_path)

            # ✅ 新增：Final Rollout Evaluation at epoch end
            try:
                # 创建一个新的测试环境（避免污染训练数据分布）
                import sys
                import os
                # 添加项目根目录到路径中
                sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from Envs.JSSPEnv.JSSPEnv import JSSPEnv
                test_env = JSSPEnv(
                    n_job_range=[rollout_eval_n_jobs, rollout_eval_n_jobs],
                    n_mac_range=[rollout_eval_n_machines, rollout_eval_n_machines],
                    seed=42,
                )

                with _offload_optimizer_state_for_rollout(
                    optimizer,
                    enabled=bool(getattr(args, "rollout_offload_optimizer_state", False)),
                    restore_device=device,
                ):
                    rollout_log = {}
                    rollout_record = {
                        "step": int(global_step),
                        "score": float("nan"),
                        "mean_gap_all_instances": float("nan"),
                        "target_return": None,
                        "per_scale": {},
                    }
                    if rollout_eval_panels:
                        rollout_summary = evaluate_rollout_panel_multi_target(
                            dt_model=dt_model,
                            panels=rollout_eval_panels,
                            target_returns=rollout_target_returns,
                            rtg_mode=args.rollout_rtg_mode,
                            debug_prints=args.debug_prints,
                            rollout_top_k=args.rollout_top_k,
                            rollout_temperature=args.rollout_temperature,
                            rollout_best_of_n=args.rollout_best_of_n,
                            rollout_crop_history=bool(getattr(args, "rollout_crop_history", True)),
                            rollout_eval_max_length=rollout_eval_max_length,
                            scale_weights=rollout_panel_weights,
                        )
                        avg_gap = float(rollout_summary.get("score", np.nan))
                        rollout_log["eval_relative_gap_panel_score"] = avg_gap
                        rollout_log["eval_relative_gap_all_instances"] = float(rollout_summary.get("mean_gap_all_instances", np.nan))
                        if rollout_summary.get("target_return", None) is not None:
                            rollout_log["eval_rollout_best_target_return"] = float(rollout_summary["target_return"])
                        rollout_record["score"] = avg_gap
                        rollout_record["mean_gap_all_instances"] = float(rollout_summary.get("mean_gap_all_instances", np.nan))
                        rollout_record["target_return"] = (
                            float(rollout_summary["target_return"])
                            if rollout_summary.get("target_return", None) is not None
                            else None
                        )
                        for scale, stats in rollout_summary.get("per_scale", {}).items():
                            mean_gap = float(stats.get("mean_gap", np.nan))
                            valid_instances = int(stats.get("valid_instances", 0))
                            total_instances = int(stats.get("total_instances", 0))
                            rollout_log[f"eval_gap_{scale}"] = mean_gap
                            rollout_log[f"eval_gap_valid_{scale}"] = valid_instances
                            rollout_record["per_scale"][str(scale)] = {
                                "mean_gap": mean_gap,
                                "valid_instances": valid_instances,
                                "total_instances": total_instances,
                            }
                    else:
                        avg_gap = evaluate_rollout_multi_target(
                            dt_model=dt_model,
                            env=test_env,
                            num_episodes=args.rollout_episodes,
                            target_returns=rollout_target_returns,
                            rtg_mode=args.rollout_rtg_mode,
                            debug_prints=args.debug_prints,
                            rollout_top_k=args.rollout_top_k,
                            rollout_temperature=args.rollout_temperature,
                            rollout_best_of_n=args.rollout_best_of_n,
                            rollout_crop_history=bool(getattr(args, "rollout_crop_history", True)),
                            rollout_eval_max_length=rollout_eval_max_length,
                        )
                        rollout_log["eval_relative_gap"] = avg_gap
                        rollout_record["score"] = float(avg_gap)

                print(f"Final Epoch {epoch + 1} - Evaluation Relative Gap Score: {avg_gap:.2f}%")

                # Log to wandb (if you restore it later)
                wandb.log(rollout_log)

                # 保存gap历史记录和对应的global_step
                eval_gap_history.append(avg_gap)
                eval_gap_steps_history.append(global_step)
                eval_steps_history.append(global_step)
                rollout_gap_records.append(rollout_record)

                if not np.isnan(avg_gap) and avg_gap < best_eval_gap:
                    best_eval_gap = float(avg_gap)
                    best_eval_gap_step = global_step
                    save_path = os.path.join(dir_out, f"best_gap_model_final_epoch_{epoch + 1}")
                    best_gap_model_path = save_path
                    save_peft_model(dt_model, save_path)
                    print(f"[BestGap] 最终epoch gap改善到 {best_eval_gap:.4f}% ，已保存最佳gap模型")

            except Exception as e:
                print(f"Final epoch rollout evaluation failed: {e}")
                # 保存一个默认的gap值，避免绘图时出错
                eval_gap_history.append(0.0)
                eval_gap_steps_history.append(global_step)  # 添加对应的step记录
                eval_steps_history.append(global_step)
                rollout_gap_records.append({
                    "step": int(global_step),
                    "score": float("nan"),
                    "mean_gap_all_instances": float("nan"),
                    "target_return": None,
                    "per_scale": {},
                    "error": str(e),
                })

        # Save periodic checkpoint
        if (global_step % args.save_step == 0) and (global_step > 0):
            save_path = os.path.join(dir_out, f"checkpoint_step_{global_step}")
            # 使用统一的模型保存函数
            save_peft_model(dt_model, save_path)

    # Save final model
    final_save_path = os.path.join(dir_out, "final_model")
    # 直接使用统一的save_peft_model函数保存final model
    save_peft_model(dt_model, final_save_path)

    # Save model configuration
    config = {
        'model_name': args.model_name,
        'state_dim': args.state_dim,
        'n_jobs': args.n_jobs,
        'max_length': args.max_length,
        'max_ep_len': args.max_ep_len,
        'lora_r': args.lora_r,
        'lora_alpha': args.lora_alpha,
        'model_max_jobs': model_max_jobs,
        'model_max_machines': model_max_machines,
        'state_encoder_type': args.state_encoder_type,
        'use_candidate_action_head': args.use_candidate_action_head,
        'state_tensorization_mode': args.state_tensorization_mode,
        'scale_param_mask_conflict_mode': getattr(args, 'scale_param_mask_conflict_mode', 'legacy_global_mean'),
        'ssnc_neighbor_k': getattr(args, 'ssnc_neighbor_k', 3),
    }

    with open(os.path.join(dir_out, 'model_config.json'), 'w') as f:
        json.dump(config, f)

    # Create and save loss and gap visualization
    def save_loss_visualization():
        _do_save_plots(
            dir_out=dir_out,
            train_loss_history=train_loss_history,
            eval_loss_history=eval_loss_history,
            eval_gap_history=eval_gap_history,
            eval_gap_steps_history=eval_gap_steps_history,
            global_steps_history=global_steps_history,
            eval_steps_history=eval_steps_history,
            rollout_gap_records=rollout_gap_records,
        )

    # Generate and save loss visualization
    save_loss_visualization()

    wandb.finish()
    return dt_model


def main():
    """Main function to run the training script."""
    args = parse_args()
    train_decision_transformer(args)


if __name__ == "__main__":
    main()
