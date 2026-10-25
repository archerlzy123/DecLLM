"""Decision Transformer model for JSSP problem, based on Qwen2.5-7B."""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
import numpy as np
import os


class DecisionTransformer(nn.Module):
    """Decision Transformer model that adapts Qwen2.5-7B for JSSP using Causal LM paradigm."""
    
    def __init__(self,
                 model_name="unsloth/Qwen2.5-7B",
                 state_dim=None,
                 n_jobs=5,  # Number of discrete job options
                 n_machines=10,  # Number of machines
                 max_length=1024,
                 max_ep_len=512,
                 hidden_size=4096,
                 use_peft=True,
                 lora_r=64,
                 lora_alpha=128,
                 model_source="huggingface",  # huggingface 或 modelscope
                 load_in_4bit=False,  # 是否使用4bit量化
                 lang_model=None,  # 预初始化的语言模型，用于离线测试
                 dtype="float32",  # 数据类型选项 - 默认使用float32以支持AMP
                 use_action_decoder=True,  # 是否使用JSSPActionDecoder
                 **kwargs):
        """
        Initialize the Decision Transformer.
        
        Args:
            model_name: Name of the pre-trained model to use
            state_dim: Dimension of state embeddings
            act_dim: Dimension of action embeddings
            max_length: Maximum sequence length
            max_ep_len: Maximum episode length
            hidden_size: Hidden size of the transformer
            use_peft: Whether to use PEFT for efficient finetuning
            lora_r: LoRA rank
            lora_alpha: LoRA alpha parameter
            model_source: Source of the model ('huggingface' or 'modelscope')
        """
        super().__init__()
        
        self.model_name = model_name
        self.state_dim = state_dim
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.max_length = max_length
        self.max_ep_len = max_ep_len
        self.initial_hidden_size = hidden_size  # 存储初始hidden_size
        self.use_peft = use_peft
        self.model_source = model_source
        self.load_in_4bit = load_in_4bit  # 存储4bit量化参数
        self.dtype = dtype  # 存储数据类型
        self.state_encoder = None  # 初始化state_encoder为None，注册为子模块
        self.use_action_decoder = use_action_decoder  # 存储是否使用JSSPActionDecoder
        self.use_gradient_checkpointing = bool(kwargs.pop("use_gradient_checkpointing", False))
        self.state_encoder_type = str(kwargs.pop("state_encoder_type", "jssp")).lower().strip()
        self.graph_use_queue_pos = bool(kwargs.pop("graph_use_queue_pos", True))
        self.input_grad_clip = float(kwargs.pop("input_grad_clip", 1.0))
        self.use_scale_embedding = bool(kwargs.pop("use_scale_embedding", False))
        self.scale_alpha_init = float(kwargs.pop("scale_alpha_init", 0.0))
        self.scale_embedding_init_std = float(kwargs.pop("scale_embedding_init_std", 1e-3))
        self.use_candidate_harmony_mask = bool(kwargs.pop("use_candidate_harmony_mask", False))
        self.candidate_harmony_group_by = str(kwargs.pop("candidate_harmony_group_by", "machines") or "machines").lower().strip()
        if self.candidate_harmony_group_by in {"machine", "m"}:
            self.candidate_harmony_group_by = "machines"
        elif self.candidate_harmony_group_by in {"scale", "scales", "size", "sizes", "problem_size", "exact_scale"}:
            self.candidate_harmony_group_by = "scale"
        else:
            self.candidate_harmony_group_by = "machines"
        self.register_buffer("candidate_harmony_mask", None, persistent=True)
        self.use_scale_adapter_expert = bool(kwargs.pop("use_scale_adapter_expert", False))
        self.use_scale_context_adapter = bool(kwargs.pop("use_scale_context_adapter", self.use_scale_adapter_expert))
        self.use_scale_candidate_residual = bool(kwargs.pop("use_scale_candidate_residual", self.use_scale_adapter_expert))
        self.use_scale_adapter_harmony_mask = bool(kwargs.pop("use_scale_adapter_harmony_mask", False))
        self.scale_adapter_group_by = str(kwargs.pop("scale_adapter_group_by", "scale") or "scale").lower().strip()
        if self.scale_adapter_group_by in {"machine", "machines", "m"}:
            self.scale_adapter_group_by = "machines"
        elif self.scale_adapter_group_by in {"scale", "scales", "size", "sizes", "problem_size", "exact_scale"}:
            self.scale_adapter_group_by = "scale"
        else:
            self.scale_adapter_group_by = "scale"
        self.scale_adapter_bottleneck = max(1, int(kwargs.pop("scale_adapter_bottleneck", 256)))
        self.scale_adapter_alpha_init = float(kwargs.pop("scale_adapter_alpha_init", 0.01))
        self.register_buffer("scale_context_harmony_mask", None, persistent=True)
        self.register_buffer("scale_candidate_harmony_mask", None, persistent=True)
        self.use_scale_param_mask = bool(kwargs.pop("use_scale_param_mask", False))
        self.mask_gnn_encoder = bool(kwargs.pop("mask_gnn_encoder", False))
        self.mask_state_projection = bool(kwargs.pop("mask_state_projection", False))
        self.mask_lora = bool(kwargs.pop("mask_lora", False))
        self.mask_candidate_head = bool(kwargs.pop("mask_candidate_head", False))
        self.mask_scale_adapter = bool(kwargs.pop("mask_scale_adapter", False))
        self.scale_param_mask_group_by = str(kwargs.pop("scale_param_mask_group_by", "scale") or "scale").lower().strip()
        if self.scale_param_mask_group_by in {"machine", "machines", "m"}:
            self.scale_param_mask_group_by = "machines"
        elif self.scale_param_mask_group_by in {"scale", "scales", "size", "sizes", "problem_size", "exact_scale"}:
            self.scale_param_mask_group_by = "scale"
        else:
            self.scale_param_mask_group_by = "scale"
        self.scale_param_mask_lora_rank = max(1, int(lora_r))
        self.register_buffer("scale_param_mask_gnn_encoder", None, persistent=True)
        self.register_buffer("scale_param_mask_state_projection", None, persistent=True)
        self.register_buffer("scale_param_mask_lora", None, persistent=True)
        self.register_buffer("scale_param_mask_candidate_head", None, persistent=True)
        self.register_buffer("scale_param_mask_scale_adapter", None, persistent=True)
        self._scale_param_active_group = None
        self._scale_param_lora_hooks_registered = False
        self._scale_param_disable_lora_grad_mask = False
        
        # 使用预初始化的语言模型（用于离线测试）或加载预训练模型
        self.tokenizer = None
        
        if lang_model is not None:
            # 使用预初始化的语言模型
            self.base_model = lang_model
            # 不再强制要求tokenizer
            if hasattr(lang_model, 'tokenizer'):
                self.tokenizer = lang_model.tokenizer
            # 保持tokenizer为None，因为我们不再需要它
            # 确保模型有config属性
            if not hasattr(self.base_model, 'config'):
                # 创建一个简单的config对象
                class MockConfig:
                    def __init__(self):
                        self.hidden_size = hidden_size
                        self.vocab_size = 30
                        self.pad_token_id = 0
                        self.bos_token_id = 1
                        self.eos_token_id = 2
                        self.model_type = "causal_lm"
                self.base_model.config = MockConfig()
            # Ensure base_model has config if it's a wrapped model (e.g., PEFT)
            if hasattr(self.base_model, 'base_model'):
                if not hasattr(self.base_model.base_model, 'config'):
                    self.base_model.base_model.config = self.base_model.config
                
                # Ensure nested base_model has config if it's a wrapped model (e.g., PEFT)
                if hasattr(self.base_model.base_model, 'base_model'):
                    if not hasattr(self.base_model.base_model.base_model, 'config'):
                        self.base_model.base_model.base_model.config = self.base_model.base_model.config
        else:
            # Load pre-trained model and tokenizer from the specified source
            if model_source == "huggingface":
                # 检查是否为本地路径（目录或文件）
                import os
                def is_local_path(path):
                    # 去除路径前后的空格
                    if path is None:
                        return False
                    path = path.strip()
                    
                    # 首先尝试直接检查路径是否存在
                    if os.path.exists(path):
                        return True
                    # 尝试检查绝对路径是否存在
                    abs_path = os.path.abspath(path)
                    if os.path.exists(abs_path):
                        return True
                    # 检查是否以本地路径分隔符或当前/父目录前缀开头
                    # 特别处理Linux绝对路径（以/开头）和Windows绝对路径（如C:\）
                    return (path.startswith('/') or 
                            path.startswith('\\') or 
                            path.startswith('./') or 
                            path.startswith('../') or 
                            ':' in path or  # Windows驱动器号
                            os.sep in path)
                    
                    # 最后的兜底检查：如果路径包含点和斜杠，很可能是本地路径
                    return '.' in path and ('/' in path or '\\' in path)
                
                is_local_model = is_local_path(model_name)
                
                # 根据dtype参数选择合适的torch_dtype
                if self.dtype == "bfloat16":
                    torch_dtype = torch.bfloat16
                elif self.dtype == "float16":
                    torch_dtype = torch.float16
                elif self.dtype == "float32":
                    torch_dtype = torch.float32
                else:
                    torch_dtype = torch.float16  # 默认值
                
                # 构建模型加载参数 - 针对本地模型和Hub模型使用不同策略
                model_kwargs = {
                    "trust_remote_code": True,
                    "torch_dtype": torch_dtype,
                    "low_cpu_mem_usage": True,
                    "local_files_only": True,  # 强制只从本地加载
                }
                
                # 添加4bit量化相关参数（如果需要）
                if self.load_in_4bit:
                    model_kwargs["device_map"] = "auto"
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch_dtype,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                
                # 无论是否为本地模型，都使用本地加载的参数集
                # 这是最安全的方式，避免transformers库将本地路径误判为Hub模型ID
                local_kwargs = {
                    "trust_remote_code": True,
                    "torch_dtype": torch_dtype,
                    "low_cpu_mem_usage": True,
                    "local_files_only": True,  # 强制只从本地加载
                }
                
                # 添加4bit量化相关参数（如果需要）
                if self.load_in_4bit:
                    local_kwargs["device_map"] = "auto"
                    local_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch_dtype,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                
                # 尝试加载模型
                try:
                    # 猴子补丁1：绕过huggingface_hub对repo_id格式的验证
                    import huggingface_hub.utils._validators as validators
                    original_validate_repo_id = validators.validate_repo_id
                    
                    def patched_validate_repo_id(repo_id):
                        # 允许任何格式的repo_id（包括绝对路径）
                        return
                    
                    # 应用猴子补丁1
                    validators.validate_repo_id = patched_validate_repo_id
                    
                    # 猴子补丁2：修复find_adapter_config_file函数，确保它能正确处理本地路径
                    from transformers.utils.peft_utils import find_adapter_config_file as original_find_adapter_config_file
                    import transformers.utils.peft_utils
                    import os
                    
                    def patched_find_adapter_config_file(model_id, *args, **kwargs):
                        # 如果model_id是本地路径，直接返回None，避免尝试查找adapter_config.json
                        if model_id and (os.path.isdir(model_id) or os.path.exists(model_id)):
                            return None
                        return original_find_adapter_config_file(model_id, *args, **kwargs)
                    
                    # 应用猴子补丁2
                    transformers.utils.peft_utils.find_adapter_config_file = patched_find_adapter_config_file

                    # 使用AutoModelForCausalLM加载模型，并确保trust_remote_code=True
                    from transformers import AutoModelForCausalLM
                    self.base_model = AutoModelForCausalLM.from_pretrained(model_name, **local_kwargs)
                    # 将模型的name_or_path设置为空字符串，避免PEFT库将其视为Hugging Face Hub上的仓库ID
                    self.base_model.name_or_path = ""
                    
                    # 只恢复validate_repo_id，保留find_adapter_config_file的猴子补丁直到PEFT设置完成
                    validators.validate_repo_id = original_validate_repo_id
                    
                    # 设置tokenizer为None，因为我们不再使用文本生成
                    self.tokenizer = None
                except Exception as e:
                    print(f"Error loading model: {e}")
                    print("Trying alternative loading method...")
                    
                    # 最后的备选方案：使用最简化的参数加载本地模型
                    last_resort_kwargs = {
                        "trust_remote_code": True,
                        "torch_dtype": torch_dtype,
                        "low_cpu_mem_usage": True,
                        "local_files_only": True,  # 强制只从本地加载
                    }
                    
                    try:
                        # 再次应用猴子补丁1：绕过huggingface_hub对repo_id格式的验证
                        import huggingface_hub.utils._validators as validators
                        original_validate_repo_id = validators.validate_repo_id
                        
                        def patched_validate_repo_id(repo_id):
                            # 允许任何格式的repo_id（包括绝对路径）
                            return
                        
                        validators.validate_repo_id = patched_validate_repo_id
                        
                        # 再次应用猴子补丁2：修复find_adapter_config_file函数
                        from transformers.utils.peft_utils import find_adapter_config_file as original_find_adapter_config_file
                        import transformers.utils.peft_utils
                        import os
                        
                        def patched_find_adapter_config_file(model_id, *args, **kwargs):
                            # 如果model_id是本地路径，直接返回None，避免尝试查找adapter_config.json
                            if model_id and (os.path.isdir(model_id) or os.path.exists(model_id)):
                                return None
                            return original_find_adapter_config_file(model_id, *args, **kwargs)
                        
                        transformers.utils.peft_utils.find_adapter_config_file = patched_find_adapter_config_file

                        # 使用AutoModelForCausalLM加载模型，并确保trust_remote_code=True
                        from transformers import AutoModelForCausalLM
                        self.base_model = AutoModelForCausalLM.from_pretrained(model_name, **last_resort_kwargs)
                        
                        # 只恢复validate_repo_id，保留find_adapter_config_file的猴子补丁直到PEFT设置完成
                        validators.validate_repo_id = original_validate_repo_id
                    except Exception as e2:
                        # 确保无论如何都恢复原始验证函数
                        if 'validators' in locals() and 'original_validate_repo_id' in locals():
                            validators.validate_repo_id = original_validate_repo_id
                        print(f"Alternative method failed: {e2}")
                        raise
                    
                    self.tokenizer = None
            elif model_source == "modelscope":
                # Try to use modelscope library if available
                try:
                    from modelscope import AutoModelForCausalLM as MS_AutoModelForCausalLM, AutoTokenizer as MS_AutoTokenizer
                    print(f"Loading model {model_name} from ModelScope...")
                    # 检查是否为本地路径（目录或文件）
                    is_local_model = os.path.exists(model_name)
                    
                    # 不再需要加载tokenizer
                    self.tokenizer = None
                    
                    # 根据dtype参数选择合适的torch_dtype
                    if self.dtype == "bfloat16":
                        torch_dtype = torch.bfloat16
                    elif self.dtype == "float16":
                        torch_dtype = torch.float16
                    elif self.dtype == "float32":
                        torch_dtype = torch.float32
                    else:
                        torch_dtype = torch.float32  # 默认值
                    
                    # 构建ModelScope模型加载参数
                    ms_kwargs = {
                        "trust_remote_code": True,
                        "dtype": torch_dtype,  # 使用用户指定的dtype
                        "device_map": "auto" if self.load_in_4bit else None,  # 4bit量化时自动分配设备
                        "load_in_4bit": self.load_in_4bit,  # 明确指定是否使用4bit量化
                        "low_cpu_mem_usage": True,
                        "local_files_only": True,  # 关键！强制只从本地加载，防止误判为远程ID
                        "revision": "main"  # 指定模型版本
                    }
                    
                    # 如果用户指定了4bit量化，使用BitsAndBytesConfig进行配置
                    if self.load_in_4bit:
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.bfloat16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4"
                        )
                        ms_kwargs['quantization_config'] = quantization_config
                    
                    # 尝试加载模型，处理可能的config解析问题
                    try:
                        self.base_model = MS_AutoModelForCausalLM.from_pretrained(model_name, **ms_kwargs)
                    except Exception as e:
                        print(f"Error loading ModelScope model: {e}")
                        print("Trying alternative model loading method...")
                        # 保留trust_remote_code参数，这是必须的
                        self.base_model = MS_AutoModelForCausalLM.from_pretrained(model_name, **ms_kwargs)
                except ImportError:
                    print("ModelScope library not found. Please install it with: pip install modelscope")
                    print("Falling back to Hugging Face URL format for ModelScope models...")
                    # Fallback to using ModelScope's Hugging Face compatible URLs
                    # Example: modelscope model "qwen/Qwen2.5-7B" becomes "modelscope/qwen/Qwen2.5-7B"
                    hf_compatible_name = f"modelscope/{model_name}"
                    # Use the model from transformers library explicitly
                    from transformers import AutoModelForCausalLM as HF_AutoModelForCausalLM
                    # 检查是否为本地路径（目录或文件）
                    is_local_model = os.path.exists(hf_compatible_name)
                    
                    # 根据dtype参数选择合适的torch_dtype
                    if self.dtype == "bfloat16":
                        torch_dtype = torch.bfloat16
                    elif self.dtype == "float16":
                        torch_dtype = torch.float16
                    elif self.dtype == "float32":
                        torch_dtype = torch.float32
                    else:
                        torch_dtype = torch.float32  # 默认值
                    
                    # 构建HuggingFace兼容模式模型加载参数
                    hf_fallback_kwargs = {
                        "trust_remote_code": True,
                        "dtype": torch_dtype,  # 使用用户指定的dtype
                        "device_map": "auto" if self.load_in_4bit else None,  # 4bit量化时自动分配设备
                        "load_in_4bit": self.load_in_4bit,  # 明确指定是否使用4bit量化
                        "low_cpu_mem_usage": True,
                        "local_files_only": True,  # 关键！强制只从本地加载，防止误判为远程ID
                        "revision": "main"  # 指定模型版本
                    }
                    # 设置tokenizer为None，因为我们不再使用文本生成
                    self.tokenizer = None
                    
                    # 如果用户指定了4bit量化，使用BitsAndBytesConfig进行配置
                    if self.load_in_4bit:
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.bfloat16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4"
                        )
                        hf_fallback_kwargs['quantization_config'] = quantization_config
                    
                    # 尝试加载模型，处理可能的config解析问题
                    try:
                        self.base_model = HF_AutoModelForCausalLM.from_pretrained(hf_compatible_name, **hf_fallback_kwargs)
                    except Exception as e:
                        print(f"Error loading fallback model: {e}")
                        print("Trying alternative model loading method...")
                        # 保留trust_remote_code参数，这是必须的
                        self.base_model = HF_AutoModelForCausalLM.from_pretrained(hf_compatible_name, **hf_fallback_kwargs)
            else:
                raise ValueError(f"Unknown model source: {model_source}. Use 'huggingface' or 'modelscope'")
        
        # 从加载的模型动态获取hidden_size，避免硬编码导致的维度不匹配
        # 不同的模型可能有不同的hidden_size属性名
        if hasattr(self.base_model.config, 'hidden_size'):
            model_hidden_size = self.base_model.config.hidden_size
        elif hasattr(self.base_model.config, 'n_embd'):
            model_hidden_size = self.base_model.config.n_embd
        else:
            # 如果无法从配置中获取，则使用默认值
            model_hidden_size = hidden_size
            print(f"Warning: Could not determine model hidden_size, using default: {hidden_size}")
        
        # 更新实例的hidden_size
        self.hidden_size = model_hidden_size
        print(f"Using hidden_size: {self.hidden_size}")
        
        # 定义必要的投影层和嵌入层（在获取实际hidden_size后）
        # 使用正确的嵌入层结构，确保输出维度都是self.hidden_size
        self.rtg_proj = nn.Linear(1, self.hidden_size)  # 将returns_to_go投影到hidden_size
        nn.init.normal_(self.rtg_proj.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.rtg_proj.bias)
        # 仅当state_dim不为None时创建state_proj层
        self.state_proj = None
        if self.state_dim is not None:
            self.state_proj = nn.Linear(self.state_dim, self.hidden_size)  # 将状态投影到hidden_size
        # 修改为作业ID的嵌入层，添加padding_idx参数处理占位符
        self.action_proj = nn.Embedding(self.n_jobs + 1, self.hidden_size, padding_idx=self.n_jobs)  # 将作业ID嵌入到hidden_size，padding_idx使用超出有效范围的值
        nn.init.normal_(self.action_proj.weight, mean=0.0, std=0.02)
        if self.action_proj.padding_idx is not None:
            with torch.no_grad():
                self.action_proj.weight[self.action_proj.padding_idx].zero_()
        # 添加dropout层（降低到0.05：原0.1对短序列/小batch过于激进，导致训练波动大）
        self.embed_dropout = nn.Dropout(0.05)
        self.timestep_embedding = nn.Embedding(self.max_ep_len, self.hidden_size)  # 时间步嵌入
        nn.init.normal_(self.timestep_embedding.weight, mean=0.0, std=0.02)
        self.type_embedding = nn.Embedding(4, self.hidden_size)  # 类型嵌入：0=returns_to_go, 1=states, 2=actions, 3=timesteps
        nn.init.normal_(self.type_embedding.weight, mean=0.0, std=0.02)
        self.scale_embedding = None
        self.scale_alpha = None
        if self.use_scale_embedding:
            # Scale-conditioned state prompt.  The +1 keeps index 0 available for
            # unknown/missing sizes and lets scale_id = jobs * (M+1) + machines.
            self.scale_embedding = nn.Embedding(
                (int(self.n_jobs) + 1) * (int(self.n_machines) + 1),
                self.hidden_size,
            )
            nn.init.normal_(self.scale_embedding.weight, mean=0.0, std=self.scale_embedding_init_std)
            self.scale_alpha = nn.Parameter(torch.tensor(self.scale_alpha_init, dtype=torch.float32))
        # 在送入 Qwen 前先做一次无参数 LayerNorm，再统一缩放，避免自定义 embedding 分布过于激进。
        self.input_fuse_norm = nn.LayerNorm(self.hidden_size, elementwise_affine=False)
        self.input_embed_scale = 0.1
        # 修改动作输出头，只输出作业ID
        self.action_head = nn.Linear(self.hidden_size, self.n_jobs)  # 动作输出头，输出作业ID
        nn.init.normal_(self.action_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.action_head.bias)
        self.use_candidate_action_head = (
            self.state_encoder_type == "graph"
            and bool(kwargs.pop("use_candidate_action_head", False))
        )
        self.candidate_job_proj = None
        self.candidate_context_proj = None
        self.candidate_fuse_norm = None
        self.candidate_action_head = None
        self.scale_context_adapter_norm = None
        self.scale_context_adapter_down = None
        self.scale_context_adapter_up = None
        self.scale_context_adapter_alpha = None
        self.scale_candidate_residual_norm = None
        self.scale_candidate_residual_down = None
        self.scale_candidate_residual_up = None
        self.scale_candidate_residual_alpha = None
        value_hidden_dim = min(512, max(128, self.hidden_size // 4))
        self.value_head = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, value_hidden_dim),
            nn.GELU(),
            nn.Linear(value_hidden_dim, 1),
        )
        nn.init.normal_(self.value_head[1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.value_head[1].bias)
        nn.init.normal_(self.value_head[3].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.value_head[3].bias)

        scale_adapter_rows = int(self.n_machines) + 1
        if self.scale_adapter_group_by == "scale":
            scale_adapter_rows = (int(self.n_jobs) + 1) * (int(self.n_machines) + 1)
        scale_param_rows = int(self.n_machines) + 1
        if self.scale_param_mask_group_by == "scale":
            scale_param_rows = (int(self.n_jobs) + 1) * (int(self.n_machines) + 1)
        adapter_dim = min(int(self.scale_adapter_bottleneck), int(self.hidden_size))
        if self.use_scale_param_mask:
            if self.mask_state_projection:
                self.scale_param_mask_state_projection = torch.ones(
                    scale_param_rows,
                    self.hidden_size,
                    dtype=torch.float32,
                )
            if self.mask_lora:
                self.scale_param_mask_lora = torch.ones(
                    scale_param_rows,
                    self.scale_param_mask_lora_rank,
                    dtype=torch.float32,
                )
            if self.mask_scale_adapter and (self.use_scale_context_adapter or self.use_scale_candidate_residual):
                self.scale_param_mask_scale_adapter = torch.ones(
                    scale_param_rows,
                    adapter_dim,
                    dtype=torch.float32,
                )
        if self.use_scale_context_adapter:
            self.scale_context_adapter_norm = nn.LayerNorm(self.hidden_size)
            self.scale_context_adapter_down = nn.Linear(self.hidden_size, adapter_dim)
            self.scale_context_adapter_up = nn.Linear(adapter_dim, self.hidden_size)
            self.scale_context_adapter_alpha = nn.Parameter(
                torch.full((scale_adapter_rows, 1), float(self.scale_adapter_alpha_init), dtype=torch.float32)
            )
            nn.init.normal_(self.scale_context_adapter_down.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.scale_context_adapter_down.bias)
            nn.init.normal_(self.scale_context_adapter_up.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.scale_context_adapter_up.bias)
            if self.use_scale_adapter_harmony_mask:
                self.scale_context_harmony_mask = torch.ones(
                    scale_adapter_rows,
                    adapter_dim,
                    dtype=torch.float32,
                )

        # 删除action token映射，因为我们不再使用文本生成
        self.action_tokens = None
        self.reverse_action_tokens = None

        # Action decoder configuration - 统一使用作业级别动作空间，禁用action_decoder
        self.use_action_decoder = False
        self.action_decoder = None  # 不再使用作业-机器级别解码器

        # === 修改点：在这里直接创建 JSSPStateEncoder 作为子模块 ===
        if self.use_action_decoder:
            # 如果使用 action decoder (旧模式)
            if self.state_dim is not None:
                self.state_proj = nn.Linear(self.state_dim, self.hidden_size)
            self.state_encoder = None
        else:
            # 如果不使用 action decoder (新模式，使用 JSSPStateEncoder)
            state_dims = {
                'n_jobs': self.n_jobs,
                'n_machines': self.n_machines
            }
            encoder_dim = self.state_dim if self.state_dim is not None else min(256, self.hidden_size)
            if self.state_proj is None or self.state_proj.in_features != encoder_dim:
                self.state_proj = nn.Linear(encoder_dim, self.hidden_size)
                # 用小权重初始化：让GNN输出在训练初期对Qwen影响极小，避免破坏预训练权重
                nn.init.normal_(self.state_proj.weight, mean=0.0, std=0.01)
                nn.init.zeros_(self.state_proj.bias)
            # LayerNorm 防止 GNN 输出（随机初始化）被预训练权重量级淹没
            self.state_proj_norm = nn.LayerNorm(self.hidden_size)
            if self.state_encoder_type == "graph":
                self.state_encoder = GraphJSSPStateEncoder(
                    state_dims,
                    hidden_size=encoder_dim,
                    use_queue_position=self.graph_use_queue_pos,
                )
                if self.use_scale_param_mask and self.mask_gnn_encoder:
                    self.scale_param_mask_gnn_encoder = torch.ones(
                        scale_param_rows,
                        encoder_dim,
                        dtype=torch.float32,
                    )
                if self.use_candidate_action_head:
                    self.candidate_job_proj = nn.Linear(encoder_dim, self.hidden_size)
                    self.candidate_context_proj = nn.Linear(self.hidden_size, self.hidden_size)
                    self.candidate_fuse_norm = nn.LayerNorm(self.hidden_size)
                    self.candidate_action_head = nn.Linear(self.hidden_size, 1)
                    nn.init.normal_(self.candidate_job_proj.weight, mean=0.0, std=0.01)
                    nn.init.zeros_(self.candidate_job_proj.bias)
                    nn.init.normal_(self.candidate_context_proj.weight, mean=0.0, std=0.01)
                    nn.init.zeros_(self.candidate_context_proj.bias)
                    nn.init.normal_(self.candidate_action_head.weight, mean=0.0, std=0.01)
                    nn.init.zeros_(self.candidate_action_head.bias)
                    if self.use_scale_candidate_residual:
                        self.scale_candidate_residual_norm = nn.LayerNorm(self.hidden_size)
                        self.scale_candidate_residual_down = nn.Linear(self.hidden_size, adapter_dim)
                        self.scale_candidate_residual_up = nn.Linear(adapter_dim, 1)
                        self.scale_candidate_residual_alpha = nn.Parameter(
                            torch.full((scale_adapter_rows, 1), float(self.scale_adapter_alpha_init), dtype=torch.float32)
                        )
                        nn.init.normal_(self.scale_candidate_residual_down.weight, mean=0.0, std=0.01)
                        nn.init.zeros_(self.scale_candidate_residual_down.bias)
                        nn.init.normal_(self.scale_candidate_residual_up.weight, mean=0.0, std=0.01)
                        nn.init.zeros_(self.scale_candidate_residual_up.bias)
                        if self.use_scale_adapter_harmony_mask:
                            self.scale_candidate_harmony_mask = torch.ones(
                                scale_adapter_rows,
                                adapter_dim,
                                dtype=torch.float32,
                            )
                    if self.use_candidate_harmony_mask:
                        mask_rows = int(self.n_machines) + 1
                        if self.candidate_harmony_group_by == "scale":
                            mask_rows = (int(self.n_jobs) + 1) * (int(self.n_machines) + 1)
                        self.candidate_harmony_mask = torch.ones(
                            mask_rows,
                            self.hidden_size,
                            dtype=torch.float32,
                        )
                    if self.use_scale_param_mask and self.mask_candidate_head:
                        self.scale_param_mask_candidate_head = torch.ones(
                            scale_param_rows,
                            self.hidden_size,
                            dtype=torch.float32,
                        )
                print(f"✅ GraphJSSPStateEncoder已创建为子模块，encoder_dim={encoder_dim}, proj_to_hidden_size={self.hidden_size}")
            else:
                self.state_encoder = JSSPStateEncoder(state_dims, hidden_size=encoder_dim)
                print(f"✅ JSSPStateEncoder已创建为子模块，encoder_dim={encoder_dim}, proj_to_hidden_size={self.hidden_size}")

        # If using PEFT for efficient finetuning (only for loaded models, not pre-initialized ones)
        if use_peft and lang_model is None:
            from peft import get_peft_model, LoraConfig

            # 对于Qwen2模型，完整的LoRA目标模块应包括attention和MLP层
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"]

            # 设置base_model_name_or_path为空字符串，避免PEFT库尝试从Hugging Face Hub下载adapter_config.json
            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                base_model_name_or_path=""  # 显式设置为空字符串，避免PEFT库尝试从Hugging Face Hub下载
            )

            # 确保在使用get_peft_model之前find_adapter_config_file的猴子补丁仍然有效
            # 保存当前的find_adapter_config_file函数
            try:
                from transformers.utils.peft_utils import find_adapter_config_file as current_find_adapter_config_file
                import transformers.utils.peft_utils
                import os

                # 如果当前不是我们的补丁函数，重新应用补丁
                if current_find_adapter_config_file.__name__ != "patched_find_adapter_config_file":
                    from transformers.utils.peft_utils import find_adapter_config_file as original_find_adapter_config_file

                    def patched_find_adapter_config_file(model_id, *args, **kwargs):
                        # 如果model_id是本地路径，直接返回None，避免尝试查找adapter_config.json
                        if model_id and (os.path.isdir(model_id) or os.path.exists(model_id)):
                            return None
                        return original_find_adapter_config_file(model_id, *args, **kwargs)

                    # 应用猴子补丁
                    transformers.utils.peft_utils.find_adapter_config_file = patched_find_adapter_config_file

                    # 使用PEFT库创建模型
                    self.base_model = get_peft_model(self.base_model, peft_config)

                    # 恢复原始函数
                    transformers.utils.peft_utils.find_adapter_config_file = original_find_adapter_config_file
                else:
                    # 如果当前已经是补丁函数，直接使用
                    self.base_model = get_peft_model(self.base_model, peft_config)

                    # 恢复原始函数
                    from transformers.utils.peft_utils import find_adapter_config_file as original_find_adapter_config_file
                    transformers.utils.peft_utils.find_adapter_config_file = original_find_adapter_config_file
            except Exception as e:
                print(f"Warning: Error in PEFT setup: {e}")
                # 即使出现错误，也要尝试创建模型
                self.base_model = get_peft_model(self.base_model, peft_config)

            self.base_model.print_trainable_parameters()
            self.register_scale_param_lora_hooks()

        if self.use_gradient_checkpointing:
            try:
                if hasattr(self.base_model, "gradient_checkpointing_enable"):
                    self.base_model.gradient_checkpointing_enable()
                elif hasattr(self.base_model, "model") and hasattr(self.base_model.model, "gradient_checkpointing_enable"):
                    self.base_model.model.gradient_checkpointing_enable()
            except Exception as e:
                print(f"Warning: Failed to enable gradient checkpointing: {e}")

            for m in [self.base_model, getattr(self.base_model, "base_model", None), getattr(getattr(self.base_model, "base_model", None), "model", None)]:
                if m is not None and hasattr(m, "config") and m.config is not None and hasattr(m.config, "use_cache"):
                    m.config.use_cache = False

        # 自定义可训练模块固定保留在 float32，避免在 bf16/fp16 下训练初期数值不稳定。
        custom_module_dtype = torch.float32
        self.rtg_proj = self.rtg_proj.to(dtype=custom_module_dtype)
        if hasattr(self, 'state_proj') and self.state_proj is not None:
            self.state_proj = self.state_proj.to(dtype=custom_module_dtype)
        if hasattr(self, 'state_proj_norm') and self.state_proj_norm is not None:
            self.state_proj_norm = self.state_proj_norm.to(dtype=custom_module_dtype)
        self.action_proj = self.action_proj.to(dtype=custom_module_dtype)
        self.embed_dropout = self.embed_dropout.to(dtype=custom_module_dtype)
        self.timestep_embedding = self.timestep_embedding.to(dtype=custom_module_dtype)
        self.type_embedding = self.type_embedding.to(dtype=custom_module_dtype)
        if self.scale_embedding is not None:
            self.scale_embedding = self.scale_embedding.to(dtype=custom_module_dtype)
        if self.scale_alpha is not None:
            self.scale_alpha.data = self.scale_alpha.data.to(dtype=custom_module_dtype)
        self.input_fuse_norm = self.input_fuse_norm.to(dtype=custom_module_dtype)
        self.action_head = self.action_head.to(dtype=custom_module_dtype)
        self.value_head = self.value_head.to(dtype=custom_module_dtype)
        if self.candidate_job_proj is not None:
            self.candidate_job_proj = self.candidate_job_proj.to(dtype=custom_module_dtype)
        if self.candidate_context_proj is not None:
            self.candidate_context_proj = self.candidate_context_proj.to(dtype=custom_module_dtype)
        if self.candidate_fuse_norm is not None:
            self.candidate_fuse_norm = self.candidate_fuse_norm.to(dtype=custom_module_dtype)
        if self.candidate_action_head is not None:
            self.candidate_action_head = self.candidate_action_head.to(dtype=custom_module_dtype)
        if self.scale_context_adapter_norm is not None:
            self.scale_context_adapter_norm = self.scale_context_adapter_norm.to(dtype=custom_module_dtype)
        if self.scale_context_adapter_down is not None:
            self.scale_context_adapter_down = self.scale_context_adapter_down.to(dtype=custom_module_dtype)
        if self.scale_context_adapter_up is not None:
            self.scale_context_adapter_up = self.scale_context_adapter_up.to(dtype=custom_module_dtype)
        if self.scale_context_adapter_alpha is not None:
            self.scale_context_adapter_alpha.data = self.scale_context_adapter_alpha.data.to(dtype=custom_module_dtype)
        if self.scale_candidate_residual_norm is not None:
            self.scale_candidate_residual_norm = self.scale_candidate_residual_norm.to(dtype=custom_module_dtype)
        if self.scale_candidate_residual_down is not None:
            self.scale_candidate_residual_down = self.scale_candidate_residual_down.to(dtype=custom_module_dtype)
        if self.scale_candidate_residual_up is not None:
            self.scale_candidate_residual_up = self.scale_candidate_residual_up.to(dtype=custom_module_dtype)
        if self.scale_candidate_residual_alpha is not None:
            self.scale_candidate_residual_alpha.data = self.scale_candidate_residual_alpha.data.to(dtype=custom_module_dtype)
        if hasattr(self, "state_encoder") and self.state_encoder is not None:
            self.state_encoder = self.state_encoder.to(dtype=custom_module_dtype)

    def _attach_safe_grad_hook(self, tensor):
        """Sanitize gradients flowing back from the frozen backbone into custom frontend modules."""
        if not isinstance(tensor, torch.Tensor) or not tensor.requires_grad:
            return tensor

        clip_value = float(getattr(self, "input_grad_clip", 1.0) or 0.0)

        def _sanitize_grad(grad):
            if grad is None:
                return grad
            grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
            if clip_value > 0:
                grad = torch.clamp(grad, min=-clip_value, max=clip_value)
            return grad

        tensor.register_hook(_sanitize_grad)
        return tensor

    def register_scale_param_lora_hooks(self):
        """Apply scale-specific rank masks to LoRA gradients during inner-loop training.

        LoRA modules expose rank channels as rows of lora_A and columns of lora_B.
        The hook keeps the forward pass unchanged, but prevents closed rank channels
        from being updated for the active scale group.
        """
        if (
            not self.use_scale_param_mask
            or not self.mask_lora
            or self._scale_param_lora_hooks_registered
            or self.scale_param_mask_lora is None
            or not torch.is_tensor(self.scale_param_mask_lora)
            or not hasattr(self, "base_model")
        ):
            return 0

        def _make_hook(param_name):
            def _hook(grad):
                if grad is None or bool(getattr(self, "_scale_param_disable_lora_grad_mask", False)):
                    return grad
                mask_table = getattr(self, "scale_param_mask_lora", None)
                group = getattr(self, "_scale_param_active_group", None)
                if not torch.is_tensor(mask_table) or group is None:
                    return grad
                group = int(group)
                if group < 0 or group >= int(mask_table.shape[0]):
                    return grad
                mask = mask_table[group].to(device=grad.device, dtype=grad.dtype)
                if "lora_A" in param_name and grad.dim() >= 2 and grad.shape[0] == mask.numel():
                    view_shape = [mask.numel()] + [1] * (grad.dim() - 1)
                    return grad * mask.reshape(*view_shape)
                if "lora_B" in param_name and grad.dim() >= 2 and grad.shape[1] == mask.numel():
                    view_shape = [1, mask.numel()] + [1] * (grad.dim() - 2)
                    return grad * mask.reshape(*view_shape)
                return grad

            return _hook

        count = 0
        for name, param in self.base_model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                param.register_hook(_make_hook(name))
                count += 1
        self._scale_param_lora_hooks_registered = count > 0
        if count > 0:
            print(f"[ScaleParamMask] Registered LoRA rank-gradient hooks: {count}")
        return count

    def _extract_problem_size_tensor(self, states, name, default_value, batch_size, seq_len, device):
        value = None
        if isinstance(states, dict):
            value = states.get(name, None)
        if value is None:
            return torch.full((batch_size, seq_len), float(default_value), device=device)
        if not torch.is_tensor(value):
            value = torch.as_tensor(value, dtype=torch.float32, device=device)
        else:
            value = value.to(device=device)
        if value.dim() == 0:
            value = value.reshape(1, 1).expand(batch_size, seq_len)
        elif value.dim() == 1:
            if value.numel() == batch_size:
                value = value.reshape(batch_size, 1).expand(batch_size, seq_len)
            elif value.numel() >= seq_len:
                value = value[:seq_len].reshape(1, seq_len).expand(batch_size, seq_len)
            else:
                value = value.reshape(1, 1).expand(batch_size, seq_len)
        elif value.dim() == 2:
            if value.shape[0] == batch_size:
                if value.shape[1] >= seq_len:
                    value = value[:, :seq_len]
                elif value.shape[1] == 1:
                    value = value.expand(batch_size, seq_len)
                else:
                    pad = value[:, -1:].expand(batch_size, seq_len - value.shape[1])
                    value = torch.cat([value, pad], dim=1)
            elif value.shape[0] == 1 and value.shape[1] >= seq_len:
                value = value[:, :seq_len].expand(batch_size, seq_len)
            else:
                value = value.reshape(1, 1).expand(batch_size, seq_len)
        else:
            if value.shape[0] == batch_size and value.shape[1] >= seq_len:
                value = value[:, :seq_len, ...].reshape(batch_size, seq_len, -1)[..., 0]
            else:
                value = value.reshape(batch_size, -1, value.shape[-1])
                if value.shape[1] >= seq_len:
                    value = value[:, :seq_len, 0]
                else:
                    value = value[:, :1, 0].expand(batch_size, seq_len)
        return value.to(dtype=torch.float32)

    def _get_scale_condition_embedding(self, states, batch_size, seq_len, device, dtype):
        """Return a scale embedding aligned with state tokens, or None when disabled.

        This is intentionally added to the already-projected state token so old
        checkpoints remain usable: scale_alpha defaults to 0, making the initial
        behavior identical to the previous model.
        """
        if not self.use_scale_embedding or self.scale_embedding is None or self.scale_alpha is None:
            return None
        jobs = self._extract_problem_size_tensor(
            states, "problem_n_jobs", self.n_jobs, batch_size, seq_len, device
        ).round().long().clamp(0, int(self.n_jobs))
        machines = self._extract_problem_size_tensor(
            states, "problem_n_machines", self.n_machines, batch_size, seq_len, device
        ).round().long().clamp(0, int(self.n_machines))
        scale_ids = jobs * (int(self.n_machines) + 1) + machines
        scale_emb = self.scale_embedding(scale_ids)
        scale = self.scale_alpha.to(device=device, dtype=scale_emb.dtype)
        return (scale * scale_emb).to(device=device, dtype=dtype)

    def _get_scale_param_group_ids(self, states, batch_size, seq_len, device, max_group=None):
        if self.scale_param_mask_group_by == "machines":
            group_ids = self._extract_problem_size_tensor(
                states, "problem_n_machines", self.n_machines, batch_size, seq_len, device
            ).round().long()
        else:
            jobs = self._extract_problem_size_tensor(
                states, "problem_n_jobs", self.n_jobs, batch_size, seq_len, device
            ).round().long().clamp(0, int(self.n_jobs))
            machines = self._extract_problem_size_tensor(
                states, "problem_n_machines", self.n_machines, batch_size, seq_len, device
            ).round().long().clamp(0, int(self.n_machines))
            group_ids = jobs * (int(self.n_machines) + 1) + machines
        if max_group is not None:
            group_ids = group_ids.clamp(0, int(max_group))
        return group_ids

    def _set_scale_param_active_group(self, states, batch_size, seq_len, device):
        self._scale_param_active_group = None
        if not self.use_scale_param_mask:
            return
        max_group = None
        for table in [
            self.scale_param_mask_gnn_encoder,
            self.scale_param_mask_state_projection,
            self.scale_param_mask_lora,
            self.scale_param_mask_candidate_head,
            self.scale_param_mask_scale_adapter,
        ]:
            if torch.is_tensor(table) and table.numel() > 0:
                max_group = int(table.shape[0] - 1)
                break
        if max_group is None:
            return
        try:
            group_ids = self._get_scale_param_group_ids(
                states,
                batch_size=batch_size,
                seq_len=seq_len,
                device=device,
                max_group=max_group,
            ).reshape(-1)
            if group_ids.numel() > 0:
                self._scale_param_active_group = int(torch.mode(group_ids).values.item())
        except Exception:
            self._scale_param_active_group = None

    def _get_scale_param_mask(self, mask_table, states, batch_size, seq_len, device, dtype):
        if (
            not self.use_scale_param_mask
            or mask_table is None
            or not torch.is_tensor(mask_table)
            or mask_table.numel() == 0
        ):
            return None
        group_ids = self._get_scale_param_group_ids(
            states,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            max_group=int(mask_table.shape[0] - 1),
        )
        return mask_table.to(device=device, dtype=dtype)[group_ids]

    def _apply_scale_param_mask(self, tensor, mask_table, states, batch_size, seq_len, device, per_job=False):
        if tensor is None:
            return tensor
        mask = self._get_scale_param_mask(
            mask_table,
            states,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            dtype=tensor.dtype,
        )
        if mask is None:
            return tensor
        if per_job:
            mask = mask.unsqueeze(2)
        return tensor * mask

    def _get_candidate_harmony_mask(self, states, batch_size, seq_len, device, dtype):
        """Return a per-position candidate-channel mask for the configured problem grouping."""
        if (
            not self.use_candidate_harmony_mask
            or self.candidate_harmony_mask is None
            or not torch.is_tensor(self.candidate_harmony_mask)
        ):
            return None
        mask_table = self.candidate_harmony_mask
        if mask_table.numel() == 0:
            return None
        max_group = int(mask_table.shape[0] - 1)
        if self.candidate_harmony_group_by == "scale":
            jobs = self._extract_problem_size_tensor(
                states, "problem_n_jobs", self.n_jobs, batch_size, seq_len, device
            ).round().long().clamp(0, int(self.n_jobs))
            machines = self._extract_problem_size_tensor(
                states, "problem_n_machines", self.n_machines, batch_size, seq_len, device
            ).round().long().clamp(0, int(self.n_machines))
            group_ids = jobs * (int(self.n_machines) + 1) + machines
            group_ids = group_ids.clamp(0, max_group)
        else:
            group_ids = self._extract_problem_size_tensor(
                states, "problem_n_machines", self.n_machines, batch_size, seq_len, device
            ).round().long().clamp(0, max_group)
        mask = mask_table.to(device=device, dtype=dtype)[group_ids]
        return mask.unsqueeze(2)

    def _get_scale_adapter_group_ids(self, states, batch_size, seq_len, device, max_group=None):
        if self.scale_adapter_group_by == "scale":
            jobs = self._extract_problem_size_tensor(
                states, "problem_n_jobs", self.n_jobs, batch_size, seq_len, device
            ).round().long().clamp(0, int(self.n_jobs))
            machines = self._extract_problem_size_tensor(
                states, "problem_n_machines", self.n_machines, batch_size, seq_len, device
            ).round().long().clamp(0, int(self.n_machines))
            group_ids = jobs * (int(self.n_machines) + 1) + machines
        else:
            group_ids = self._extract_problem_size_tensor(
                states, "problem_n_machines", self.n_machines, batch_size, seq_len, device
            ).round().long()
        if max_group is not None:
            group_ids = group_ids.clamp(0, int(max_group))
        return group_ids

    def _get_scale_adapter_mask(self, mask_table, states, batch_size, seq_len, device, dtype):
        if mask_table is None or not torch.is_tensor(mask_table) or mask_table.numel() == 0:
            return None
        group_ids = self._get_scale_adapter_group_ids(
            states,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            max_group=int(mask_table.shape[0] - 1),
        )
        return mask_table.to(device=device, dtype=dtype)[group_ids]

    def _get_scale_adapter_alpha(self, alpha_table, states, batch_size, seq_len, device, dtype):
        if alpha_table is None or not torch.is_tensor(alpha_table) or alpha_table.numel() == 0:
            return None
        group_ids = self._get_scale_adapter_group_ids(
            states,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            max_group=int(alpha_table.shape[0] - 1),
        )
        return alpha_table.to(device=device, dtype=dtype)[group_ids]

    def _scale_context_adapter_delta(self, context_states, states, device):
        if (
            not self.use_scale_context_adapter
            or self.scale_context_adapter_norm is None
            or self.scale_context_adapter_down is None
            or self.scale_context_adapter_up is None
            or self.scale_context_adapter_alpha is None
        ):
            return None
        batch_size, seq_len = context_states.shape[:2]
        adapter_dtype = self.scale_context_adapter_down.weight.dtype
        x = context_states.to(dtype=adapter_dtype)
        hidden = self.scale_context_adapter_norm(x)
        hidden = self.scale_context_adapter_down(hidden)
        mask = self._get_scale_adapter_mask(
            self.scale_context_harmony_mask,
            states,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            dtype=hidden.dtype,
        )
        if mask is not None:
            hidden = hidden * mask
        param_mask = self._get_scale_param_mask(
            self.scale_param_mask_scale_adapter,
            states,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            dtype=hidden.dtype,
        )
        if param_mask is not None:
            hidden = hidden * param_mask
        hidden = torch.nn.functional.gelu(hidden)
        delta = self.scale_context_adapter_up(hidden)
        alpha = self._get_scale_adapter_alpha(
            self.scale_context_adapter_alpha,
            states,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            dtype=delta.dtype,
        )
        if alpha is not None:
            delta = delta * alpha
        return delta

    def _scale_candidate_residual_logits(self, fused, states, device):
        if (
            not self.use_scale_candidate_residual
            or self.scale_candidate_residual_norm is None
            or self.scale_candidate_residual_down is None
            or self.scale_candidate_residual_up is None
            or self.scale_candidate_residual_alpha is None
        ):
            return None
        batch_size, seq_len = fused.shape[:2]
        residual_dtype = self.scale_candidate_residual_down.weight.dtype
        x = fused.to(dtype=residual_dtype)
        hidden = self.scale_candidate_residual_norm(x)
        hidden = self.scale_candidate_residual_down(hidden)
        mask = self._get_scale_adapter_mask(
            self.scale_candidate_harmony_mask,
            states,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            dtype=hidden.dtype,
        )
        if mask is not None:
            hidden = hidden * mask.unsqueeze(2)
        param_mask = self._get_scale_param_mask(
            self.scale_param_mask_scale_adapter,
            states,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            dtype=hidden.dtype,
        )
        if param_mask is not None:
            hidden = hidden * param_mask.unsqueeze(2)
        hidden = torch.nn.functional.gelu(hidden)
        residual = self.scale_candidate_residual_up(hidden).squeeze(-1)
        alpha = self._get_scale_adapter_alpha(
            self.scale_candidate_residual_alpha,
            states,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            dtype=residual.dtype,
        )
        if alpha is not None:
            residual = residual * alpha.squeeze(-1).unsqueeze(-1)
        return residual

    def forward(self, states, returns_to_go, timesteps=None, actions=None, attention_mask=None, action_mask=None, env=None):
        """
        Forward pass of the Decision Transformer in standard DT architecture.

        Standard DT Prediction Target: Given [R_t, S_t, t_t, A_{t-1}] predict A_t.
        We place the timestep token before the action token so the hidden state used
        for action prediction can attend to the current timestep as well.

        Args:
            states: Batch of states (batch_size, seq_length, state_dim) or dict of tensors
                - Input: [S_1, S_2, ..., S_T] or dict with state components
                - Used to predict actions [A_1, A_2, ..., A_T]
            returns_to_go: Batch of returns-to-go (batch_size, seq_length, 1)
                - Input: [R_1, R_2, ..., R_T]
                - Used to predict actions [A_1, A_2, ..., A_T]
            timesteps: Batch of timesteps (batch_size, seq_length)
                - Input: [t_1, t_2, ..., t_T]
                - Used for positional encoding and padding detection
            actions: Batch of previous-action tokens, length seq_length or seq_length-1, or None
                    - Training: Prefer aligned previous actions [dummy, A_0, ..., A_{T-2}]
                    - Inference: Use the same aligned history, or length seq_length-1 to let the model prepend dummy
            attention_mask: Custom attention mask (batch_size, seq_length) or None
                    - If None, uses timesteps != 0 as padding mask
            action_mask: Action mask tensor (batch_size, seq_length, n_jobs) or (n_jobs,)
                    - Used for masking invalid actions during training and inference
                    - Priority over env-based masking
            env: JSSP environment object (optional)
                    - Used for action masking during inference if action_mask is not provided
                    - Must have get_action_mask() method

        Returns:
            Model outputs containing action logits (batch_size, seq_length, n_jobs)
            - Output logits at position t correspond to prediction for A_t given
              [R_t, S_t, t_t, A_{t-1}]
        """
        # Get batch_size and seq_length from states (handle both tensor and dict cases)
        if isinstance(states, dict):
            # Get shape from the first value in the dictionary
            first_key = next(iter(states.keys()))
            batch_size, seq_length, _ = states[first_key].shape
        else:
            batch_size, seq_length, _ = states.shape
        device = next(self.base_model.parameters()).device

        # 确保timesteps是Long类型（整数索引）
        if timesteps is not None and timesteps.dtype != torch.long:
            timesteps = timesteps.long()

        # 确保actions是Long类型（整数索引）
        if actions is not None and actions.dtype != torch.long:
            actions = actions.long()

        # Ensure all inputs are on the correct device
        returns_to_go = returns_to_go.to(device)
        # Handle states device move (supports both tensor and dict)
        if isinstance(states, dict):
            states = {k: v.to(device) for k, v in states.items()}
        else:
            states = states.to(device)
        if timesteps is not None:
            timesteps = timesteps.to(device)
        if actions is not None:
            actions = actions.to(device)
        self._set_scale_param_active_group(
            states,
            batch_size=batch_size,
            seq_len=seq_length,
            device=device,
        )

        # Encode states if state encoder is available
        debug_prints = bool(getattr(self, "debug_prints", False))
        if debug_prints:
            print(f"[DEBUG] DecisionTransformer.forward: hasattr(state_encoder)={hasattr(self, 'state_encoder')}")
            print(f"[DEBUG] DecisionTransformer.forward: state_encoder is None={self.state_encoder is None}")

        graph_aux = None
        if hasattr(self, 'state_encoder') and self.state_encoder is not None:
            if debug_prints:
                print(f"[DEBUG] DecisionTransformer.forward: Using state_encoder to encode states")
                print(f"[DEBUG] DecisionTransformer.forward: states type={type(states)}")
                if isinstance(states, dict):
                    print(f"[DEBUG] DecisionTransformer.forward: states keys={list(states.keys())}")
                    for k, v in states.items():
                        print(f"[DEBUG] DecisionTransformer.forward: states['{k}'] shape={v.shape}, dtype={v.dtype}")
            encoder_device_type = device.type if hasattr(device, "type") else "cuda"
            with torch.amp.autocast(device_type=encoder_device_type, enabled=False):
                if isinstance(states, dict):
                    states_for_encoder = {
                        k: (v.to(dtype=torch.float32) if torch.is_tensor(v) and v.dtype.is_floating_point else v)
                        for k, v in states.items()
                    }
                else:
                    states_for_encoder = states.to(dtype=torch.float32) if states.dtype.is_floating_point else states
                if self.state_encoder_type == "graph" and hasattr(self.state_encoder, "forward_with_aux"):
                    encoded_states, graph_aux = self.state_encoder.forward_with_aux(states_for_encoder)
                    encoded_states = self._apply_scale_param_mask(
                        encoded_states,
                        self.scale_param_mask_gnn_encoder,
                        states,
                        batch_size=batch_size,
                        seq_len=seq_length,
                        device=device,
                    )
                    if graph_aux is not None and graph_aux.get("job_tokens", None) is not None:
                        graph_aux = dict(graph_aux)
                        graph_aux["job_tokens"] = self._apply_scale_param_mask(
                            graph_aux["job_tokens"],
                            self.scale_param_mask_gnn_encoder,
                            states,
                            batch_size=batch_size,
                            seq_len=seq_length,
                            device=device,
                            per_job=True,
                        )
                else:
                    encoded_states = self.state_encoder(states_for_encoder)
                # encoded_states: (B, T, D) — GNN 内部已完成 job-level attention 聚合
                if hasattr(self, 'state_proj') and self.state_proj is not None:
                    states_input = encoded_states.to(dtype=self.state_proj.weight.dtype)
                    projected = self.state_proj(states_input)
                    if hasattr(self, 'state_proj_norm'):
                        projected = self.state_proj_norm(projected.to(dtype=self.state_proj_norm.weight.dtype))
                    scale_condition = self._get_scale_condition_embedding(
                        states,
                        batch_size=batch_size,
                        seq_len=seq_length,
                        device=device,
                        dtype=projected.dtype,
                    )
                    if scale_condition is not None:
                        projected = projected + scale_condition
                    projected = self._apply_scale_param_mask(
                        projected,
                        self.scale_param_mask_state_projection,
                        states,
                        batch_size=batch_size,
                        seq_len=seq_length,
                        device=device,
                    )
                    projected = self._attach_safe_grad_hook(projected)
                    encoded_states = self.embed_dropout(projected)
            if debug_prints:
                print(f"[DEBUG] DecisionTransformer.forward: encoded_states type={type(encoded_states)}, shape={encoded_states.shape}")
        else:
            if debug_prints:
                print(f"[DEBUG] DecisionTransformer.forward: state_encoder not available, using states directly")
            encoded_states = states
            # 仅当state_proj存在时才进行投影，且states不是字典
            if hasattr(self, 'state_proj') and self.state_proj is not None and not isinstance(encoded_states, dict):
                # 确保输入与state_proj层的dtype一致
                states_input = encoded_states.to(dtype=self.state_proj.weight.dtype)
                encoded_states = self.embed_dropout(self.state_proj(states_input))
                # 不再手动转换为float16 - 让autocast自动管理混合精度

        # === 新增的健壮性检查 ===
        if isinstance(encoded_states, dict):
            raise TypeError("The state encoder did not convert the state dictionary into a tensor. "
                        "Please ensure a valid state_encoder is attached to the model.")

        # Now it's safe to assume encoded_states is a tensor

        # Project returns-to-go to hidden size
        # 确保输入与rtg_proj层的dtype一致
        rtg_input = returns_to_go.to(dtype=self.rtg_proj.weight.dtype)
        # 添加数值稳定性：裁剪输入
        rtg_input = torch.clamp(rtg_input, min=-1e4, max=1e4)
        rtg_proj_output = self.rtg_proj(rtg_input)
        # 检查并处理NaN值
        if torch.isnan(rtg_proj_output).any():
            rtg_proj_output = torch.nan_to_num(rtg_proj_output, nan=0.0)
        rtg_embeddings = self.embed_dropout(rtg_proj_output)
        # 添加数值稳定性：裁剪embeddings
        rtg_embeddings = torch.clamp(rtg_embeddings, min=-1e4, max=1e4)

        # Get timestep embeddings if timesteps provided
        if timesteps is not None:
            # 确保timesteps的值不超过max_ep_len，避免索引错误
            safe_timesteps = torch.clamp(timesteps.long(), 0, self.max_ep_len - 1)
            timestep_embeddings = self.timestep_embedding(safe_timesteps)
        else:
            # Default to 0 timestep embedding
            timestep_embeddings = self.timestep_embedding(torch.zeros(batch_size, seq_length, dtype=torch.long, device=device))

        # Get action embeddings with explicit causal handling.
        # We use the sequence structure [R_t, S_t, t_t, A_{t-1}] so the action
        # prediction token can see the current timestep without leaking future info.
        if actions is None:
            # Inference time: use dummy action for all steps except the first
            # Input sequence: [A_0, A_0, ..., A_0] where A_0 is dummy
            # Output predictions: [A_1, A_2, ..., A_T]
            dummy_actions = torch.full((batch_size, seq_length), fill_value=self.n_jobs, device=device, dtype=torch.long)
            # 确保输入与action_proj层的dtype一致（这里是long类型，不需要转换）
            action_embeddings = self.action_proj(dummy_actions)
            # 不再手动转换为float16 - 让autocast自动管理混合精度
        else:
            # Actions shape validation and causal padding
            if actions.ndim == 2:
                # Ensure actions has the correct shape: (batch_size, seq_length, 1) -> add feature dimension
                actions = actions.unsqueeze(-1)

            max_action = self.n_jobs

            if actions.shape[1] == seq_length - 1:
                # Historical-actions case: actions = [A0, A1, ..., A_{T-2}].
                # Pad a dummy action at t=0 so tokens align as [dummy, A0, ..., A_{T-2}]
                # and predict the current action at every state position.
                action_embeddings_padded = torch.full((batch_size, seq_length), fill_value=self.n_jobs, dtype=actions.dtype, device=device)
                action_embeddings_padded[:, 1:] = actions.squeeze(-1)  # A_{t-1} at position t
                # 确保输入与action_proj层的dtype一致（这里是long类型，不需要转换）
                action_ids = action_embeddings_padded.long()
                action_ids[action_ids < 0] = self.n_jobs
                action_embeddings = self.action_proj(action_ids.clamp(min=0, max=max_action))
                # 不再手动转换为float16 - 让autocast自动管理混合精度
            elif actions.shape[1] == seq_length:
                # Previous-action tokens are already aligned with states:
                # [dummy, A0, A1, ...] for predicting [A0, A1, A2, ...].
                # 确保输入与action_proj层的dtype一致（这里是long类型，不需要转换）
                action_ids = actions.squeeze(-1).long()
                action_ids[action_ids < 0] = self.n_jobs
                action_embeddings = self.action_proj(action_ids.clamp(min=0, max=max_action))
                # 不再手动转换为float16 - 让autocast自动管理混合精度
            else:
                # Unexpected sequence length, warn and handle
                print(f"Warning: actions sequence length ({actions.shape[1]}) doesn't match expected length ({seq_length} or {seq_length-1})")
                # Truncate or pad to match expected causal structure
                if actions.shape[1] > seq_length - 1:
                    # Truncate to seq_length-1 and pad at beginning
                    truncated_actions = actions[:, :seq_length-1]
                    action_embeddings_padded = torch.full((batch_size, seq_length), fill_value=self.n_jobs, dtype=actions.dtype, device=device)
                    action_embeddings_padded[:, 1:] = truncated_actions.squeeze(-1)
                    # 确保输入与action_proj层的dtype一致（这里是long类型，不需要转换）
                    action_ids = action_embeddings_padded.long()
                    action_ids[action_ids < 0] = self.n_jobs
                    action_embeddings = self.action_proj(action_ids.clamp(min=0, max=max_action))
                    # 不再手动转换为float16 - 让autocast自动管理混合精度
                else:
                    # Pad with dummy actions at beginning and end to match seq_length-1
                    padded_actions = torch.full((batch_size, seq_length-1), fill_value=self.n_jobs, dtype=actions.dtype, device=device)
                    padded_actions[:, :actions.shape[1]] = actions
                    action_embeddings_padded = torch.full((batch_size, seq_length), fill_value=self.n_jobs, dtype=actions.dtype, device=device)
                    action_embeddings_padded[:, 1:] = padded_actions.squeeze(-1)
                    action_ids = action_embeddings_padded.long()
                    action_ids[action_ids < 0] = self.n_jobs
                    action_embeddings = self.action_proj(action_ids.clamp(min=0, max=max_action))

        # Standard DT architecture with timestep-aware action prediction:
        # [R1, S1, t1, A0, R2, S2, t2, A1, ..., RT, ST, tT, A_{T-1}]
        # Each A_t is predicted given [R_t, S_t, t_t, A_{t-1}], preventing information leakage.

        # Process timestep embeddings
        expanded_timesteps = timesteps if timesteps is not None else torch.zeros(batch_size, seq_length, dtype=torch.long, device=device)
        # Ensure expanded_timesteps is Long type and values don't exceed max_ep_len
        expanded_timesteps = expanded_timesteps.long()
        safe_expanded_timesteps = torch.clamp(expanded_timesteps, 0, self.max_ep_len - 1)
        timestep_embeddings = self.timestep_embedding(safe_expanded_timesteps)
        # 添加数值稳定性：裁剪timestep_embeddings
        timestep_embeddings = torch.clamp(timestep_embeddings, min=-1e4, max=1e4)

        # === NEW: INTERLEAVED SEQUENCE CONSTRUCTION ===
        # Handle both tensor and dict cases for encoded_states
        if debug_prints:
            print(f"[DEBUG] DecisionTransformer.forward: encoded_states type={type(encoded_states)}")
        if isinstance(encoded_states, dict):
            first_key = next(iter(encoded_states.keys()))
            batch_size, seq_len, d_model = encoded_states[first_key].shape
        else:
            if encoded_states.dim() != 3:
                raise ValueError(f"encoded_states must be 3D, got {encoded_states.dim()}D with shape {encoded_states.shape}")
            batch_size, seq_len, d_model = encoded_states.shape

        if debug_prints:
            print(f"[DEBUG] DecisionTransformer.forward: batch_size={batch_size}, seq_len={seq_len}, d_model={d_model}")

        # Expand each component to (B, T, 1, D)
        rtg_exp = rtg_embeddings.unsqueeze(2)                    # (B, T, 1, D)

        if isinstance(encoded_states, dict):
            state_components = []
            for k, v in encoded_states.items():
                v_tensor = v.to(device)
                if hasattr(self, 'state_proj') and self.state_proj is not None:
                    v_tensor = v_tensor.to(dtype=self.state_proj.weight.dtype)
                    v_proj = self.state_proj(v_tensor)
                    state_components.append(v_proj)
                else:
                    state_components.append(v_tensor)
            concatenated_states = torch.cat(state_components, dim=-1)
            concatenated_states = self.embed_dropout(concatenated_states)
            state_exp = concatenated_states.unsqueeze(2)
        else:
            state_exp = encoded_states.unsqueeze(2)               # (B, T, 1, D)

        time_exp = timestep_embeddings.unsqueeze(2)            # (B, T, 1, D)
        act_exp = action_embeddings.unsqueeze(2)               # (B, T, 1, D)

        # 序列结构：[RTG_t, S_t, Time_t, A_{t-1}]。
        # 这样用于预测动作的 token 可以直接看到当前时间步条件。
        interleaved = torch.cat([rtg_exp, state_exp, time_exp, act_exp], dim=2)
        tokens_per_step = 4

        if torch.isnan(interleaved).any():
            interleaved = torch.nan_to_num(interleaved, nan=0.0)

        stacked_inputs = interleaved.view(batch_size, tokens_per_step * seq_len, d_model)
        stacked_inputs = torch.clamp(stacked_inputs, min=-1e4, max=1e4)

        # type: 0=RTG, 1=state, 2=action, 3=timestep
        type_ids = torch.tensor([0, 1, 3, 2], device=device).expand(seq_len, 4).reshape(1, -1).expand(batch_size, -1)
        type_embs = self.type_embedding(type_ids)
        stacked_inputs = stacked_inputs + type_embs
        stacked_inputs = self.input_fuse_norm(stacked_inputs.to(dtype=self.input_fuse_norm.weight.dtype if hasattr(self.input_fuse_norm, 'weight') and self.input_fuse_norm.weight is not None else torch.float32))
        stacked_inputs = stacked_inputs * float(self.input_embed_scale)
        stacked_inputs = self._attach_safe_grad_hook(stacked_inputs)

        # Expand attention mask
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)
        attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, tokens_per_step).view(batch_size, -1)
        attention_mask = attention_mask.to(device)

        # Note: For Hugging Face CausalLM models, causal masking is typically handled internally
        # We only need to provide the padding mask (2D tensor: batch_size, seq_length)
        # The model will automatically apply causal masking to ensure autoregressive property

        # The attention_mask indicates which tokens should be attended to (True) or masked (False)
        # For padding tokens, we set them to False to prevent attending to them
        # Convert to float32 as expected by HF models and ensure no NaN values
        attention_mask = attention_mask.to(torch.float32)

        # Safety check: replace NaN values with 0.0
        if torch.isnan(attention_mask).any():
            attention_mask = torch.nan_to_num(attention_mask, nan=0.0)

        # Forward pass through transformer backbone
        # 获取模型的dtype，确保输入数据类型与模型一致
        model_dtype = self.base_model.dtype if hasattr(self.base_model, 'dtype') else next(self.base_model.parameters()).dtype
        stacked_inputs = stacked_inputs.to(dtype=model_dtype)

        # 内存优化：移除调试打印语句以减少内存占用

        if hasattr(self.base_model, 'model'):
            # For models like Qwen, the transformer is in base_model.model
            transformer_outputs = self.base_model.model(
                inputs_embeds=stacked_inputs,
                attention_mask=attention_mask,
                use_cache=False,  # Disable cache for training
                output_hidden_states=True,  # 确保返回所有隐藏状态
                return_dict=True
            )
            # 适配不同模型的输出格式
            if hasattr(transformer_outputs, 'last_hidden_state'):
                hidden_states = transformer_outputs.last_hidden_state
            elif hasattr(transformer_outputs, 'hidden_states') and transformer_outputs.hidden_states is not None:
                # 有些模型可能返回的是包含所有隐藏状态的元组，检查是否为None且非空
                hidden_states = transformer_outputs.hidden_states[-1] if transformer_outputs.hidden_states else transformer_outputs[0]
            else:
                # 如果以上都没有，尝试直接从模型输出中获取
                hidden_states = transformer_outputs[0]
        elif hasattr(self.base_model, 'transformer'):
            # For other models like GPT2
            transformer_outputs = self.base_model.transformer(
                inputs_embeds=stacked_inputs,
                attention_mask=attention_mask
            )
            # 适配不同模型的输出格式
            if hasattr(transformer_outputs, 'last_hidden_state'):
                hidden_states = transformer_outputs.last_hidden_state
            elif hasattr(transformer_outputs, 'hidden_states') and transformer_outputs.hidden_states is not None:
                # 检查hidden_states是否为None且非空
                hidden_states = transformer_outputs.hidden_states[-1] if transformer_outputs.hidden_states else transformer_outputs[0]
            else:
                hidden_states = transformer_outputs[0]
        else:
            # Fallback for models with different structure
            transformer_outputs = self.base_model(
                inputs_embeds=stacked_inputs,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            # 适配不同模型的输出格式
            if hasattr(transformer_outputs, 'last_hidden_state'):
                hidden_states = transformer_outputs.last_hidden_state
            elif hasattr(transformer_outputs, 'hidden_states') and transformer_outputs.hidden_states is not None:
                # 检查hidden_states是否为None且非空
                hidden_states = transformer_outputs.hidden_states[-1] if transformer_outputs.hidden_states else transformer_outputs[0]
            else:
                hidden_states = transformer_outputs[0]

        # 内存优化：移除调试打印语句以减少内存占用

        # 添加数值稳定性：检查并处理hidden_states中的NaN值
        if torch.isnan(hidden_states).any():
            hidden_states = torch.nan_to_num(hidden_states, nan=0.0)

        # 添加数值稳定性：裁剪hidden_states
        hidden_states = torch.clamp(hidden_states, min=-1e4, max=1e4)

        # ── 提取 Action 位置（4t+3）的隐藏态来预测动作 ──────────────────────
        # 序列结构：[RTG_t(4t), S_t(4t+1), Time_t(4t+2), A_{t-1}(4t+3)]
        action_positions = torch.arange(3, 4 * seq_length, 4, device=device)

        # 确保索引不超过hidden_states的实际序列长度
        max_hidden_seq_len = hidden_states.shape[1]
        valid_indices = action_positions < max_hidden_seq_len
        action_positions = action_positions[valid_indices]

        # 如果没有有效的动作位置索引，返回所有hidden_states
        if action_positions.numel() == 0:
            action_hidden_states = hidden_states
        else:
            try:
                # 安全地提取动作位置的隐藏状态
                action_hidden_states = hidden_states[:, action_positions, :]
            except IndexError as e:
                print(f"Error extracting action hidden states: {e}")
                print(f"Action positions: {action_positions}")
                print(f"Hidden states shape: {hidden_states.shape}")
                # 回退到返回所有hidden_states
                action_hidden_states = hidden_states

        # 添加数值稳定性：检查并处理action_hidden_states中的NaN值
        if torch.isnan(action_hidden_states).any():
            action_hidden_states = torch.nan_to_num(action_hidden_states, nan=0.0)

        # 添加数值稳定性：裁剪action_hidden_states
        action_hidden_states = torch.clamp(action_hidden_states, min=-1e4, max=1e4)

        # Predict action logits from the Action position hidden states
        # Output shape: (batch_size, seq_length, n_jobs) if using action_head,
        # 输出形状: (batch_size, seq_length, n_jobs) - 统一使用作业级别动作空间

        # 候选作业感知动作头：对每个 job token 单独打分，而不是直接做全局平面分类。
        action_hidden_states = action_hidden_states.to(dtype=self.action_head.weight.dtype)

        # 添加数值稳定性措施：裁剪隐藏状态
        action_hidden_states = torch.clamp(action_hidden_states, min=-1e4, max=1e4)

        value_hidden_states = action_hidden_states.to(dtype=self.value_head[1].weight.dtype)
        value_preds = self.value_head(value_hidden_states).squeeze(-1)

        use_candidate_logits = (
            self.use_candidate_action_head
            and graph_aux is not None
            and self.candidate_job_proj is not None
            and self.candidate_context_proj is not None
            and self.candidate_action_head is not None
        )
        if use_candidate_logits:
            job_tokens = graph_aux.get("job_tokens", None)
            job_valid_mask = graph_aux.get("job_valid_mask", None)
            if job_tokens is not None and job_valid_mask is not None:
                score_len = min(action_hidden_states.shape[1], job_tokens.shape[1])
                context_states = action_hidden_states[:, :score_len, :].to(dtype=self.candidate_context_proj.weight.dtype)
                job_tokens = job_tokens[:, :score_len, :, :].to(dtype=self.candidate_job_proj.weight.dtype)
                job_valid_mask = job_valid_mask[:, :score_len, :].to(device=device, dtype=torch.bool)

                context_proj = self.candidate_context_proj(context_states).unsqueeze(2)
                job_proj = self.candidate_job_proj(job_tokens)
                fused = context_proj + job_proj
                fused = self.candidate_fuse_norm(fused.to(dtype=self.candidate_fuse_norm.weight.dtype))
                fused = torch.tanh(fused)
                harmony_mask = self._get_candidate_harmony_mask(
                    states,
                    batch_size=action_hidden_states.shape[0],
                    seq_len=score_len,
                    device=device,
                    dtype=fused.dtype,
                )
                if harmony_mask is not None:
                    fused = fused * harmony_mask
                scale_param_candidate_mask = self._get_scale_param_mask(
                    self.scale_param_mask_candidate_head,
                    states,
                    batch_size=action_hidden_states.shape[0],
                    seq_len=score_len,
                    device=device,
                    dtype=fused.dtype,
                )
                if scale_param_candidate_mask is not None:
                    fused = fused * scale_param_candidate_mask.unsqueeze(2)
                action_logits = self.candidate_action_head(fused.to(dtype=self.candidate_action_head.weight.dtype)).squeeze(-1)
                residual_fused = fused
                context_delta = self._scale_context_adapter_delta(context_states, states, device)
                if context_delta is not None:
                    residual_fused = residual_fused + context_delta.to(dtype=residual_fused.dtype).unsqueeze(2)
                residual_logits = self._scale_candidate_residual_logits(residual_fused, states, device)
                if residual_logits is not None:
                    action_logits = action_logits + residual_logits.to(dtype=action_logits.dtype)
                invalid_jobs = ~job_valid_mask
                action_logits = action_logits.masked_fill(invalid_jobs, -1e9)
            else:
                action_logits = self.action_head(action_hidden_states)
        else:
            action_logits = self.action_head(action_hidden_states)

        # 添加数值稳定性措施：裁剪logits
        action_logits = torch.clamp(action_logits, min=-1e4, max=1e4)

        if action_mask is None and env is not None and hasattr(env, 'get_action_mask'):
            action_mask = env.get_action_mask()
        if action_mask is not None:
            if not isinstance(action_mask, torch.Tensor):
                action_mask = torch.as_tensor(action_mask, dtype=torch.bool, device=device)
            else:
                action_mask = action_mask.to(device=device, dtype=torch.bool)
            if action_mask.dim() == 2:
                action_mask = action_mask.any(dim=-1)
            if action_mask.dim() == 1:
                if action_mask.numel() < self.n_jobs:
                    pad = torch.zeros((self.n_jobs - action_mask.numel(),), dtype=torch.bool, device=device)
                    action_mask = torch.cat([action_mask, pad], dim=0)
                elif action_mask.numel() > self.n_jobs:
                    action_mask = action_mask[:self.n_jobs]
                action_mask_expanded = action_mask.unsqueeze(0).unsqueeze(0).expand(action_logits.shape[0], action_logits.shape[1], -1)
            elif action_mask.dim() == 3:
                if action_mask.shape[-1] < self.n_jobs:
                    pad = torch.zeros(action_mask.shape[0], action_mask.shape[1], self.n_jobs - action_mask.shape[-1], dtype=torch.bool, device=device)
                    action_mask = torch.cat([action_mask, pad], dim=-1)
                elif action_mask.shape[-1] > self.n_jobs:
                    action_mask = action_mask[..., :self.n_jobs]
                action_mask_expanded = action_mask
            else:
                raise ValueError(f"Unsupported action_mask dim: {action_mask.dim()}")
            invalid_mask = ~action_mask_expanded
            action_logits = action_logits.masked_fill(invalid_mask, -1e9)

        # Keep logits aligned to state positions. Training and rollout both use:
        #   action input tokens = [dummy, A0, A1, ...]
        #   prediction targets  = [A0,    A1, A2, ...]
        # Therefore the logits should have seq_length positions, not actions.shape[1].
        if actions is not None:
            # 动作范围检查：确保target_actions在[0, n_jobs-1]范围内，忽略padding值(-100)
            valid_actions = actions[(actions != -100) & (actions != self.n_jobs)]
            if valid_actions.numel() > 0:
                assert valid_actions.min() >= 0 and valid_actions.max() < self.n_jobs, f"无效动作范围: {valid_actions.min()} ～ {valid_actions.max()}，应为 [0, {self.n_jobs-1}]"
            if action_logits.shape[1] > seq_length:
                action_logits = action_logits[:, :seq_length, :]
                value_preds = value_preds[:, :seq_length]

        # Create outputs object similar to Hugging Face format
        class DTOutputs:
            def __init__(self, logits, value_preds=None):
                self.logits = logits
                self.value_preds = value_preds

        return DTOutputs(action_logits, value_preds=value_preds)

    def predict_action(self, states, returns_to_go, timesteps=None, actions=None, env=None, **kwargs):
        """
        Predict the next action A_t given [R_t, S_t, t_t, A_{t-1}] using the Decision Transformer.

        Args:
            states: Current and historical states (batch_size, seq_length, state_dim)
                - Input: [S_1, S_2, ..., S_T] where S_T is the current state
            returns_to_go: Target returns-to-go (batch_size, seq_length, 1)
                - Input: [R_1, R_2, ..., R_T] where R_T is the current target return
            timesteps: Timesteps (batch_size, seq_length) or None
                - Input: [t_1, t_2, ..., t_T] for timestep conditioning
            actions: Historical actions (batch_size, seq_length-1) or None
                - Input: [A_0, A_1, ..., A_{T-1}] where A_{T-1} is the most recent action
                - If None, uses dummy actions for all positions
            env: Environment object with get_action_mask() method
                - Used to get valid actions for masking

        Returns:
            Predicted action indices (batch_size,) for A_T (the next action)
        """
        with torch.no_grad():
            # 统一输入要求：确保所有输入都是3D张量

            # 强制 states 为 3D
            if states.dim() == 2:
                states = states.unsqueeze(0)  # (L, D) → (1, L, D)
            elif states.dim() == 4:
                # 处理 (B, L, J, M) → 展平最后两维
                states = states.flatten(start_dim=2)
            assert states.dim() == 3, f"States must be 3D, got {states.dim()}D"

            # 确保 returns_to_go 为 3D
            if returns_to_go.dim() == 2:
                returns_to_go = returns_to_go.unsqueeze(-1)  # (B, L) → (B, L, 1)
            elif returns_to_go.dim() == 1:
                returns_to_go = returns_to_go.unsqueeze(0).unsqueeze(-1)  # (L,) → (1, L, 1)
            assert returns_to_go.dim() == 3, f"Returns-to-go must be 3D, got {returns_to_go.dim()}D"

            # 确保 timesteps 为 3D（如果提供）
            if timesteps is not None:
                if timesteps.dim() == 2:
                    # 已经是 (B, L)，不需要改变
                    pass
                elif timesteps.dim() == 1:
                    timesteps = timesteps.unsqueeze(0)  # (L,) → (1, L)
                assert timesteps.dim() <= 3, f"Timesteps must be at most 3D, got {timesteps.dim()}D"

            # Forward pass through the model
            outputs = self.forward(states, returns_to_go, timesteps, actions, env=env)

            # Get the action logits for the last timestep
            batch_size, seq_length, num_actions = outputs.logits.shape
            last_logits = outputs.logits[:, -1, :]

            if env is not None and hasattr(env, 'get_action_mask'):
                valid_mask = env.get_action_mask()
                if not isinstance(valid_mask, torch.Tensor):
                    valid_mask = torch.as_tensor(valid_mask, dtype=torch.bool, device=last_logits.device)
                else:
                    valid_mask = valid_mask.to(device=last_logits.device, dtype=torch.bool)
                if valid_mask.dim() == 2:
                    valid_mask = valid_mask.any(dim=-1)
                if valid_mask.dim() != 1:
                    raise ValueError(f"Unsupported env action mask dim: {valid_mask.dim()}")
                if valid_mask.numel() < self.n_jobs:
                    pad = torch.zeros((self.n_jobs - valid_mask.numel(),), dtype=torch.bool, device=last_logits.device)
                    valid_mask = torch.cat([valid_mask, pad], dim=0)
                elif valid_mask.numel() > self.n_jobs:
                    valid_mask = valid_mask[:self.n_jobs]
                if int(valid_mask.sum().item()) == 0:
                    return torch.tensor([-1] * batch_size, dtype=torch.long, device=last_logits.device)
                last_logits = last_logits.masked_fill(~valid_mask.unsqueeze(0).expand(batch_size, -1), -1e9)

            # Sample or argmax based on parameters
            if kwargs.get('sample', False):
                # Temperature sampling
                temperature = kwargs.get('temperature', 1.0)
                if temperature <= 0:
                    temperature = 1.0

                # Apply temperature to logits
                logits_with_temp = last_logits / temperature
                probs = torch.softmax(logits_with_temp, dim=-1)
                predicted_actions = torch.multinomial(probs, 1).squeeze(-1)
            else:
                # Argmax selection
                predicted_actions = torch.argmax(last_logits, dim=-1)

        return predicted_actions

    def generate_trajectory(self, env, initial_state, target_makespan, max_steps=100):
        """
        Generate a trajectory using the Decision Transformer by interacting with a JSSP environment.

        Args:
            env: JSSP environment that must implement the following interface:
                 - env.step(action_idx): Takes an action index and returns tuple (next_state, reward, done)
                 - env.current_time: Current time in the scheduling process (usually the maximum machine completion time)
                 - next_state: The state representation after taking the action
                 - reward: The reward received for taking the action
                 - done: Boolean indicating if the episode has terminated
            initial_state: Initial state representation (either tensor, dict, or list)
            target_makespan: Target makespan for the trajectory. This is the maximum allowed time to complete all jobs.
                         For JSSP, this should be a positive value (e.g., OR-Tools result * 1.1).
            max_steps: Maximum number of steps to generate

        Returns:
            Trajectory as tuple (states, actions, rewards)
            - states: List of state representations
            - actions: List of action indices taken
            - rewards: List of rewards received

        Note:
            The returns-to-go calculation uses remaining time budget: remaining_budget = max(0, target_makespan - current_time).
            This is more effective for JSSP as it directly relates to the makespan objective, providing clearer gradient signals.
        """
        states = [initial_state]
        actions = []
        rewards = []
        done = False

        # Prepare initial inputs - ensure state is properly formatted for the model
        # Convert initial_state to tensor format
        if hasattr(self, 'state_encoder') and self.state_encoder is not None:
            # If we have a state encoder, use it to encode the initial state
            if not isinstance(initial_state, torch.Tensor):
                # Convert to tensor if needed
                if isinstance(initial_state, dict):
                    # For dict states, use state encoder directly
                    encoded_state = self.state_encoder(initial_state)
                    state = encoded_state.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, hidden_size)
                else:
                    # For other formats, convert to tensor first
                    state_tensor = torch.tensor(initial_state, dtype=torch.float32)
                    state = state_tensor.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, D)
            else:
                state = initial_state.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, D)
        else:
            # If no state encoder, ensure it's a tensor
            state = torch.tensor(initial_state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Initialize returns_to_go using remaining time budget (directly using environment time, decoupled from reward)
        initial_current_time = getattr(env, 'current_time', 0.0)
        returns_to_go = torch.tensor([[max(0, target_makespan - initial_current_time)]], dtype=torch.float32)

        for t in range(max_steps):
            if done:
                break

            # Truncate to max_length//2 if needed (each step contributes 2 tokens: r and s)
            max_state_steps = self.max_length // 2
            if state.shape[1] >= max_state_steps:
                state = state[:, -max_state_steps:]
                returns_to_go = returns_to_go[:, -max_state_steps:]

            # Prepare timestep
            timestep = torch.tensor([[min(t, self.max_ep_len)]], dtype=torch.long)

            # Create dummy actions tensor (not used in forward pass)
            dummy_actions = None

            # --- 关键修改开始 ---
            while True:  # 循环直到找到一个有效动作或环境结束
                # Predict next action with environment for action masking
                next_action = self.predict_action(
                    state, returns_to_go, timestep, dummy_actions, env=env,
                    sample=True
                )
                action_idx = next_action.item()

                if action_idx == -1:
                    # 无合法动作，强制推进时间，但不记录为动作
                    try:
                        updated_state, _, env_done, _ = env.step()  # 只推进时间，获取新状态
                        if env_done:
                            done = True
                            states.append(updated_state)  # 记录最终状态
                            break

                        # 更新当前状态和 rtg 以反映新时间
                        # 更新模型输入的状态和 rtg
                        if hasattr(self, 'state_encoder') and self.state_encoder is not None:
                            if isinstance(updated_state, dict):
                                encoded_updated_state = self.state_encoder(updated_state)
                                updated_state_tensor = encoded_updated_state.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, hidden_size)
                            else:
                                updated_state_tensor = torch.tensor(updated_state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                        else:
                            updated_state_tensor = torch.tensor(updated_state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

                        # 更新 state
                        state = torch.cat([state, updated_state_tensor], dim=1)

                        # 更新 returns_to_go
                        current_time = getattr(env, 'current_time', float('inf'))
                        remaining_budget = max(0, target_makespan - current_time)
                        returns_to_go = torch.cat([returns_to_go, torch.tensor([[remaining_budget]], dtype=torch.float32)], dim=1)

                        # 记录状态但不记录动作
                        states.append(updated_state)

                        # 继续循环，尝试预测下一个动作
                        continue
                    except Exception as e:
                        raise RuntimeError(f"Environment step failed when advancing time. Error: {e}")
                else:
                    # 找到有效动作，执行并记录
                    try:
                        next_state, reward, done, _ = env.step(action_idx)
                    except Exception as e:
                        raise RuntimeError(f"Environment step failed when executing action. Error: {e}")

                    # 更新轨迹数据
                    actions.append(action_idx)  # 只记录有效的 job-machine 分配动作
                    rewards.append(reward)
                    states.append(next_state)

                    # 更新模型输入的状态和 rtg
                    if hasattr(self, 'state_encoder') and self.state_encoder is not None:
                        if isinstance(next_state, dict):
                            encoded_next_state = self.state_encoder(next_state)
                            next_state_tensor = encoded_next_state.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, hidden_size)
                        else:
                            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    else:
                        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

                    # 连接新状态
                    state = torch.cat([state, next_state_tensor], dim=1)

                    # 更新 returns_to_go
                    current_time = getattr(env, 'current_time', float('inf'))
                    remaining_budget = max(0, target_makespan - current_time)
                    returns_to_go = torch.cat([returns_to_go, torch.tensor([[remaining_budget]], dtype=torch.float32)], dim=1)

                    break  # 跳出循环，继续下一个步骤
            # --- 关键修改结束 ---

        return states, actions, rewards


def create_decision_transformer_for_jssp(model_name="unsloth/Qwen2.5-7B", load_in_4bit=False, lang_model=None, **kwargs):
    """
    Create a Decision Transformer specifically configured for JSSP.

    Args:
        model_name: Name of the pre-trained model
        load_in_4bit: Whether to load the model in 4-bit quantization
        lang_model: Pre-loaded language model (optional)
        kwargs: Additional parameters for DecisionTransformer and JSSPStateEncoder

    Returns:
        Decision Transformer model
    """
    # For JSSP, we need to determine appropriate state and action dimensions
    # This depends on how you represent your JSSP states and actions

    # Example configuration for JSSP with dynamic feature dimensions
    # Default to typical JSSP feature dimensions if not specified
    n_jobs = kwargs.pop('n_jobs', 10)  # Get and remove n_jobs from kwargs
    n_machines = kwargs.pop('n_machines', 10)  # Get and remove n_machines from kwargs

    max_length = kwargs.pop('max_length', 128)  # Get max_length from kwargs or use default
    max_ep_len = kwargs.pop('max_ep_len', 100)    # Get max_ep_len from kwargs or use default
    model_source = kwargs.pop('model_source', 'huggingface')  # Get model_source from kwargs or use default

    # Pop state_dim and act_dim from kwargs if they exist to avoid duplicate parameters
    kwargs.pop('state_dim', None)
    kwargs.pop('act_dim', None)  # Remove act_dim if present

    # Create the decision transformer with proper state_dim
    # When using JSSPStateEncoder, the output dimension will be dt.hidden_size
    dt = DecisionTransformer(
        model_name=model_name,
        state_dim=None,  # Will be handled appropriately in forward pass
        n_jobs=n_jobs,  # Use n_jobs directly for action space
        n_machines=n_machines,  # Use n_machines for action space
        max_length=max_length,
        max_ep_len=max_ep_len,
        model_source=model_source,
        load_in_4bit=load_in_4bit,  # 传递4bit量化参数
        lang_model=lang_model,
        use_action_decoder=False,  # 统一使用作业级别动作空间，禁用action_decoder
        **kwargs
    )

    # ✅ 注意：JSSPStateEncoder 现在在 DecisionTransformer.__init__ 中直接创建为子模块
    # 不再需要在这里手动设置 state_encoder
    print(f"✅ DecisionTransformer已创建，state_encoder已自动配置")
    print(f"[DEBUG] state_encoder type: {type(dt.state_encoder)}")
    print(f"[DEBUG] state_encoder is None: {dt.state_encoder is None}")

    return dt


class GraphJSSPStateEncoder(nn.Module):
    """
    改进版析取图GNN状态编码器 v2。

    核心改进：
    1. 析取边：连接同一机器上所有「未完成操作」（next_op及其后续），而非只连接next_op；
       大幅提高图连通性，让GNN捕获真实机器竞争关系。
    2. 节点特征：增至16维，新增线性归一化处理时间、剩余总处理时间比例、是否为待调度操作。
    3. 消息聚合：共轭边/析取边各用独立可学习Linear门控（向量级）替代原来的共享标量gate；
       残差后接GRU门控更新，提升深层稳定性。
    4. 已排定机器顺序边：把 partial schedule 中真实形成的机器加工链作为第三类有向边。
    5. 多粒度池化：job级mean + machine级mean + global mean，拼接后投影到embedding_dim，
       保留局部结构信息而非单纯global mean-pool。
    """
    def __init__(self, state_dims, embedding_dim=None, hidden_size=None, num_layers=3, use_queue_position=True):
        super().__init__()
        if hidden_size is not None:
            self.embedding_dim = int(hidden_size)
        elif embedding_dim is not None:
            self.embedding_dim = int(embedding_dim)
        else:
            self.embedding_dim = 256

        self.max_jobs = int(state_dims.get("n_jobs", 10) or 10)
        self.max_machines = int(state_dims.get("n_machines", 10) or 10)
        self.max_nodes = int(self.max_jobs * self.max_machines)
        self.num_layers = int(num_layers)
        self.use_queue_position = bool(use_queue_position)
        try:
            self.disj_edge_chunk_size = max(1, int(os.environ.get("JSSP_GRAPH_EDGE_CHUNK_SIZE", "65536")))
        except (TypeError, ValueError):
            self.disj_edge_chunk_size = 65536

        # 节点索引缓存
        job_ids = torch.arange(self.max_jobs, dtype=torch.long).repeat_interleave(self.max_machines)
        op_ids = torch.arange(self.max_machines, dtype=torch.long).repeat(self.max_jobs)
        self.register_buffer("_node_job_ids", job_ids, persistent=False)
        self.register_buffer("_node_op_ids", op_ids, persistent=False)

        # 共轭边（precedence edges）：每个job内部的操作序列 op_k -> op_{k+1}
        src = []
        dst = []
        for j in range(self.max_jobs):
            base = j * self.max_machines
            for k in range(self.max_machines - 1):
                src.append(base + k)
                dst.append(base + k + 1)
        edge_src = torch.tensor(src, dtype=torch.long) if src else torch.zeros((0,), dtype=torch.long)
        edge_dst = torch.tensor(dst, dtype=torch.long) if dst else torch.zeros((0,), dtype=torch.long)
        self.register_buffer("_pre_src", edge_src, persistent=False)
        self.register_buffer("_pre_dst", edge_dst, persistent=False)

        # 节点特征维度：18维（17维 + 1维LB特征）
        node_feat_dim = 18
        self.node_in = nn.Linear(node_feat_dim, self.embedding_dim)

        # 共轭/析取消息网络：各层独立Linear
        self.pre_msg = nn.ModuleList([
            nn.Linear(self.embedding_dim, self.embedding_dim) for _ in range(self.num_layers)
        ])
        self.disj_msg = nn.ModuleList([
            nn.Linear(self.embedding_dim, self.embedding_dim) for _ in range(self.num_layers)
        ])
        self.scheduled_msg = nn.ModuleList([
            nn.Linear(self.embedding_dim, self.embedding_dim) for _ in range(self.num_layers)
        ])
        # 向量级可学习门控（替代原来的共享标量gate）
        self.pre_gate_vec = nn.ModuleList([
            nn.Linear(self.embedding_dim, self.embedding_dim, bias=False) for _ in range(self.num_layers)
        ])
        self.disj_gate_vec = nn.ModuleList([
            nn.Linear(self.embedding_dim, self.embedding_dim, bias=False) for _ in range(self.num_layers)
        ])
        self.scheduled_gate_vec = nn.ModuleList([
            nn.Linear(self.embedding_dim, self.embedding_dim, bias=False) for _ in range(self.num_layers)
        ])
        # GRU门控更新（替代简单残差+ReLU）
        self.gru_cells = nn.ModuleList([
            nn.GRUCell(self.embedding_dim, self.embedding_dim) for _ in range(self.num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(self.embedding_dim) for _ in range(self.num_layers)
        ])

        # job-level token投影：每个job的embedding + global_pool拼接 -> embedding_dim
        self.job_token_proj = nn.Linear(2 * self.embedding_dim, self.embedding_dim)
        # job-level self-attention：在 GNN 侧做 job 间 attention 聚合成单个向量
        # 这样 Qwen 序列结构不变（4 tokens/step），但 state token 包含 job 间比较信息
        self.job_attn = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=4,
            batch_first=True,
            dropout=0.0,
        )
        self.job_attn_norm = nn.LayerNorm(self.embedding_dim)
        # 可学习的 query：从 job tokens 中聚合出单个全局 state 表示
        self.job_query = nn.Parameter(torch.zeros(1, 1, self.embedding_dim))
        nn.init.normal_(self.job_query, std=0.02)
        # 保留旧名称兼容性
        self.pool_proj = nn.Linear(3 * self.embedding_dim, self.embedding_dim)

    def forward(self, state):
        state_vec, _ = self.forward_with_aux(state)
        return state_vec

    def forward_with_aux(self, state):
        if not isinstance(state, dict):
            raise TypeError(f"Expected dict state, got {type(state)}")

        device = next(self.parameters()).device
        # GNN 编码器内部固定按参数 dtype 计算，避免在 autocast 下反复切到低精度导致梯度不稳定。
        model_dtype = next(self.parameters()).dtype

        def _to(x):
            if isinstance(x, np.ndarray):
                return torch.tensor(x, dtype=model_dtype, device=device)
            if not isinstance(x, torch.Tensor):
                return torch.as_tensor(x, dtype=model_dtype, device=device)
            return x.to(device=device, dtype=model_dtype)

        raw_job_next_op = state.get("job_next_op", state.get("job_status", None))
        raw_machine_sequence = state.get("machine_sequence", None)
        raw_remaining_processing_times = state.get("remaining_processing_times", None)
        raw_machine_availability = state.get("normalized_machine_availability", state.get("machine_status", None))
        raw_machine_busy = state.get("machine_busy_mask", None)
        raw_overall_info = state.get("overall_info", None)
        raw_scheduled_machine_adj = state.get("scheduled_machine_adj", None)
        raw_problem_n_jobs = state.get("problem_n_jobs", None)
        raw_problem_n_machines = state.get("problem_n_machines", None)

        job_next_op = _to(raw_job_next_op) if raw_job_next_op is not None else None
        machine_sequence = _to(raw_machine_sequence) if raw_machine_sequence is not None else None
        remaining_processing_times = _to(raw_remaining_processing_times) if raw_remaining_processing_times is not None else None
        machine_availability = _to(raw_machine_availability) if raw_machine_availability is not None else None
        machine_busy = _to(raw_machine_busy) if raw_machine_busy is not None else None
        overall_info = _to(raw_overall_info) if raw_overall_info is not None else None
        scheduled_machine_adj = _to(raw_scheduled_machine_adj) if raw_scheduled_machine_adj is not None else None
        problem_n_jobs = _to(raw_problem_n_jobs) if raw_problem_n_jobs is not None else None
        problem_n_machines = _to(raw_problem_n_machines) if raw_problem_n_machines is not None else None

        if job_next_op is None or machine_sequence is None or remaining_processing_times is None or machine_availability is None or overall_info is None:
            raise ValueError("GraphJSSPStateEncoder requires job_next_op, machine_sequence, remaining_processing_times, normalized_machine_availability, overall_info")

        # 统一维度为 (B, T, D)
        def _ensure_3d(t):
            if t is None:
                return t
            if t.dim() == 1:
                return t.unsqueeze(0).unsqueeze(0)
            if t.dim() == 2:
                return t.unsqueeze(1)
            return t

        job_next_op = _ensure_3d(job_next_op)
        machine_sequence = _ensure_3d(machine_sequence)
        remaining_processing_times = _ensure_3d(remaining_processing_times)
        machine_availability = _ensure_3d(machine_availability)
        machine_busy = _ensure_3d(machine_busy)
        overall_info = _ensure_3d(overall_info)
        scheduled_machine_adj = _ensure_3d(scheduled_machine_adj)
        problem_n_jobs = _ensure_3d(problem_n_jobs)
        problem_n_machines = _ensure_3d(problem_n_machines)

        batch_size, seq_len, _ = job_next_op.shape
        flat_bt = int(batch_size * seq_len)

        # ── 动态 padding：将输入 pad 到 max_jobs / max_machines 容量 ──────────
        # 这样训练时用10x5的数据，推理时可以用10x10，GNN权重完全复用
        def _pad_to(t, target_len):
            """将最后一维 pad/截断到 target_len。"""
            cur = t.shape[-1]
            if cur == target_len:
                return t
            if cur < target_len:
                pad_shape = list(t.shape)
                pad_shape[-1] = target_len - cur
                return torch.cat([t, torch.zeros(pad_shape, dtype=t.dtype, device=t.device)], dim=-1)
            return t[..., :target_len]

        job_next_op = _pad_to(job_next_op.reshape(flat_bt, -1), self.max_jobs)
        machine_availability = _pad_to(machine_availability.reshape(flat_bt, -1), self.max_machines)
        overall_info = overall_info.reshape(flat_bt, 3)

        if machine_busy is None:
            machine_busy = torch.zeros((flat_bt, self.max_machines), dtype=model_dtype, device=device)
        else:
            machine_busy = _pad_to(machine_busy.reshape(flat_bt, -1), self.max_machines)

        if problem_n_jobs is None:
            problem_n_jobs = torch.full((flat_bt, 1), float(self.max_jobs), dtype=model_dtype, device=device)
        else:
            problem_n_jobs = problem_n_jobs.reshape(flat_bt, 1)
        if problem_n_machines is None:
            problem_n_machines = torch.full((flat_bt, 1), float(self.max_machines), dtype=model_dtype, device=device)
        else:
            problem_n_machines = problem_n_machines.reshape(flat_bt, 1)

        # machine_sequence 和 remaining_processing_times pad 到 max_nodes
        machine_sequence = _pad_to(machine_sequence.reshape(flat_bt, -1), self.max_nodes)
        remaining_processing_times = _pad_to(remaining_processing_times.reshape(flat_bt, -1), self.max_nodes)
        if scheduled_machine_adj is None:
            scheduled_machine_adj = torch.zeros((flat_bt, self.max_nodes * self.max_nodes), dtype=model_dtype, device=device)
        else:
            scheduled_machine_adj = _pad_to(scheduled_machine_adj.reshape(flat_bt, -1), self.max_nodes * self.max_nodes)

        job_ids = self._node_job_ids.to(device=device)
        op_ids = self._node_op_ids.to(device=device)

        pn_jobs = problem_n_jobs
        pn_machines = problem_n_machines
        valid_job = (job_ids.unsqueeze(0) < pn_jobs).to(dtype=model_dtype)
        valid_op = (op_ids.unsqueeze(0) < pn_machines).to(dtype=model_dtype)
        node_mask = (valid_job * valid_op).clamp(0, 1)  # (flat_bt, max_nodes)

        next_op_for_node = job_next_op[:, job_ids]  # (flat_bt, max_nodes)
        is_done = (op_ids.unsqueeze(0).to(dtype=model_dtype) < next_op_for_node).to(dtype=model_dtype) * node_mask
        is_next = (op_ids.unsqueeze(0).to(dtype=model_dtype) == next_op_for_node).to(dtype=model_dtype) * node_mask
        is_next = is_next * (next_op_for_node < pn_machines).to(dtype=model_dtype)
        # 新增：是否为「未来待处理」操作（next_op之后的）
        is_future = (op_ids.unsqueeze(0).to(dtype=model_dtype) > next_op_for_node).to(dtype=model_dtype) * node_mask

        denom_m = torch.clamp(pn_machines, min=1.0)
        rem_ops = torch.clamp((pn_machines - next_op_for_node) / denom_m, min=0.0, max=1.0) * node_mask

        proc = torch.clamp(remaining_processing_times, min=0.0)
        proc_scale = torch.amax(proc, dim=1, keepdim=True).clamp(min=1.0)
        # 改进1：同时保留log归一化和线性归一化两个特征
        proc_log_norm = (torch.log1p(proc) / torch.log1p(proc_scale)).nan_to_num(nan=0.0) * node_mask
        proc_lin_norm = (proc / proc_scale).nan_to_num(nan=0.0) * node_mask
        # 新增：job内剩余总处理时间（对job内所有未完成操作的处理时间求和再归一化）
        proc_2d = proc.reshape(flat_bt, self.max_jobs, self.max_machines)
        next_op_int_for_sum = torch.clamp(job_next_op.to(dtype=torch.long), min=0, max=self.max_machines)
        op_idx_grid = torch.arange(self.max_machines, device=device, dtype=torch.long).unsqueeze(0).unsqueeze(0).expand(flat_bt, self.max_jobs, -1)  # max_machines 已是容量上限，动态padding后始终匹配
        future_mask_2d = (op_idx_grid >= next_op_int_for_sum.unsqueeze(-1)).to(dtype=model_dtype)  # (flat_bt, J, M)
        job_rem_total = (proc_2d * future_mask_2d).sum(dim=-1)  # (flat_bt, J)
        job_rem_total_norm = job_rem_total / (job_rem_total.max(dim=1, keepdim=True).values.clamp(min=1.0))
        # 将job剩余总处理时间广播到节点维度
        job_rem_per_node = job_rem_total_norm[:, job_ids]  # (flat_bt, max_nodes)

        mach = machine_sequence.to(dtype=torch.long)
        mach = torch.clamp(mach, min=0, max=self.max_machines - 1)
        denom_mid = float(max(1, self.max_machines - 1))
        mach_norm = mach.to(dtype=model_dtype) / denom_mid

        avail = torch.gather(machine_availability, dim=1, index=mach)
        busy = torch.gather(machine_busy, dim=1, index=mach)

        denom_job = float(max(1, self.max_jobs - 1))
        denom_op = float(max(1, self.max_machines - 1))
        job_norm = job_ids.to(dtype=model_dtype).unsqueeze(0).expand(flat_bt, -1) / denom_job
        op_norm = op_ids.to(dtype=model_dtype).unsqueeze(0).expand(flat_bt, -1) / denom_op

        overall = overall_info.unsqueeze(1).expand(flat_bt, self.max_nodes, 3)

        # ── 机器队列位置特征（论文核心特征：该操作在同机器未完成操作中的排队位置） ──
        # 对每台机器，统计在该操作之前已有多少其他job的未完成操作在等待
        # queue_pos[b, n] = 同机器上在节点n之前（按job_id排序）的未完成操作数 / max_jobs
        # 实现：对每个节点，同机器所有未完成操作中 job_id < 本节点job_id 的数量
        mach_seq_int_for_queue = mach.clone()  # (flat_bt, max_nodes) 机器ID int
        # is_unfinished: is_next + is_future
        is_unfinished = (is_next + is_future).clamp(0, 1)  # (flat_bt, N)
        # 用scatter统计：对每台机器，计算在节点n之前（job_id更小）有多少未完成节点
        # 编码为 global_mach_id * max_jobs + job_id_of_node -> 累计
        # 简化：queue_pos[b,n] = sum_{n2 != n, mach[b,n2]==mach[b,n], job_id[n2]<job_id[n]} is_unfinished[b,n2]
        # 向量化实现：
        job_ids_bt = job_ids.unsqueeze(0).expand(flat_bt, -1).to(dtype=model_dtype)  # (flat_bt, N)
        mach_eq = (mach_seq_int_for_queue.unsqueeze(2) == mach_seq_int_for_queue.unsqueeze(1))  # (flat_bt, N, N)
        job_lt = (job_ids_bt.unsqueeze(2) < job_ids_bt.unsqueeze(1))  # (flat_bt, N, N): job_id[n2] < job_id[n]
        unfinished_bt = is_unfinished.unsqueeze(1).expand(flat_bt, self.max_nodes, self.max_nodes)  # (flat_bt, N, N)
        # queue_pos[b,n] = sum over n2 of (mach_eq[b,n,n2] & job_lt[b,n,n2] & unfinished[b,n2])
        # 但N*N*B可能很大（50*50*B=2500B），限制在max_nodes<=50时可接受
        if self.use_queue_position and self.max_nodes <= 100:  # 仅在节点数合理时用dense方式
            queue_pos = (mach_eq & job_lt & unfinished_bt.bool()).to(dtype=model_dtype).sum(dim=2)  # (flat_bt, N)
        else:
            queue_pos = torch.zeros(flat_bt, self.max_nodes, dtype=model_dtype, device=device)
        queue_pos_norm = (queue_pos / float(max(1, self.max_jobs - 1))).clamp(0, 1) * node_mask

        # ── LB（end time lower bound）特征 ──────────────────────────────────
        # LB[op] = max(前序op完成时间, 机器可用时间) + 本op处理时间
        # 用累积处理时间近似前序完成时间（不依赖动态调度状态）
        proc_2d_for_lb = proc.reshape(flat_bt, self.max_jobs, self.max_machines)
        # 每个op的前序累积处理时间（job内op_0到op_{k-1}的处理时间之和）
        cumsum_proc = torch.cumsum(proc_2d_for_lb, dim=-1)  # (flat_bt, J, M)
        # LB近似 = 前序累积时间 + 本op处理时间 = cumsum（包含自身）
        lb_2d = cumsum_proc  # (flat_bt, J, M)
        lb_flat = lb_2d.reshape(flat_bt, self.max_nodes)
        lb_scale = lb_flat.amax(dim=1, keepdim=True).clamp(min=1.0)
        lb_norm = (lb_flat / lb_scale).nan_to_num(nan=0.0) * node_mask  # (flat_bt, N)

        # 18维节点特征（17维 + 1维LB）
        node_feat = torch.stack(
            [
                proc_log_norm,      # 0: log归一化处理时间
                proc_lin_norm,      # 1: 线性归一化处理时间
                job_rem_per_node,   # 2: job内剩余总处理时间归一化
                mach_norm.to(dtype=model_dtype),  # 3: 机器ID归一化
                job_norm,           # 4: job ID归一化
                op_norm,            # 5: 操作序号归一化
                is_done,            # 6: 已完成标志
                is_next,            # 7: 当前待调度标志
                is_future,          # 8: 未来待处理标志
                rem_ops,            # 9: 剩余操作比例
                busy,               # 10: 机器忙碌状态
                avail,              # 11: 机器可用性
                node_mask,          # 12: 节点有效性掩码
                queue_pos_norm,     # 13: 机器队列位置
                lb_norm,            # 14: LB归一化（end time lower bound）
            ],
            dim=-1,
        )
        node_feat = torch.cat([node_feat, overall], dim=-1)  # 15+3=18维

        # 确保 node_in 与 model_dtype 一致（条件检查，避免每次forward都转换）
        if next(self.node_in.parameters()).dtype != model_dtype:
            self.node_in = self.node_in.to(dtype=model_dtype)
        h = torch.relu(self.node_in(node_feat.to(dtype=model_dtype)))
        h = h * node_mask.unsqueeze(-1).to(dtype=model_dtype)

        # ── 共轭边（静态，预计算） ──────────────────────────────────────────
        pre_src = self._pre_src.to(device=device)
        pre_dst = self._pre_dst.to(device=device)
        if pre_src.numel() > 0:
            offsets = (torch.arange(flat_bt, device=device, dtype=torch.long) * self.max_nodes).unsqueeze(1)
            pre_src_all = (pre_src.unsqueeze(0) + offsets).reshape(-1)
            pre_dst_all = (pre_dst.unsqueeze(0) + offsets).reshape(-1)
        else:
            pre_src_all = torch.zeros((0,), dtype=torch.long, device=device)
            pre_dst_all = torch.zeros((0,), dtype=torch.long, device=device)

        # ── 析取边（动态，全向量化实现，消除Python循环瓶颈） ────────────────
        # 向量化构建：对每台机器上所有未完成操作建全连接双向边
        # unfinished_mask: (flat_bt, max_nodes) bool
        unfinished_mask = (is_next + is_future).bool()  # (flat_bt, max_nodes)

        # machine_sequence: (flat_bt, max_nodes) int64，每个节点属于哪台机器
        mach_seq_int = machine_sequence.to(dtype=torch.long).clamp(0, self.max_machines - 1)  # (flat_bt, N)

        # 全局节点ID：batch偏移 + 节点偏移
        batch_node_offset = (torch.arange(flat_bt, device=device, dtype=torch.long) * self.max_nodes).unsqueeze(1)  # (flat_bt, 1)
        global_node_ids = batch_node_offset + torch.arange(self.max_nodes, device=device, dtype=torch.long).unsqueeze(0)  # (flat_bt, N)

        # 全局机器ID：batch偏移 + 机器ID，使不同batch的机器不混淆
        batch_mach_offset = (torch.arange(flat_bt, device=device, dtype=torch.long) * self.max_machines).unsqueeze(1)  # (flat_bt, 1)
        global_mach_ids = batch_mach_offset + mach_seq_int  # (flat_bt, N)

        # 只保留未完成节点
        valid = unfinished_mask  # (flat_bt, N)

        gm_flat = global_mach_ids.reshape(-1)   # (flat_bt*N,)
        gn_flat = global_node_ids.reshape(-1)   # (flat_bt*N,)
        v_flat  = valid.reshape(-1)             # (flat_bt*N,) bool

        # 只保留有效（未完成）节点
        valid_idx = v_flat.nonzero(as_tuple=False).squeeze(-1)  # (K,)
        if valid_idx.numel() >= 2:
            gm_valid = gm_flat[valid_idx]   # (K,) 全局机器ID
            gn_valid = gn_flat[valid_idx]   # (K,) 全局节点ID

            # ── 完全向量化的析取边构建（消除Python循环） ──────────────────
            # 思路：对每对有效节点(n_i, n_j)，若属于同一全局机器且 n_i != n_j，则建边
            # 用矩阵外积方式：(K, K) 的机器ID相等矩阵 & 自身不等
            # K = valid节点数，通常远小于 flat_bt * max_nodes
            K = gm_valid.shape[0]
            # 安全检查：K过大时（如批量较大）降级到分块处理避免OOM
            if K <= 2048:
                # 机器相等矩阵: (K, K)
                mach_eq_mat = (gm_valid.unsqueeze(1) == gm_valid.unsqueeze(0))  # (K, K)
                # 自身不等
                not_self = ~torch.eye(K, dtype=torch.bool, device=device)
                # 有效边掩码
                edge_mask = mach_eq_mat & not_self  # (K, K)
                if edge_mask.any():
                    src_idx, dst_idx = edge_mask.nonzero(as_tuple=True)  # 各 (E,)
                    disj_src_all = gn_valid[src_idx]
                    disj_dst_all = gn_valid[dst_idx]
                else:
                    disj_src_all = torch.zeros((0,), dtype=torch.long, device=device)
                    disj_dst_all = torch.zeros((0,), dtype=torch.long, device=device)
            else:
                # K 过大：按机器分组构建边（避免 K*K OOM）
                disj_src_list = []
                disj_dst_list = []
                unique_m = gm_valid.unique()
                for m_id in unique_m:
                    m_mask = (gm_valid == m_id)
                    grp = gn_valid[m_mask]
                    c = grp.shape[0]
                    if c >= 2:
                        idx_i = grp.repeat_interleave(c)
                        idx_j = grp.repeat(c)
                        ne = (idx_i != idx_j)
                        disj_src_list.append(idx_i[ne])
                        disj_dst_list.append(idx_j[ne])
                if disj_src_list:
                    disj_src_all = torch.cat(disj_src_list)
                    disj_dst_all = torch.cat(disj_dst_list)
                else:
                    disj_src_all = torch.zeros((0,), dtype=torch.long, device=device)
                    disj_dst_all = torch.zeros((0,), dtype=torch.long, device=device)
        else:
            disj_src_all = torch.zeros((0,), dtype=torch.long, device=device)
            disj_dst_all = torch.zeros((0,), dtype=torch.long, device=device)

        # ── 已排定机器顺序边：partial schedule 中真实形成的 machine-order 有向边 ──
        scheduled_adj = scheduled_machine_adj.reshape(flat_bt, self.max_nodes, self.max_nodes)
        valid_scheduled_adj = scheduled_adj > 0.5
        valid_scheduled_adj = valid_scheduled_adj & node_mask.bool().unsqueeze(2) & node_mask.bool().unsqueeze(1)
        if valid_scheduled_adj.any():
            sched_b, sched_src, sched_dst = valid_scheduled_adj.nonzero(as_tuple=True)
            sched_src_all = sched_b * self.max_nodes + sched_src
            sched_dst_all = sched_b * self.max_nodes + sched_dst
        else:
            sched_src_all = torch.zeros((0,), dtype=torch.long, device=device)
            sched_dst_all = torch.zeros((0,), dtype=torch.long, device=device)

        # ── GNN消息传播（GRU门控更新 + 向量级边类型门控） ─────────────────
        # dtype 统一只做一次性检查，不在每层循环中重复转换
        if next(self.node_in.parameters()).dtype != model_dtype:
            self.node_in = self.node_in.to(dtype=model_dtype)
        if next(self.job_token_proj.parameters()).dtype != model_dtype:
            self.job_token_proj = self.job_token_proj.to(dtype=model_dtype)
        if next(self.pre_msg[0].parameters()).dtype != model_dtype:
            self.pre_msg = self.pre_msg.to(dtype=model_dtype)
        if next(self.disj_msg[0].parameters()).dtype != model_dtype:
            self.disj_msg = self.disj_msg.to(dtype=model_dtype)
        if next(self.scheduled_msg[0].parameters()).dtype != model_dtype:
            self.scheduled_msg = self.scheduled_msg.to(dtype=model_dtype)
        if next(self.pre_gate_vec[0].parameters()).dtype != model_dtype:
            self.pre_gate_vec = self.pre_gate_vec.to(dtype=model_dtype)
        if next(self.disj_gate_vec[0].parameters()).dtype != model_dtype:
            self.disj_gate_vec = self.disj_gate_vec.to(dtype=model_dtype)
        if next(self.scheduled_gate_vec[0].parameters()).dtype != model_dtype:
            self.scheduled_gate_vec = self.scheduled_gate_vec.to(dtype=model_dtype)
        if next(self.gru_cells[0].parameters()).dtype != model_dtype:
            self.gru_cells = self.gru_cells.to(dtype=model_dtype)
        if next(self.norms[0].parameters()).dtype != model_dtype:
            self.norms = self.norms.to(dtype=model_dtype)
        h_flat = h.reshape(flat_bt * self.max_nodes, self.embedding_dim).to(dtype=model_dtype)
        for l in range(self.num_layers):
            # 共轭边：mean聚合（保序关系，度归一化）
            agg_pre = torch.zeros(h_flat.shape, dtype=model_dtype, device=device)
            deg_pre = torch.zeros((flat_bt * self.max_nodes, 1), dtype=model_dtype, device=device)
            if pre_src_all.numel() > 0:
                src_h = h_flat[pre_src_all]
                msg = self.pre_msg[l](src_h).to(dtype=model_dtype)
                gate = torch.sigmoid(self.pre_gate_vec[l](src_h).to(dtype=model_dtype))
                msg = msg * gate
                agg_pre.index_add_(0, pre_dst_all, msg)
                deg_pre.index_add_(0, pre_dst_all,
                                   torch.ones((pre_dst_all.numel(), 1), dtype=model_dtype, device=device))
            agg_pre = agg_pre / torch.clamp(deg_pre, min=1.0)

            # 析取边：sum聚合（竞争者数量本身是重要信息，不做度归一化）
            agg_disj = torch.zeros(h_flat.shape, dtype=model_dtype, device=device)
            if disj_src_all.numel() > 0:
                chunk_size = int(getattr(self, "disj_edge_chunk_size", 65536) or 65536)
                if disj_src_all.numel() <= chunk_size:
                    src_h2 = h_flat[disj_src_all]
                    msg2 = self.disj_msg[l](src_h2).to(dtype=model_dtype)
                    gate2 = torch.sigmoid(self.disj_gate_vec[l](src_h2).to(dtype=model_dtype))
                    msg2 = msg2 * gate2
                    agg_disj.index_add_(0, disj_dst_all, msg2)
                else:
                    for start in range(0, disj_src_all.numel(), chunk_size):
                        end = min(start + chunk_size, disj_src_all.numel())
                        src_idx = disj_src_all[start:end]
                        dst_idx = disj_dst_all[start:end]
                        src_h2 = h_flat[src_idx]
                        msg2 = self.disj_msg[l](src_h2).to(dtype=model_dtype)
                        gate2 = torch.sigmoid(self.disj_gate_vec[l](src_h2).to(dtype=model_dtype))
                        msg2 = msg2 * gate2
                        agg_disj.index_add_(0, dst_idx, msg2)

            # 已排定机器顺序边：mean聚合，记录已形成的同机器加工链。
            agg_sched = torch.zeros(h_flat.shape, dtype=model_dtype, device=device)
            deg_sched = torch.zeros((flat_bt * self.max_nodes, 1), dtype=model_dtype, device=device)
            if sched_src_all.numel() > 0:
                src_h3 = h_flat[sched_src_all]
                msg3 = self.scheduled_msg[l](src_h3).to(dtype=model_dtype)
                gate3 = torch.sigmoid(self.scheduled_gate_vec[l](src_h3).to(dtype=model_dtype))
                msg3 = msg3 * gate3
                agg_sched.index_add_(0, sched_dst_all, msg3)
                deg_sched.index_add_(0, sched_dst_all,
                                     torch.ones((sched_dst_all.numel(), 1), dtype=model_dtype, device=device))
            agg_sched = agg_sched / torch.clamp(deg_sched, min=1.0)

            agg = agg_pre + agg_disj + agg_sched

            # GRU门控更新
            h_flat = self.norms[l](self.gru_cells[l](agg, h_flat))
            # 屏蔽padding节点
            h_flat = h_flat * node_mask.reshape(flat_bt * self.max_nodes, 1).to(dtype=model_dtype)

        h = h_flat.reshape(flat_bt, self.max_nodes, self.embedding_dim)

        # ── Job-level tokens + cross-attention 聚合成单个 state 向量 ──────
        h_job = h.reshape(flat_bt, self.max_jobs, self.max_machines, self.embedding_dim)
        job_node_mask = node_mask.reshape(flat_bt, self.max_jobs, self.max_machines)
        job_denom = job_node_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        job_pool = (h_job * job_node_mask.unsqueeze(-1)).sum(dim=2) / job_denom  # (flat_bt, J, D)

        global_denom = node_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        global_pool = (h * node_mask.unsqueeze(-1)).sum(dim=1) / global_denom  # (flat_bt, D)

        global_exp = global_pool.unsqueeze(1).expand(flat_bt, self.max_jobs, self.embedding_dim)
        job_tokens = torch.cat([job_pool, global_exp], dim=-1)  # (flat_bt, J, 2D)
        job_valid_mask = (torch.arange(self.max_jobs, device=device).unsqueeze(0) < pn_jobs.to(dtype=torch.long))
        if next(self.job_token_proj.parameters()).dtype != model_dtype:
            self.job_token_proj = self.job_token_proj.to(dtype=model_dtype)
        job_tokens = torch.relu(self.job_token_proj(job_tokens.to(dtype=model_dtype)))  # (flat_bt, J, D)
        job_tokens = job_tokens * job_valid_mask.unsqueeze(-1).to(dtype=model_dtype)
        has_valid_job = job_valid_mask.any(dim=1, keepdim=True)  # (flat_bt, 1)
        safe_job_valid_mask = job_valid_mask.clone()
        if not bool(has_valid_job.all()):
            safe_job_valid_mask[~has_valid_job.squeeze(1), 0] = True

        # 用可学习 query 做 cross-attention，从 J 个 job token 聚合成 1 个向量
        # 这样输出仍是 (B, T, D)，Qwen 序列结构不变
        if next(self.job_attn.parameters()).dtype != model_dtype:
            self.job_attn = self.job_attn.to(dtype=model_dtype)
        if next(self.job_attn_norm.parameters()).dtype != model_dtype:
            self.job_attn_norm = self.job_attn_norm.to(dtype=model_dtype)
        query = self.job_query.expand(flat_bt, -1, -1).to(dtype=model_dtype)  # (flat_bt, 1, D)
        attn_out, _ = self.job_attn(
            query,
            job_tokens,
            job_tokens,
            key_padding_mask=~safe_job_valid_mask,
        )  # (flat_bt, 1, D)
        state_vec = self.job_attn_norm(attn_out.squeeze(1) + global_pool.to(dtype=model_dtype))  # (flat_bt, D)
        state_vec = state_vec * has_valid_job.to(dtype=model_dtype)
        if torch.isnan(state_vec).any():
            state_vec = torch.nan_to_num(state_vec, nan=0.0, posinf=0.0, neginf=0.0)

        job_tokens = job_tokens.reshape(batch_size, seq_len, self.max_jobs, self.embedding_dim)
        job_valid_mask = job_valid_mask.reshape(batch_size, seq_len, self.max_jobs)
        if torch.isnan(job_tokens).any():
            job_tokens = torch.nan_to_num(job_tokens, nan=0.0, posinf=0.0, neginf=0.0)

        aux = {
            "job_tokens": job_tokens,
            "job_valid_mask": job_valid_mask,
        }

        # 输出：(batch_size, seq_len, embedding_dim) — 与原来形状完全一致
        return state_vec.reshape(batch_size, seq_len, self.embedding_dim), aux


class JSSPStateEncoder(nn.Module):
    """Simplified and efficient encoder for JSSP states with compact representation."""

    def __init__(self, state_dims, embedding_dim=None, hidden_size=None):
        """
        Initialize the simplified JSSP state encoder.

        Args:
            state_dims: Dictionary containing dimensions of different parts of the state
            embedding_dim: Output embedding dimension (deprecated, use hidden_size instead)
            hidden_size: Output dimension that matches the transformer's hidden_size
        """
        super().__init__()

        # If hidden_size is provided, use it; otherwise use embedding_dim (with deprecation warning)
        if hidden_size is not None:
            self.embedding_dim = hidden_size
        elif embedding_dim is not None:
            self.embedding_dim = embedding_dim
            print("Warning: JSSPStateEncoder's embedding_dim parameter is deprecated. Use hidden_size instead.")
        else:
            self.embedding_dim = 3584  # Default to Qwen2.5-7B's hidden_size
            print("Warning: No embedding dimension specified for JSSPStateEncoder. Using default hidden_size=3584.")

        # Get dimensions from state_dims
        self.max_jobs = state_dims.get('n_jobs', 10)
        self.max_machines = state_dims.get('n_machines', 10)

        # 为每个作业创建独立的编码器，保留作业结构信息
        self.job_encoders = nn.ModuleList()
        # 计算每个作业编码器的输出维度，确保总和等于embedding_dim
        if self.max_jobs > 0:
            base_job_dim = self.embedding_dim // self.max_jobs
            extra_job_dim = self.embedding_dim % self.max_jobs

            for i in range(self.max_jobs):
                # 前extra_job_dim个作业多分配一个维度
                per_job_dim = base_job_dim + (1 if i < extra_job_dim else 0)
                self.job_encoders.append(nn.Linear(2, per_job_dim))

        # 为每个机器创建独立的编码器，保留机器结构信息
        self.machine_encoders = nn.ModuleList()
        # 计算每个机器编码器的输出维度，确保总和等于embedding_dim
        if self.max_machines > 0:
            base_machine_dim = self.embedding_dim // self.max_machines
            extra_machine_dim = self.embedding_dim % self.max_machines

            for i in range(self.max_machines):
                # 前extra_machine_dim个机器多分配一个维度
                per_machine_dim = base_machine_dim + (1 if i < extra_machine_dim else 0)
                self.machine_encoders.append(nn.Linear(2, per_machine_dim))

        # 作业-机器兼容性编码器，保留作业和机器之间的关系
        self.job_machine_compatibility_encoder = nn.Linear(self.max_jobs * self.max_machines, self.embedding_dim)

        # 剩余处理时间编码器（结构化保留）
        self.remaining_processing_encoder = nn.Linear(self.max_jobs * self.max_machines, self.embedding_dim)

        # 整体状态编码器
        self.overall_info_encoder = nn.Linear(3, self.embedding_dim)

        # 作业-机器交互模块 - 轻量级cross-attention机制
        class JobMachineInteraction(nn.Module):
            def __init__(self, hidden_size, max_jobs, max_machines):
                super().__init__()
                self.max_jobs = max_jobs
                self.max_machines = max_machines
                self.hidden_size = hidden_size

                # 投影层用于将作业和机器嵌入映射到固定维度
                self.job_proj = nn.Linear(hidden_size, hidden_size * max_jobs)
                self.machine_proj = nn.Linear(hidden_size, hidden_size * max_machines)

                # 交叉注意力
                self.attn = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)

                # 输出处理
                self.output_proj = nn.Linear(hidden_size, hidden_size)
                self.layer_norm = nn.LayerNorm(hidden_size)

            def forward(self, job_emb, machine_emb):
                batch_size = job_emb.shape[0]

                # 使用线性投影将作业嵌入转换为 (B, J, D) 格式
                job_emb_proj = self.job_proj(job_emb)  # (B, J*D)
                job_emb_reshaped = job_emb_proj.view(batch_size, self.max_jobs, self.hidden_size)  # (B, J, D)

                # 使用线性投影将机器嵌入转换为 (B, M, D) 格式
                machine_emb_proj = self.machine_proj(machine_emb)  # (B, M*D)
                machine_emb_reshaped = machine_emb_proj.view(batch_size, self.max_machines, self.hidden_size)  # (B, M, D)

                # 交叉注意力
                attn_out, _ = self.attn(job_emb_reshaped, machine_emb_reshaped, machine_emb_reshaped)  # (B, J, D)

                # 输出投影和归一化
                output = self.layer_norm(attn_out + job_emb_reshaped)
                output = self.output_proj(output)  # (B, J, D)

                # 压缩为 (B, D)
                return output.mean(dim=1)

        self.job_machine_interaction = JobMachineInteraction(self.embedding_dim, self.max_jobs, self.max_machines)

        # 最终投影层 - 6个组件：作业编码、机器编码、兼容性编码、剩余处理时间编码、整体信息编码、作业-机器交互编码
        self.final_proj = nn.Linear(6 * self.embedding_dim, self.embedding_dim)

        # 扁平向量处理层 - 动态创建
        self.flat_state_proj = None

        self.layer_norm = nn.LayerNorm(self.embedding_dim)

    def _safe_slice(self, x, target_dim, keepdim=False):
        """
        Safely slice a tensor along the last dimension with shape handling.

        Args:
            x: Input tensor
            target_dim: Target dimension to slice to
            keepdim: Whether to keep the dimension when slicing

        Returns:
            Sliced tensor
        """
        if x.dim() == 3:
            # 处理(B, T, D)形状
            return x[..., :target_dim]
        elif x.dim() == 2:
            # 处理(B, D)形状
            return x[:, :target_dim]
        elif x.dim() == 1:
            # 处理(D,)形状
            return x[:target_dim].unsqueeze(0) if keepdim else x[:target_dim]
        else:
            raise ValueError(f"Unexpected tensor dim: {x.dim()}")

    def forward(self, state):
        """
        Encode a simplified JSSP state into an embedding.

        Args:
            state: Either a dictionary containing compact JSSP state components, or a tensor
                  Tensor format should be (B, L, D) for sequence data or (B, D) for batch data

        Returns:
            Embedded state: (B, L, E) if sequence data, (B, E) otherwise
        """
        # Handle tensor input (主要用于DecisionTransformer的轨迹输入)
        if isinstance(state, torch.Tensor):
            if state.dim() == 4:
                # 处理4D输入: (B, L, J, M)
                batch_size, seq_length, jobs, machines = state.shape
                # 展平为 (B*L, J*M)
                state = state.view(batch_size * seq_length, jobs * machines)

                # 确保扁平投影层与输入维度匹配
                flattened_dim = state.shape[1]
                if self.flat_state_proj is None or self.flat_state_proj.in_features != flattened_dim:
                    # 创建时使用与state相同的dtype
                    self.flat_state_proj = nn.Linear(flattened_dim, self.embedding_dim).to(state.device).to(dtype=state.dtype)
                    print(f"flat_state_proj dtype设置为: {self.flat_state_proj.weight.dtype}")

                # 编码展平后的状态
                state = state.to(dtype=self.flat_state_proj.weight.dtype)
                embedded = self.flat_state_proj(state)  # (B*L, E)
                # 恢复维度为 (B, L, E)
                return embedded.view(batch_size, seq_length, -1)
            elif state.dim() == 3:
                # 处理序列数据: (B, L, D)
                batch_size, seq_length, flattened_dim = state.shape
                embedded = []

                # 确保扁平投影层与输入维度匹配
                if self.flat_state_proj is None or self.flat_state_proj.in_features != flattened_dim:
                    # 创建时使用与state相同的dtype
                    self.flat_state_proj = nn.Linear(flattened_dim, self.embedding_dim).to(state.device).to(dtype=state.dtype)
                    print(f"flat_state_proj dtype设置为: {self.flat_state_proj.weight.dtype}")

                # 处理每个时间步
                for i in range(seq_length):
                    state_step = state[:, i, :]  # (B, D)
                    # 确保输入数据与flat_state_proj层的dtype一致
                    state_step = state_step.to(dtype=self.flat_state_proj.weight.dtype)
                    embedded_step = self.flat_state_proj(state_step)  # (B, E)
                    # 不再手动转换为float16 - 让autocast自动管理混合精度
                    embedded.append(embedded_step.unsqueeze(1))  # (B, 1, E)

                return torch.cat(embedded, dim=1)  # (B, L, E)
            elif state.dim() == 2:
                # 处理批量数据: (B, D)
                if self.flat_state_proj is None or self.flat_state_proj.in_features != state.shape[1]:
                    # 创建时使用与state相同的dtype
                    self.flat_state_proj = nn.Linear(state.shape[1], self.embedding_dim).to(state.device).to(dtype=state.dtype)
                    print(f"flat_state_proj dtype设置为: {self.flat_state_proj.weight.dtype}")

                # 确保输入数据与flat_state_proj层的dtype一致
                state = state.to(dtype=self.flat_state_proj.weight.dtype)
                embedded = self.flat_state_proj(state)  # (B, E)
                # 不再手动转换为float16 - 让autocast自动管理混合精度
                return embedded  # (B, E)
            else:
                raise ValueError(f"State tensor must have 2, 3, or 4 dimensions, got {state.dim()}")

        # Dictionary-based state processing
        if not isinstance(state, dict):
            raise TypeError(f"Expected state to be a dictionary or tensor, got {type(state)}")

        # 验证必要的键并进行向后兼容处理
        required_keys = ['job_next_op', 'normalized_machine_availability', 'overall_info']
        enhanced_keys = ['remaining_processing_times', 'machine_busy_mask', 'job_completion_mask']

        # 提取所有状态特征
        if all(key in state for key in required_keys):
            # 基础状态特征
            job_status = state['job_next_op']
            machine_status = state['normalized_machine_availability']
            overall_info = state['overall_info']

            # 增强状态特征（如果存在）
            remaining_processing = state.get('remaining_processing_times', None)
            machine_busy = state.get('machine_busy_mask', None)
            job_completion = state.get('job_completion_mask', None)
            problem_n_jobs = state.get('problem_n_jobs', None)
            problem_n_machines = state.get('problem_n_machines', None)
        else:
            # 新格式状态
            job_status = state['job_status']
            machine_status = state['machine_status']
            overall_info = state['overall_info']

            # 增强状态特征（如果存在）
            remaining_processing = state.get('remaining_processing_times', None)
            machine_busy = state.get('machine_busy_mask', None)
            job_completion = state.get('job_completion_mask', None)
            problem_n_jobs = state.get('problem_n_jobs', None)
            problem_n_machines = state.get('problem_n_machines', None)

        # 转换numpy数组为tensor并移动到正确的设备和dtype
        device = next(self.parameters()).device  # 获取模型当前设备
        model_dtype = next(self.parameters()).dtype  # 获取模型当前dtype

        # --- 【修复】处理字典输入，确保正确识别batch和seq维度 ---
        # 从第一个状态组件推断batch_size和seq_length
        has_seq_dim = False
        batch_size = 1
        seq_length = 1

        # 检查第一个组件的维度
        sample_component = job_status
        if isinstance(sample_component, torch.Tensor):
            if sample_component.dim() == 3:
                # 形状: (B, T, D) 或 (B, D1, D2)
                batch_size, seq_length, _ = sample_component.shape
                has_seq_dim = True
            elif sample_component.dim() == 2:
                # 形状: (B, D)
                batch_size, _ = sample_component.shape
            elif sample_component.dim() == 1:
                # 形状: (D)
                batch_size = 1

        debug_prints = bool(getattr(self, "debug_prints", False))
        if debug_prints:
            print(f"[DEBUG] JSSPStateEncoder: Initial batch_size={batch_size}, seq_length={seq_length}, has_seq_dim={has_seq_dim}")

        # 转换所有组件到正确的device和dtype
        def convert_component(comp):
            if isinstance(comp, np.ndarray):
                comp = torch.tensor(comp, dtype=model_dtype).to(device)
            else:
                comp = comp.to(device).to(dtype=model_dtype)
            return comp

        job_status = convert_component(job_status)
        machine_status = convert_component(machine_status)
        overall_info = convert_component(overall_info)

        # 处理增强特征
        remaining_processing = convert_component(remaining_processing) if remaining_processing is not None else None
        machine_busy = convert_component(machine_busy) if machine_busy is not None else None
        job_completion = convert_component(job_completion) if job_completion is not None else None
        problem_n_jobs = convert_component(problem_n_jobs) if problem_n_jobs is not None else None
        problem_n_machines = convert_component(problem_n_machines) if problem_n_machines is not None else None

        # --- 处理job_status ---
        if job_status.dim() == 1:
            # 只有features维度，添加batch和seq维度
            job_status = job_status.unsqueeze(0).unsqueeze(1)  # (1, 1, D)
        elif job_status.dim() == 2:
            # 有batch但没有seq，添加seq维度
            job_status = job_status.unsqueeze(1)  # (B, 1, D)
        elif job_status.dim() == 3:
            # 已经有batch和seq维度，保持不变
            batch_size, seq_length, _ = job_status.shape
            has_seq_dim = True

        # --- 处理machine_status ---
        if machine_status.dim() == 1:
            machine_status = machine_status.unsqueeze(0).unsqueeze(1)  # (1, 1, D)
        elif machine_status.dim() == 2:
            machine_status = machine_status.unsqueeze(1)  # (B, 1, D)
        elif machine_status.dim() == 3:
            batch_size, seq_length, _ = machine_status.shape
            has_seq_dim = True

        # --- 处理machine_busy ---
        if machine_busy is not None:
            if machine_busy.dim() == 1:
                machine_busy = machine_busy.unsqueeze(0).unsqueeze(1)  # (1, 1, D)
            elif machine_busy.dim() == 2:
                machine_busy = machine_busy.unsqueeze(1)  # (B, 1, D)
            elif machine_busy.dim() == 3:
                batch_size, seq_length, _ = machine_busy.shape
                has_seq_dim = True

        # --- 处理overall_info ---
        if overall_info.dim() == 1:
            overall_info = overall_info.unsqueeze(0).unsqueeze(1)  # (1, 1, D)
        elif overall_info.dim() == 2:
            overall_info = overall_info.unsqueeze(1)  # (B, 1, D)
        elif overall_info.dim() == 3:
            batch_size, seq_length, _ = overall_info.shape
            has_seq_dim = True

        if problem_n_jobs is None:
            problem_n_jobs = torch.full((batch_size, seq_length, 1), float(self.max_jobs), device=device, dtype=model_dtype)
        if problem_n_machines is None:
            problem_n_machines = torch.full((batch_size, seq_length, 1), float(self.max_machines), device=device, dtype=model_dtype)

        if problem_n_jobs.dim() == 1:
            problem_n_jobs = problem_n_jobs.unsqueeze(0).unsqueeze(1)
        elif problem_n_jobs.dim() == 2:
            problem_n_jobs = problem_n_jobs.unsqueeze(1)
        elif problem_n_jobs.dim() == 3:
            pass

        if problem_n_machines.dim() == 1:
            problem_n_machines = problem_n_machines.unsqueeze(0).unsqueeze(1)
        elif problem_n_machines.dim() == 2:
            problem_n_machines = problem_n_machines.unsqueeze(1)
        elif problem_n_machines.dim() == 3:
            pass

        if debug_prints:
            print(f"[DEBUG] JSSPStateEncoder: After processing components - batch_size={batch_size}, seq_length={seq_length}, has_seq_dim={has_seq_dim}")

        # --- 处理剩余处理时间 ---
        if remaining_processing is not None:
            if remaining_processing.dim() == 1:
                remaining_processing = remaining_processing.unsqueeze(0).unsqueeze(1)  # (1, 1, D)
            elif remaining_processing.dim() == 2:
                remaining_processing = remaining_processing.unsqueeze(1)  # (B, 1, D)
            elif remaining_processing.dim() == 3:
                # 可能是 (B, jobs, machines) 或 (B, seq, jobs*machines)
                if remaining_processing.shape[2] == self.max_jobs * self.max_machines:
                    # 已经是展平的特征，直接处理
                    batch_size, seq_length, features = remaining_processing.shape
                    has_seq_dim = True
                else:
                    # 形状: (B, jobs, machines) → 展平
                    batch_size, jobs, machines = remaining_processing.shape
                    remaining_processing = remaining_processing.view(batch_size, 1, jobs * machines)  # (B, 1, jobs*machines)
            elif remaining_processing.dim() == 4:
                # 形状: (B, seq, jobs, machines)
                batch_size, seq_length, jobs, machines = remaining_processing.shape
                remaining_processing = remaining_processing.view(batch_size, seq_length, jobs * machines)
                has_seq_dim = True

        # --- 处理作业完成掩码 ---
        if job_completion is not None:
            if job_completion.dim() == 1:
                job_completion = job_completion.unsqueeze(0).unsqueeze(1)  # (1, 1, D)
            elif job_completion.dim() == 2:
                job_completion = job_completion.unsqueeze(1)  # (B, 1, D)
            elif job_completion.dim() == 3:
                # 可能是 (B, jobs, machines) 或 (B, seq, jobs*machines)
                if job_completion.shape[2] == self.max_jobs * self.max_machines:
                    # 已经是展平的特征，直接处理
                    batch_size, seq_length, features = job_completion.shape
                    has_seq_dim = True
                else:
                    # 形状: (B, jobs, machines) → 展平
                    batch_size, jobs, machines = job_completion.shape
                    job_completion = job_completion.view(batch_size, 1, jobs * machines)  # (B, 1, jobs*machines)

        # --- 合并所有组件的batch和seq维度 ---
        # 保存原始的batch_size和seq_length
        original_batch_size = batch_size
        original_seq_length = seq_length

        # 将所有组件的batch和seq维度合并为一个维度，便于后续处理
        def flatten_batch_seq(comp):
            if comp.dim() == 3:
                # (B, T, D) → (B*T, D)
                return comp.view(-1, comp.shape[-1]), True
            elif comp.dim() == 2:
                # (B, D) → (B, D)
                return comp, False
            else:
                raise ValueError(f"Unsupported dimension {comp.dim()}")

        job_status_flat, job_has_seq = flatten_batch_seq(job_status)
        machine_status_flat, machine_has_seq = flatten_batch_seq(machine_status)
        overall_info_flat, overall_has_seq = flatten_batch_seq(overall_info)
        problem_n_jobs_flat, _ = flatten_batch_seq(problem_n_jobs)
        problem_n_machines_flat, _ = flatten_batch_seq(problem_n_machines)

        # 更新has_seq_dim
        has_seq_dim = has_seq_dim or job_has_seq or machine_has_seq or overall_has_seq
        if debug_prints:
            print(f"[DEBUG] JSSPStateEncoder: After flattening, has_seq_dim={has_seq_dim}")

        batch_size_flat = job_status_flat.shape[0]
        if debug_prints:
            print(f"[DEBUG] JSSPStateEncoder: batch_size_flat={batch_size_flat}")

        # 执行编码 - 使用新的结构化编码方式

        # 1. 编码每个作业的特征 - 保留作业结构信息
        job_embs = []
        # 使用job_status作为作业下一个操作索引信息
        job_next_op = job_status_flat

        for i in range(self.max_jobs):
            # 获取作业i的特征：下一个操作索引 + 是否已完成
            if job_next_op.dim() > 1 and i < job_next_op.shape[1]:
                valid_job = (problem_n_jobs_flat[:, 0] > float(i)).unsqueeze(1)
                next_op = job_next_op[:, i].unsqueeze(1) * valid_job.to(dtype=model_dtype)
                is_completed = ((next_op >= problem_n_machines_flat) & valid_job).to(dtype=model_dtype)
                job_feature = torch.cat([next_op, is_completed], dim=1)
            else:
                # 填充超出部分
                job_feature = torch.zeros(batch_size_flat, 2, device=device, dtype=model_dtype)

            # 使用作业专用编码器
            job_emb = self.job_encoders[i](job_feature)
            job_embs.append(job_emb)

        # 合并所有作业编码
        job_emb = torch.cat(job_embs, dim=1)

        # 2. 编码每个机器的特征 - 保留机器结构信息
        machine_embs = []
        # 使用machine_status作为机器可用性信息
        machine_busy_flat = None
        if machine_busy is not None:
            machine_busy_flat, _ = flatten_batch_seq(machine_busy)
        for i in range(self.max_machines):
            # 获取机器i的特征：可用性 + 忙状态
            if machine_status_flat.dim() > 1 and i < machine_status_flat.shape[1]:
                # machine_status包含机器可用性信息
                valid_machine = (problem_n_machines_flat[:, 0] > float(i)).unsqueeze(1)
                availability_raw = machine_status_flat[:, i].unsqueeze(1)
                availability = availability_raw * valid_machine.to(dtype=model_dtype)
                # 如果有machine_busy信息，使用它；否则根据可用性计算
                if machine_busy_flat is not None:
                    if machine_busy_flat.dim() > 1 and i < machine_busy_flat.shape[1]:
                        busy_raw = machine_busy_flat[:, i].unsqueeze(1)
                    else:
                        busy_raw = (1 - availability_raw).unsqueeze(1)
                else:
                    busy_raw = (1 - availability_raw).unsqueeze(1)
                busy_status = busy_raw.to(dtype=model_dtype) * valid_machine.to(dtype=model_dtype)
                machine_feature = torch.cat([availability, busy_status], dim=1)
            else:
                # 填充超出部分
                machine_feature = torch.zeros(batch_size_flat, 2, device=device, dtype=model_dtype)

            # 使用机器专用编码器
            machine_emb = self.machine_encoders[i](machine_feature)
            machine_embs.append(machine_emb)

        # 合并所有机器编码
        machine_emb = torch.cat(machine_embs, dim=1)

        # 3. 编码作业-机器兼容性 - 保留拓扑关系
        compatibility_feature = None
        if job_completion is not None:
            job_completion_flat, _ = flatten_batch_seq(job_completion)
            compatibility_feature = job_completion_flat
        else:
            # 创建默认兼容性特征
            compatibility_feature = torch.zeros(batch_size_flat, self.max_jobs * self.max_machines, device=device, dtype=model_dtype)

        # 确保兼容性特征与编码器期望的维度匹配
        expected_dim = self.max_jobs * self.max_machines
        current_dim = compatibility_feature.shape[1]
        if current_dim < expected_dim:
            # 填充到期望的维度
            padding = torch.zeros(batch_size_flat, expected_dim - current_dim, device=device, dtype=model_dtype)
            compatibility_feature = torch.cat([compatibility_feature, padding], dim=1)
        elif current_dim > expected_dim:
            # 裁剪到期望的维度
            compatibility_feature = compatibility_feature[:, :expected_dim]

        job_machine_compatibility_emb = self.job_machine_compatibility_encoder(compatibility_feature)

        # 4. 编码剩余处理时间
        if remaining_processing is not None:
            remaining_processing_flat, _ = flatten_batch_seq(remaining_processing)

            # 确保剩余处理时间特征与编码器期望的维度匹配
            expected_dim = self.max_jobs * self.max_machines
            current_dim = remaining_processing_flat.shape[1]

            if current_dim < expected_dim:
                # 填充到期望的维度
                padding = torch.zeros(batch_size_flat, expected_dim - current_dim, device=device, dtype=model_dtype)
                remaining_processing_flat = torch.cat([remaining_processing_flat, padding], dim=1)
            elif current_dim > expected_dim:
                # 裁剪到期望的维度
                remaining_processing_flat = remaining_processing_flat[:, :expected_dim]

            remaining_processing_emb = self.remaining_processing_encoder(remaining_processing_flat)
        else:
            remaining_processing_emb = torch.zeros(batch_size_flat, self.embedding_dim, device=device, dtype=model_dtype)

        # 5. 编码整体状态信息
        overall_emb = self.overall_info_encoder(overall_info_flat)

        # 确保所有编码都有正确的形状
        if job_emb.dim() > 2:
            job_emb = job_emb.mean(dim=1)
        if machine_emb.dim() > 2:
            machine_emb = machine_emb.mean(dim=1)
        if job_machine_compatibility_emb.dim() > 2:
            job_machine_compatibility_emb = job_machine_compatibility_emb.mean(dim=1)
        if remaining_processing_emb.dim() > 2:
            remaining_processing_emb = remaining_processing_emb.mean(dim=1)
        if overall_emb.dim() > 2:
            overall_emb = overall_emb.mean(dim=1)

        # 作业-机器交互编码
        job_machine_interaction_emb = self.job_machine_interaction(job_emb, machine_emb)

        # 合并所有编码 - 6个组件，保留结构化信息
        combined = torch.cat([
            job_emb,
            machine_emb,
            job_machine_compatibility_emb,
            remaining_processing_emb,
            overall_emb,
            job_machine_interaction_emb
        ], dim=1)

        # 最终投影和归一化
        embedded = self.layer_norm(self.final_proj(combined))

        # 如果有序列维度，恢复序列维度
        if has_seq_dim:
            # embedded 当前形状: (B*T, embedding_dim)
            # 需要恢复为: (B, T, embedding_dim)
            embedded = embedded.view(original_batch_size, original_seq_length, -1)
            if debug_prints:
                print(f"[DEBUG] JSSPStateEncoder: Restored sequence dimension, output shape: {embedded.shape}")
        else:
            if debug_prints:
                print(f"[DEBUG] JSSPStateEncoder: No sequence dimension, output shape: {embedded.shape}")

        return embedded


class JSSPActionDecoder(nn.Module):
    """Decoder for converting transformer outputs into JSSP actions."""

    def __init__(self, embedding_dim=None, hidden_size=None, n_jobs=10, n_machines=10, dtype=torch.float32):
        """
        Initialize the JSSP action decoder.

        Args:
            embedding_dim: Input embedding dimension (deprecated, use hidden_size instead)
            hidden_size: Input dimension that matches the transformer's hidden_size
            n_jobs: Number of jobs
            n_machines: Number of machines
            dtype: Data type to use for linear layers
        """
        super().__init__()

        # If hidden_size is provided, use it; otherwise use embedding_dim (with deprecation warning)
        if hidden_size is not None:
            self.input_dim = hidden_size
        elif embedding_dim is not None:
            self.input_dim = embedding_dim
            print("Warning: JSSPActionDecoder's embedding_dim parameter is deprecated. Use hidden_size instead.")
        else:
            self.input_dim = 3584  # Default to Qwen2.5-7B's hidden_size
            print("Warning: No input dimension specified for JSSPActionDecoder. Using default hidden_size=3584.")

        self.n_jobs = n_jobs
        self.n_machines = n_machines

        # Decode to job-machine assignment logits
        self.logits_proj = nn.Linear(self.input_dim, n_jobs * n_machines, dtype=dtype)

    def forward(self, embedding):
        """
        Decode embedding to action logits.

        Args:
            embedding: Transformer output embedding

        Returns:
            Action logits reshaped as (batch, seq_len, n_jobs, n_machines)
        """
        # 添加数值稳定性措施：裁剪输入嵌入
        embedding = torch.clamp(embedding, min=-1e4, max=1e4)
        logits = self.logits_proj(embedding)
        # 添加数值稳定性措施：裁剪logits
        logits = torch.clamp(logits, min=-1e4, max=1e4)
        # Keep batch and sequence dimensions separate
        if embedding.dim() == 3:  # (B, T, H)
            batch_size, seq_len, _ = embedding.shape
            logits = logits.view(batch_size, seq_len, self.n_jobs, self.n_machines)
        elif embedding.dim() == 2:  # (B, H)
            batch_size, _ = embedding.shape
            logits = logits.view(batch_size, self.n_jobs, self.n_machines)
        else:
            # Fallback for unexpected dimensions
            logits = logits.view(-1, self.n_jobs, self.n_machines)
        return logits

    def sample_action(self, embedding, deterministic=False):
        """
        Sample an action from the decoder outputs.

        Args:
            embedding: Transformer output embedding
            deterministic: Whether to use deterministic selection

        Returns:
            Tuple of (job, machine) indices
        """
        logits = self.forward(embedding)

        if deterministic:
            # Greedy selection
            job_machine_idx = torch.argmax(logits.view(logits.shape[0], -1), dim=-1)
        else:
            # Sampling
            probs = torch.softmax(logits.view(logits.shape[0], -1), dim=-1)
            job_machine_idx = torch.multinomial(probs, 1).squeeze(-1)
        # Convert flat index to (job, machine)
        jobs = job_machine_idx // self.n_machines
        machines = job_machine_idx % self.n_machines
        
        return jobs, machines
