# Third-party notices

## LLMCoSolver

The initial implementation was developed from LLMCoSolver, the code for *Large Language Models as End-to-end Combinatorial Optimization Solvers* (NeurIPS 2025), by Xia Jiang, Yaoxin Wu, Minshuo Li, Zhiguang Cao and Yingqian Zhang.

Source: https://github.com/Summer142857/LLMCoSolver

The upstream project uses the MIT License, copyright (c) 2025 Xia Jiang. Its notice is included in `LICENSES/LLMCoSolver-MIT.txt` and applies to inherited portions. Retain that notice when redistributing inherited code. The DecLLM model, graph/state/action interfaces and training adaptations are modifications to the starting implementation; their presence does not erase upstream authorship.

## Qwen2.5-7B-Instruct

The base model is provided separately by the Qwen team:

https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

Its upstream license is Apache-2.0. The complete base weights are not included in this source repository or the checkpoint asset. The release contains trained adapter parameters and custom scheduling modules. It includes a copy of the base-model license for reference and preservation of applicable upstream terms. The code's MIT License does not replace third-party model terms.

The checkpoint modifies the base model's behavior through LoRA and adds scheduling modules; original base weights themselves are not merged or reserialized in this release.

## Dependencies and reference implementations

PyTorch, Transformers, PEFT, bitsandbytes, OR-Tools and other dependencies retain their own licenses. Referenced baseline repositories are not claimed as DecLLM contributions. Retain any notices present in source files when redistributing them.
