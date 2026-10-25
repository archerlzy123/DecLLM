import inspect
import math

import torch

import dt_infer as infer


def draw(logits, top_p, n=1000, seed=123):
    torch.manual_seed(seed)
    return {
        infer._sample_candidate_position(logits, temperature=1.0, top_p=top_p)
        for _ in range(n)
    }


def old_sample_candidate_position(candidate_logits, temperature):
    probs = torch.softmax(candidate_logits / float(temperature), dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def main():
    probs = torch.tensor([[0.42, 0.25, 0.15, 0.08, 0.05, 0.03, 0.02]])
    logits = torch.log(probs)
    assert draw(logits, top_p=0.7) == {0, 1, 2}

    confident_probs = torch.tensor([[0.82, 0.08, 0.05, 0.03, 0.02]])
    assert draw(torch.log(confident_probs), top_p=0.7) == {0}

    masked_logits = torch.tensor([[0.0, -math.inf, -0.4, -math.inf]])
    assert draw(masked_logits, top_p=1.0, n=2000).issubset({0, 2})

    torch.manual_seed(20260715)
    samples_new = [infer._sample_candidate_position(logits, 0.7, 0.0) for _ in range(100)]
    torch.manual_seed(20260715)
    samples_old = [old_sample_candidate_position(logits, 0.7) for _ in range(100)]
    assert samples_new == samples_old

    route_names = [
        "rollout_episode",
        "rollout_episode_with_trajectory",
        "test_model_inference",
        "evaluate_multiple_instances",
        "evaluate_instances_from_jsonl",
    ]
    for route_name in route_names:
        assert inspect.signature(getattr(infer, route_name)).parameters["top_p"].default == 0.0

    print("PASS: deployed top-p implementation")


if __name__ == "__main__":
    main()
