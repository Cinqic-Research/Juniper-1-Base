import torch

from juniper1.generate import GenerationSettings, continue_ids
from juniper1.model import FROZEN_CONFIG, FROZEN_PARAMETERS, ModelConfig, Transformer, analytic_params
from juniper1.verify import tensor_tree_sha

TINY = {**FROZEN_CONFIG, "vocab_size": 512, "n_layer": 2, "d_model": 32, "n_head": 4, "n_kv_head": 2, "head_dim": 8,
        "d_ff": 64, "max_seq_len": 64}  # fmt: skip


def test_frozen_config_parameter_count_analytic():
    assert analytic_params(ModelConfig(**FROZEN_CONFIG)) == FROZEN_PARAMETERS == 142_771_456


def test_default_config_is_frozen():
    assert ModelConfig().to_dict() == FROZEN_CONFIG


def test_tiny_model_kv_cache_matches_full_forward():
    torch.manual_seed(0)
    m = Transformer(ModelConfig(**TINY)).eval()
    assert m.num_params() == analytic_params(m.cfg)
    idx = torch.randint(0, 512, (1, 20))
    with torch.no_grad():
        full = m(idx)[0]
        caches = [[] for _ in m.blocks]
        inc = [m.step_logits(idx[:, :1], caches, 0)[0]]
        for t in range(1, 20):
            inc.append(m.step_logits(idx[:, t : t + 1], caches, t)[0])
    inc_t = torch.stack(inc)
    assert torch.allclose(full, inc_t, atol=1e-4), (full - inc_t).abs().max()


def test_context_limit_enforced():
    m = Transformer(ModelConfig(**TINY)).eval()
    try:
        m(torch.zeros(1, 65, dtype=torch.long))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_generation_is_deterministic_and_masks_controls():
    torch.manual_seed(1)
    m = Transformer(ModelConfig(**TINY)).eval()
    s = GenerationSettings(max_new_tokens=16, temperature=0.8, top_k=20, seed=7, stop_at_eos=False)
    a, _ = continue_ids(m, [5, 6, 7], s)
    b, _ = continue_ids(m, [5, 6, 7], s)
    assert a == b and len(a) == 16
    assert not any(1 <= t < 64 for t in a)
    g, _ = continue_ids(m, [5, 6, 7], GenerationSettings(max_new_tokens=8))
    assert not any(1 <= t < 64 for t in g)


def test_tensor_tree_sha_is_order_independent_and_sensitive():
    sd = {"b": torch.ones(2), "a": torch.zeros(3)}
    assert tensor_tree_sha(sd) == tensor_tree_sha({"a": torch.zeros(3), "b": torch.ones(2)})
    assert tensor_tree_sha(sd) != tensor_tree_sha({"a": torch.zeros(3), "b": torch.ones(2) * 2})
