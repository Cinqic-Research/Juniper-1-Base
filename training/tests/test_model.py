import torch

from j1bench.model import ModelConfig, Transformer

TINY = ModelConfig(vocab_size=97, n_layer=2, d_model=32, n_head=4, n_kv_head=2, head_dim=8, d_ff=48, max_seq_len=64)


def test_full_config_param_count_matches_closed_form():
    for v in (24576, 28672, 32768):
        cfg = ModelConfig(vocab_size=v)
        with torch.device("meta"):
            m = Transformer(cfg)
        d, L = cfg.d_model, cfg.n_layer
        attn = d * cfg.n_head * cfg.head_dim * 2 + d * cfg.n_kv_head * cfg.head_dim * 2
        per_layer = attn + 3 * d * cfg.d_ff + 2 * d + 2 * cfg.head_dim
        assert m.num_params() == v * d + L * per_layer + d


def test_causal_future_tokens_do_not_affect_past_logits():
    torch.manual_seed(0)
    m = Transformer(TINY).eval()
    x = torch.randint(0, 97, (2, 20))
    y = x.clone()
    y[:, 12:] = torch.randint(0, 97, (2, 8))
    with torch.no_grad():
        a, b = m(x), m(y)
    assert torch.allclose(a[:, :12], b[:, :12], atol=1e-6)
    assert not torch.allclose(a[:, 12:], b[:, 12:])


def test_gqa_shapes_and_tied_embeddings():
    m = Transformer(TINY)
    assert m.blocks[0].attn.k.weight.shape == (2 * 8, 32)
    assert m.lm_head is None
    out = m(torch.randint(0, 97, (3, 10)))
    assert out.shape == (3, 10, 97)


def test_every_parameter_receives_gradient_and_ignore_index_masks_loss():
    torch.manual_seed(0)
    m = Transformer(TINY)
    x = torch.randint(0, 97, (2, 16))
    y = torch.randint(0, 97, (2, 16))
    _, loss = m(x, y)
    loss.backward()
    for n, p in m.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, n
    y2 = y.clone()
    y2[:, 8:] = -100
    _, l_masked = m(x, y2)
    _, l_half = m(x[:, :8], y[:, :8])
    assert torch.allclose(l_masked, l_half, rtol=1e-5)


def test_rejects_overlong_sequence():
    m = Transformer(TINY)
    try:
        m(torch.zeros(1, 65, dtype=torch.long))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_tiny_model_learns_permutation_chain_to_entropy_floor():
    """End-to-end learning check through the real data path. The in-document floor is
    ~0.1*ln(V-1) + 0.09 nats; document starts and EOS positions add unpredictable
    tokens, so the bound is 35% of ln(V) rather than the floor itself."""
    import math

    from j1bench.data import SyntheticPacked

    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=64, n_layer=2, d_model=64, n_head=4, n_kv_head=2, head_dim=16, d_ff=128, max_seq_len=64
    )
    m = Transformer(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    data = SyntheticPacked(64, 64, 16, seed=3, min_len=16, max_len=64, kind="perm")
    for _ in range(300):
        x, y, n = data.next_batch()
        _, ls = m(x, y)
        (ls / n).backward()
        opt.step()
        opt.zero_grad()
    assert (ls / n).item() < 0.35 * math.log(64)
