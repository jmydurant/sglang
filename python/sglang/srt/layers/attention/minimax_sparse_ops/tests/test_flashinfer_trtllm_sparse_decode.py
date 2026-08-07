from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention.minimax_sparse_backend import (
    MiniMaxSparseAttnBackend,
    _positive_python_scale,
)


@pytest.mark.parametrize("fp8", [False, True])
def test_trtllm_sparse_decode_dispatch_contract(fp8: bool):
    backend = object.__new__(MiniMaxSparseAttnBackend)
    backend.page_size = 128
    backend.topk_blocks = 16
    backend.block_size_k = 128
    backend.model_dtype = torch.bfloat16
    backend.fp8_attn_gemm = fp8
    backend._trtllm_sparse_workspace = torch.empty(32, dtype=torch.uint8)
    backend._trtllm_sparse_multi_ctas_kv_counter_buffer = torch.zeros(
        16, dtype=torch.uint8
    )

    calls = []

    def fake_decode(**kwargs):
        calls.append(kwargs)
        return torch.empty_like(kwargs["query"], dtype=kwargs["out_dtype"])

    backend._trtllm_sparse_decode_fn = fake_decode

    batch_size, num_q_heads, head_dim = 2, 16, 128
    q_dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
    kv_dtype = q_dtype
    # Match MiniMax's split QKV projection: q is a strided view in BF16 mode.
    q_storage = torch.empty((batch_size, num_q_heads, head_dim + 8), dtype=q_dtype)
    q = q_storage[..., :head_dim]
    assert not q.is_contiguous()
    k_cache = torch.empty((4 * 128, 1, head_dim), dtype=kv_dtype)
    v_cache = torch.empty_like(k_cache)
    page_table = torch.arange(batch_size * 16, dtype=torch.int32).view(batch_size, 16)
    seq_lens = torch.tensor([2048, 1937], dtype=torch.int32)
    layer = SimpleNamespace(
        scaling=head_dim**-0.5,
        q_scale_float=0.5 if fp8 else None,
        k_scale_float=0.25 if fp8 else None,
        v_scale_float=0.75 if fp8 else None,
    )

    out = backend._trtllm_sparse_main_decode(
        q, page_table, seq_lens, k_cache, v_cache, layer
    )

    assert out.shape == q.shape
    assert out.dtype == torch.bfloat16
    assert len(calls) == 1
    call = calls[0]
    assert call["query"].data_ptr() == q.data_ptr()
    assert call["query"].stride() == q.stride()
    assert call["enable_block_sparse_attention"] is True
    assert call["backend"] == "trtllm-gen"
    assert call["kv_layout"] == "HND"
    assert call["q_len_per_req"] == 1
    assert call["max_seq_len"] == 2048
    assert call["block_tables"].shape == (1, batch_size, 16)
    assert call["seq_lens"].shape == (1, batch_size)
    assert call["block_tables"].data_ptr() == page_table.data_ptr()
    assert call["seq_lens"].data_ptr() == seq_lens.data_ptr()
    assert call["multi_ctas_kv_counter_buffer"] is (
        backend._trtllm_sparse_multi_ctas_kv_counter_buffer
    )
    assert call["workspace_buffer"] is backend._trtllm_sparse_workspace

    k_hnd, v_hnd = call["kv_cache"]
    assert k_hnd.shape == (4, 1, 128, 128)
    assert v_hnd.shape == k_hnd.shape
    assert k_hnd.stride() == (128 * 128, 128 * 128, 128, 1)
    assert v_hnd.stride() == k_hnd.stride()
    if fp8:
        assert call["bmm1_scale"] == pytest.approx(0.5 * 0.25 * layer.scaling)
        assert call["bmm2_scale"] == pytest.approx(0.75)
    else:
        assert call["bmm1_scale"] == pytest.approx(layer.scaling)
        assert call["bmm2_scale"] == 1.0


def test_positive_python_scale_is_graph_stable():
    assert _positive_python_scale(None, "scale") == 1.0
    assert _positive_python_scale(-1.0, "scale") == 1.0
    assert _positive_python_scale(0.5, "scale") == 0.5
    with pytest.raises(TypeError, match="Python scalar"):
        _positive_python_scale(torch.tensor(0.5), "scale")
