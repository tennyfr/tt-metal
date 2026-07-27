# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Focused unit test for the QKV projection matmul in prefill mode.

Guards the ``MatmulMultiCoreReuseMultiCastProgramConfig`` branch in
``ModelArgs.get_attn_qkv_program_config(Mode.PREFILL, seq_len=128, ...).

See the issue #50656 for more details.
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.model_config import ModelArgs

SEQ_LEN = 128

EXPECTED_DIM = 2048
EXPECTED_N_HEADS = 32
EXPECTED_N_KV_HEADS = 8
EXPECTED_HEAD_DIM = EXPECTED_DIM // EXPECTED_N_HEADS  # 64
EXPECTED_QKV_SIZE = EXPECTED_HEAD_DIM * (2 * EXPECTED_N_KV_HEADS + EXPECTED_N_HEADS)  # 3072

PCC_THRESHOLD = 0.99


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [{"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8)}.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_qkv_prefill_projection_full_output_pcc(mesh_device, reset_seeds, ensure_gc):
    """QKV = X @ W_qkv at seq_len=128 must match a torch reference over the FULL output.

    Regression guard for the P100-specific ``per_core_M=1`` override. Any
    ``per_core_M`` that is not a divisor of ``Mt = seq_len/32 = 4`` writes
    stale L1 into the padded tail of the output shard on P100, so a
    full-output PCC compare fails hard here.
    """

    model_args = ModelArgs(
        mesh_device,
        instruct=False,
        dummy_weights=True,
        max_batch_size=1,
        max_seq_len=128,
    )

    if model_args.device_name != "P100":
        pytest.skip(
            f"P100-only regression (device_name={model_args.device_name}). The P100 "
            "override at models/tt_transformers/tt/model_config.py (search 'P100 "
            "runs OOM in L1 with 8 per_core_M') is what this test guards."
        )

    if model_args.base_model_name != "Llama-3.2-1B":
        pytest.skip(
            f"This test pins Llama-3.2-1B dims (dim={EXPECTED_DIM}, qkv_size={EXPECTED_QKV_SIZE}). "
            f"Environment resolves to base_model_name={model_args.base_model_name!r}. "
            "Set LLAMA_DIR / HF_MODEL to a Llama-3.2-1B checkpoint (dummy_weights=True is fine)."
        )

    assert model_args.dim == EXPECTED_DIM, f"dim mismatch: {model_args.dim} != {EXPECTED_DIM}"
    assert model_args.n_heads == EXPECTED_N_HEADS
    assert model_args.n_kv_heads == EXPECTED_N_KV_HEADS
    assert model_args.head_dim == EXPECTED_HEAD_DIM
    assert model_args.qkv_size == EXPECTED_QKV_SIZE

    assert not model_args.use_minimal_qkv_prefill_matmul(
        SEQ_LEN
    ), f"seq_len={SEQ_LEN} unexpectedly routed to the minimal_matmul path"

    prog_cfg = model_args.get_attn_qkv_program_config(Mode.PREFILL, SEQ_LEN, None)
    mem_cfg_out = model_args.get_attn_qkv_mm_mem_config(Mode.PREFILL, None)
    logger.info(
        "QKV prefill program_config: per_core_M={}, per_core_N={}, "
        "out_subblock_h={}, out_subblock_w={}, in0_block_w={}, grid={}, fuse_batch={}",
        prog_cfg.per_core_M,
        prog_cfg.per_core_N,
        prog_cfg.out_subblock_h,
        prog_cfg.out_subblock_w,
        prog_cfg.in0_block_w,
        prog_cfg.compute_with_storage_grid_size,
        prog_cfg.fuse_batch,
    )

    dim = model_args.dim
    qkv_size = model_args.qkv_size
    num_devices = model_args.num_devices

    torch.manual_seed(0)
    x_pt = torch.rand(1, 1, SEQ_LEN, dim, dtype=torch.bfloat16) * 2 - 1
    w_pt = (torch.rand(1, 1, dim, qkv_size, dtype=torch.bfloat16) * 2 - 1) * 0.02

    ref = torch.matmul(x_pt.float(), w_pt.float()).to(torch.bfloat16)

    x_tt = ttnn.from_torch(
        x_pt,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    w_mem_config = model_args.create_dram_sharded_mem_config(dim, qkv_size // num_devices)

    w_tt = ttnn.from_torch(
        w_pt,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=w_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    out_tt = ttnn.linear(
        x_tt,
        w_tt,
        dtype=ttnn.bfloat16,
        memory_config=mem_cfg_out,
        compute_kernel_config=model_args.compute_kernel_config_hifi2,
        program_config=prog_cfg,
    )

    out_pt = ttnn.to_torch(
        out_tt,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
    )[:, 0:1, :, :qkv_size]

    # Sanity: no NaN/Inf. Corrupted L1 tail rows often surface here first.
    assert torch.isfinite(out_pt).all(), "QKV prefill output contains non-finite values"

    passing, msg = comp_pcc(ref, out_pt, PCC_THRESHOLD)
    logger.info(f"QKV prefill PCC (full tensor, all {SEQ_LEN} rows): {msg}")
    assert passing, f"QKV prefill projection failed PCC vs torch reference (threshold {PCC_THRESHOLD}). "
