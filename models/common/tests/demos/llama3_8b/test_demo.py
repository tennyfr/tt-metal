# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
from types import SimpleNamespace

import pytest

from models.common.modules.tt_ccl import default_topology
from models.tt_transformers.tt.generator import _galaxy_data_parallel_submesh_shape

_original_mesh_device = os.environ.get("MESH_DEVICE")
os.environ.setdefault("MESH_DEVICE", "N150")

from models.common.tests.demos.llama3_8b import demo

if _original_mesh_device is None:
    os.environ.pop("MESH_DEVICE")
else:
    os.environ["MESH_DEVICE"] = _original_mesh_device


def test_expected_for_case_returns_only_complete_performance_gate():
    expected = {
        "top1": 97,
        "top5": 100,
        "batch-32": {"tok_s_u": 52.2, "ttft_ms": 41.9},
    }

    assert demo._expected_for_case(expected, "batch-32") == {"tok_s_u": 52.2, "ttft_ms": 41.9}


@pytest.mark.parametrize(
    ("case_expected", "missing_metrics"),
    [
        (None, "tok_s_u, ttft_ms"),
        ({"tok_s_u": 52.2}, "ttft_ms"),
        ({"ttft_ms": 41.9}, "tok_s_u"),
    ],
)
def test_expected_for_case_disables_in_test_gate_when_metrics_are_missing(monkeypatch, case_expected, missing_metrics):
    warnings = []
    expected = {"top1": 97, "top5": 100}
    if case_expected is not None:
        expected["batch-32-ci"] = case_expected
    monkeypatch.setattr(demo.logger, "warning", warnings.append)

    assert demo._expected_for_case(expected, "batch-32-ci") is None
    assert warnings == [
        "No complete in-test performance gate for batch-32-ci; "
        f"missing {missing_metrics}. Centralized post-run validation remains authoritative."
    ]


def test_n150_batch_32_ci_remains_explicitly_unsupported(monkeypatch, expect_error):
    monkeypatch.setattr(demo, "get_device_name", lambda mesh_device: "N150")

    with expect_error(pytest.skip.Exception, "capacity is not enabled for N150"):
        demo._skip_unsupported_case(demo.DEMO_CASES["batch-32-ci"], SimpleNamespace())


def test_t3k_batch_32_ci_is_not_skipped(monkeypatch):
    monkeypatch.setattr(demo, "get_device_name", lambda mesh_device: "T3K")

    demo._skip_unsupported_case(demo.DEMO_CASES["batch-32-ci"], SimpleNamespace())


@pytest.mark.parametrize(
    ("data_parallel", "expected_shape"),
    [
        (4, (1, 8)),
        (8, (1, 4)),
        (16, (1, 2)),
        (32, (1, 1)),
    ],
)
def test_galaxy_dp_uses_routeable_submeshes(data_parallel, expected_shape):
    assert tuple(_galaxy_data_parallel_submesh_shape(32 // data_parallel)) == expected_shape


def test_galaxy_demo_opens_the_routeable_parent_orientation():
    assert demo._MESH_DEVICE_TO_SHAPE["TG"] == (4, 8)


@pytest.mark.parametrize(
    ("cluster_type", "expected"),
    [
        (demo.ttnn.cluster.ClusterType.T3K, demo.ttnn.Topology.Ring),
        (demo.ttnn.cluster.ClusterType.GALAXY, demo.ttnn.Topology.Linear),
    ],
)
def test_eight_device_ccl_topology_respects_parent_cluster(monkeypatch, cluster_type, expected):
    monkeypatch.setattr(demo.ttnn.cluster, "get_cluster_type", lambda: cluster_type)

    assert default_topology(SimpleNamespace(get_num_devices=lambda: 8)) == expected
