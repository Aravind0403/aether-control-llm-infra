"""Tests for Kubernetes GPU Infrastructure RFCs (K8S-RFC-001)."""

import pytest
from k8s_infra.telemetry_governor import (
    DCGMMutexContentionModel,
    SharedMemoryCacheSimulator,
    KernelXIDTrapDetector,
)


def test_dcgm_mutex_contention_reduction():
    model = DCGMMutexContentionModel(total_nodes=100, gpus_per_node=8)

    # 500ms naive scraping: 80,000 scrapes/sec, 18.5% mutex hold, 350ms TTFT
    res_naive = model.evaluate_scrape_cadence(500.0, use_tiered_groups=False)
    assert res_naive.total_scrapes_per_sec == 80000.0
    assert res_naive.mutex_hold_time_pct == 18.5
    assert res_naive.estimated_p99_ttft_ms == 350.0
    assert res_naive.is_safe is False

    # 5000ms with Tiered Metric Groups: >85% reduction, locked 45ms TTFT
    res_tiered = model.evaluate_scrape_cadence(5000.0, use_tiered_groups=True)
    assert res_tiered.driver_lock_reduction_pct > 85.0
    assert res_tiered.mutex_hold_time_pct < 2.0
    assert res_tiered.estimated_p99_ttft_ms == 45.0
    assert res_tiered.is_safe is True


def test_shared_memory_cache_read_latency():
    sim = SharedMemoryCacheSimulator()
    res = sim.read_metrics_from_shm()

    assert res.storage_medium == "/dev/shm"
    assert res.is_driver_mutex_acquired is False
    assert res.num_metrics_returned == 9
    assert res.read_latency_ms < 0.5
    assert res.is_sub_millisecond is True


def test_kernel_xid_fatal_trap_and_node_quarantine():
    detector = KernelXIDTrapDetector()

    # Fatal XID 31: Page fault
    kmsg_31 = "NVRM: Xid (PCI:0000:01:00): 31, GPU Memory Page Fault"
    trap_31 = detector.process_kmsg_line(kmsg_31)
    assert trap_31 is not None
    assert trap_31.is_fatal is True
    assert trap_31.xid_code == 31
    assert trap_31.pci_address == "0000:01:00"
    assert trap_31.applied_node_taint == "node.kubernetes.io/unschedulable:NoSchedule"
    assert trap_31.quarantine_latency_ms < 5.0

    # Fatal XID 45: ECC Double-Bit Error
    kmsg_45 = "NVRM: Xid (PCI:0000:02:00): 45, Uncorrectable Double-Bit Error"
    trap_45 = detector.process_kmsg_line(kmsg_45)
    assert trap_45 is not None
    assert trap_45.is_fatal is True
    assert trap_45.xid_code == 45
    assert trap_45.applied_node_taint == "node.kubernetes.io/unschedulable:NoSchedule"

    # Non-fatal message
    kmsg_info = "NVRM: Xid (PCI:0000:01:00): 13, Graphics Engine Exception"
    trap_info = detector.process_kmsg_line(kmsg_info)
    assert trap_info is not None
    assert trap_info.is_fatal is False
    assert trap_info.applied_node_taint == "none"

    # Unrelated kernel log
    unrelated = "ext4-fs (sda1): mounted filesystem without journal"
    assert detector.process_kmsg_line(unrelated) is None
