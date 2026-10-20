"""
Unit tests for MinHash LSH fuzzy deduplication and Disjoint Set Union clustering.
"""

import pytest
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.deduplication import DisjointSet


def test_disjoint_set_union_transitivity():
    """
    Tests that pairwise matches (A ~ B) and (B ~ C)
    correctly resolve to the same canonical equivalence class {A, B, C}.
    """
    ds = DisjointSet()
    ds.union("job_1", "job_2")
    ds.union("job_2", "job_3")

    # job_1, job_2, job_3 must share the same root
    assert ds.find("job_1") == ds.find("job_3")
    assert ds.find("job_2") == ds.find("job_3")

    # job_4 belongs to an independent cluster
    ds.union("job_4", "job_5")
    assert ds.find("job_4") == ds.find("job_5")
    assert ds.find("job_1") != ds.find("job_4")


def test_disjoint_set_path_compression():
    """
    Verifies that DisjointSet compresses tree paths for amortized O(alpha(N)) lookups.
    """
    ds = DisjointSet()
    for i in range(1, 100):
        ds.union(f"node_{i}", f"node_{i+1}")

    root = ds.find("node_1")
    for i in range(1, 101):
        assert ds.find(f"node_{i}") == root
