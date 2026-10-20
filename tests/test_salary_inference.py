"""
Unit tests for the salary estimation service.
"""

import pytest
import sys
from pathlib import Path

# Add project root and dashboard to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "dashboard"))

from analytics_app.services import LakehouseService


def test_salary_estimation_seniority_scaling():
    """
    Verifies that salary predictions scale monotonically with career seniority tiers.
    Junior < Mid < Senior < Lead < Staff_Principal
    """
    junior = LakehouseService.estimate_salary("Software Engineer", seniority="Junior", state="CA")
    mid = LakehouseService.estimate_salary("Software Engineer", seniority="Mid", state="CA")
    senior = LakehouseService.estimate_salary("Software Engineer", seniority="Senior", state="CA")
    lead = LakehouseService.estimate_salary("Software Engineer", seniority="Lead", state="CA")
    staff = LakehouseService.estimate_salary("Software Engineer", seniority="Staff_Principal", state="CA")

    assert junior["predicted_salary"] < mid["predicted_salary"]
    assert mid["predicted_salary"] < senior["predicted_salary"]
    assert senior["predicted_salary"] < lead["predicted_salary"]
    assert lead["predicted_salary"] < staff["predicted_salary"]


def test_salary_estimation_remote_premium():
    """
    Verifies that remote roles receive empirical premium adjustment.
    """
    onsite = LakehouseService.estimate_salary("Data Engineer", seniority="Senior", state="TX", is_remote=False)
    remote = LakehouseService.estimate_salary("Data Engineer", seniority="Senior", state="TX", is_remote=True)

    assert remote["predicted_salary"] >= onsite["predicted_salary"]
