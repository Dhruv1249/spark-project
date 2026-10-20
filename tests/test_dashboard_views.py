"""
Integration tests for Django dashboard views and endpoints.
"""

import os
import sys
import pytest
from pathlib import Path

# Add dashboard to python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "dashboard"))

# Setup Django environment
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dashboard_project.settings")
import django
django.setup()

from django.test import Client


@pytest.fixture
def client():
    return Client()


def test_overview_page_status_200(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Executive Market Overview" in response.content.decode()


def test_skills_page_status_200(client):
    response = client.get("/skills/")
    assert response.status_code == 200
    assert "Technology Skills Intelligence" in response.content.decode()


def test_duplicates_page_status_200(client):
    response = client.get("/duplicates/")
    assert response.status_code == 200
    assert "Duplicate Clusters" in response.content.decode()


def test_velocity_page_status_200(client):
    response = client.get("/velocity/")
    assert response.status_code == 200
    assert "Hiring Velocity" in response.content.decode()


def test_salary_page_status_200(client):
    response = client.get("/salary/")
    assert response.status_code == 200
    assert "Salary Estimator" in response.content.decode()


def test_cooccurrence_api(client):
    response = client.get("/api/cooccurrence/?skill=Python")
    assert response.status_code == 200
    data = response.json()
    assert "cooccurrences" in data
    assert len(data["cooccurrences"]) > 0


def test_salary_predict_api(client):
    response = client.get("/api/salary-predict/?title=Data%20Engineer&seniority=Senior&state=CA&is_remote=true")
    assert response.status_code == 200
    data = response.json()
    assert "predicted_salary" in data
    assert data["predicted_salary"] > 50000
