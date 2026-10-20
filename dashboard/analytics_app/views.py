"""
Django Views for Spark Job Market Analytics Dashboard.
"""

from django.http import JsonResponse
from django.shortcuts import render
from .services import LakehouseService


def overview_view(request):
    """
    Executive Market Overview Dashboard.
    """
    kpis = LakehouseService.get_overview_kpis()
    seniority = LakehouseService.get_seniority_market_dynamics()
    return render(request, "overview.html", {
        "kpis": kpis,
        "seniority": seniority,
        "active_page": "overview"
    })


def skills_view(request):
    """
    Technology Skills Intelligence & GPU Co-occurrence Explorer.
    """
    category = request.GET.get("category", "all")
    selected_skill = request.GET.get("skill", "Python")

    skills_data = LakehouseService.get_skills_intelligence(category=category)
    cooccur_data = LakehouseService.get_skill_cooccurrences(skill_name=selected_skill)

    return render(request, "skills.html", {
        "skills_data": skills_data,
        "cooccur_data": cooccur_data,
        "selected_skill": selected_skill,
        "selected_category": category,
        "active_page": "skills"
    })


def cooccurrence_api(request):
    """
    JSON API for dynamic asynchronous co-occurrence chart updates.
    """
    skill = request.GET.get("skill", "Python")
    data = LakehouseService.get_skill_cooccurrences(skill_name=skill)
    return JsonResponse(data)


def duplicates_view(request):
    """
    Cross-Source Duplicate Clusters Explorer.
    """
    query = request.GET.get("q", "").strip()
    source = request.GET.get("source", "all")
    page = int(request.GET.get("page", 1))

    cluster_data = LakehouseService.get_duplicate_clusters(
        search_query=query,
        source_filter=source,
        page=page,
        page_size=20
    )

    return render(request, "duplicates.html", {
        "data": cluster_data,
        "active_page": "duplicates"
    })


def cluster_detail_api(request, cluster_id):
    """
    JSON API returning member jobs for modal inspection.
    """
    details = LakehouseService.get_cluster_details(cluster_id)
    return JsonResponse(details)


def velocity_view(request):
    """
    Company Hiring Velocity & Market Surges.
    """
    velocity_data = LakehouseService.get_company_hiring_velocity(limit=25)
    return render(request, "velocity.html", {
        "velocity": velocity_data,
        "active_page": "velocity"
    })


def salary_view(request):
    """
    Experimental Salary Estimator & MLlib Model Performance Dashboard.
    """
    title = request.GET.get("title", "Senior Software Engineer")
    seniority = request.GET.get("seniority", "Senior")
    state = request.GET.get("state", "CA")
    is_remote = request.GET.get("is_remote", "true").lower() in ("true", "1", "yes")

    prediction = LakehouseService.estimate_salary(
        title=title,
        seniority=seniority,
        state=state,
        is_remote=is_remote
    )

    return render(request, "salary.html", {
        "prediction": prediction,
        "active_page": "salary"
    })


def salary_predict_api(request):
    """
    JSON API for real-time interactive salary calculator.
    """
    title = request.GET.get("title", "Software Engineer")
    seniority = request.GET.get("seniority", "Mid")
    state = request.GET.get("state", "CA")
    is_remote = request.GET.get("is_remote", "false").lower() in ("true", "1", "yes")

    result = LakehouseService.estimate_salary(
        title=title,
        seniority=seniority,
        state=state,
        is_remote=is_remote
    )
    return JsonResponse(result)
