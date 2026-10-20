"""
URL routing for analytics_app.
"""

from django.urls import path
from . import views

urlpatterns = [
    path("", views.overview_view, name="overview"),
    path("skills/", views.skills_view, name="skills"),
    path("duplicates/", views.duplicates_view, name="duplicates"),
    path("velocity/", views.velocity_view, name="velocity"),
    path("salary/", views.salary_view, name="salary"),

    # REST APIs for interactive UI elements
    path("api/cooccurrence/", views.cooccurrence_api, name="api_cooccurrence"),
    path("api/cluster/<str:cluster_id>/", views.cluster_detail_api, name="api_cluster_detail"),
    path("api/salary-predict/", views.salary_predict_api, name="api_salary_predict"),
]
