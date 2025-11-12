from django.urls import path
from .api_views import InferAPIView, HealthAPIView # , TrainAPIView

urlpatterns = [
    path("infer/",  InferAPIView.as_view(),  name="api-infer"),
    path("health/", HealthAPIView.as_view(), name="api-health"),
    # path("train/",  TrainAPIView.as_view(),  name="api-train"),
]
