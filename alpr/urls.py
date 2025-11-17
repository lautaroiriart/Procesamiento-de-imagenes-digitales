from django.urls import path, include
from . import views

app_name = 'alpr'

urlpatterns = [
    path('upload/', views.upload_view, name='upload'),
    path('api/', include('alpr.api_urls')),
]
