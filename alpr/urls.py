from django.urls import path, include
from . import views

app_name = 'alpr'

urlpatterns = [
    # UI web
    path('upload/', views.upload_view, name='upload'),

    # API REST (si tenés un archivo alpr/api_urls.py)
    path('api/', include('alpr.api_urls')),
]
