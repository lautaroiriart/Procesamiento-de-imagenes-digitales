from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('alpr/', include('alpr.urls')),             # UI web
    path('api/',  include('alpr.api_urls')),         # API REST
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
