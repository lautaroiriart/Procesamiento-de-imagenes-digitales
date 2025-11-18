<<<<<<< HEAD
=======
from django.contrib import admin
from django.urls import path, include     # <-- ESTA LÍNEA ES LA QUE FALTABA
from django.conf import settings
from django.conf.urls.static import static

>>>>>>> 76b396c (Arreglo de logica y comparativa con Tesseract)
urlpatterns = [
    path('admin/', admin.site.urls),
    path('alpr/', include(('alpr.urls', 'alpr'), namespace='alpr')),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
