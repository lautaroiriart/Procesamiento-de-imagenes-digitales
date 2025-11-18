urlpatterns = [
    path('admin/', admin.site.urls),
    path('alpr/', include(('alpr.urls', 'alpr'), namespace='alpr')),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
