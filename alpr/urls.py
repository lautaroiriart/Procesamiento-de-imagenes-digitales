
from django.urls import path
from .views import upload_view
urlpatterns = [ path('upload/', upload_view, name='alpr-upload') ]
