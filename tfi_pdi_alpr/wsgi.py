
import os
from django.core.wsgi import get_wsgi_application
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'tfi_pdi_alpr.settings')
application = get_wsgi_application()
