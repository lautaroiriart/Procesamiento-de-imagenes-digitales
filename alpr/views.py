from django.shortcuts import render

def upload_view(request):
    # Solo renderiza el front. 
    return render(request, "alpr/upload.html")
