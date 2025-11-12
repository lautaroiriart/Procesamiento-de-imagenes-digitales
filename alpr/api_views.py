from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, parsers
from .ml.inference import infer

class HealthAPIView(APIView):
    def get(self, request):
        return Response({"ok": True}, status=200)

class InferAPIView(APIView):
    parser_classes = [parsers.MultiPartParser, parsers.FormParser]

    def post(self, request):
        f = request.FILES.get("image")
        if not f:
            return Response({"error": "No image provided (field 'image')"}, status=400)
        result = infer(f.read())
        return Response(result, status=200)

# OPCIONAL: ojo, bloquea el proceso; para PoC nomás.
# from django.core.management import call_command
# class TrainAPIView(APIView):
#     def post(self, request):
#         epochs = int(request.data.get("epochs", 5))
#         call_command("train_ocr", epochs=epochs)
#         return Response({"status": "training finished"}, status=200)
