from django.http import HttpResponse, JsonResponse
from PillServicesPythonEnv.OCR import pill_imprint_prediction
from PillServicesPythonEnv.Pytorch_Infer_Color import run_color_inferences
from PillServicesPythonEnv.Pytorch_Infer_Shape import run_shape_inferences
from django.views.decorators.csrf import csrf_exempt
import json

def index(request):
    return HttpResponse("Hi there. The Python web service is up.")

@csrf_exempt
def get_pill_shape_predictions(request):
    if request.method != "POST":
        return HttpResponse("This endpoint needs to be a POST request!")

    data = json.loads(request.body)
    input_file_location = data.get("input_file_location", "")

    print(input_file_location)

    return HttpResponse(run_shape_inferences(input_file_location)) 


@csrf_exempt
def get_pill_color_predictions(request):
    
    if request.method != "POST":
        return HttpResponse("This endpoint needs to be a POST request!")

    data = json.loads(request.body)
    input_file_location = data.get("input_file_location", "")

    print(input_file_location)

    return HttpResponse(run_color_inferences(input_file_location)) 


@csrf_exempt
def get_pill_imprint_predictions(request):
    
    if request.method != "POST":
        return HttpResponse("This endpoint needs to be a POST request!")

    data = json.loads(request.body)
    input_file_location = data.get("input_file_location", "")

    print(input_file_location)

    return HttpResponse(pill_imprint_prediction(input_file_location)) 


# From local directory
# py manage.py runserver 7000