from django.http import HttpResponse, JsonResponse
from PillServicesPythonEnv.OCR import pill_imprint_prediction
from django.views.decorators.csrf import csrf_exempt
import json


def index(request):
    return HttpResponse("Hi there. The Python web service is up.")

def get_pill_shape_predictions(request):
    return HttpResponse("PILL SHAPE: This is a string of the inferance output.")

def get_pill_color_predictions(request):
    return HttpResponse("PILL COLOR: This is a string of the inferance output.")


@csrf_exempt
def get_pill_imprint_predictions(request):
    
    if request.method != "POST":
        return HttpResponse("This endpoint needs to be a POST request!")

    #input_file_location = request.POST["input_file_location"]

    data = json.loads(request.body)
    input_file_location = data.get("input_file_location", "")

    print(input_file_location)

    return HttpResponse("PILL IMPRINT: This is a string of the inferance output: " + pill_imprint_prediction(input_file_location) + " from: " + input_file_location) 


# From local directory
# py manage.py runserver 7000