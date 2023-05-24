from keras.models import load_model
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image
import numpy as np
import controllers.plant_identification as plant_identification
import controllers.plant_disease as plant_disease
import os
from werkzeug.utils import secure_filename
import requests
import uuid

app = Flask(__name__)
model_path = os.path.join(os.getcwd(), 'models', 'RESNET_PLANT_IDENTIFICATION_CLASSES_140.h5')
model = load_model(model_path)

@app.route('/', methods=['GET'])
def hello_world():
    return 'Hello World!'


@app.route('/identify', methods=['POST'])
def identify_plant_disease():
    
    json_data = request.get_json()  # Retrieve the JSON payload
    image_url = None

    if 'image_url' in json_data:
        image_url = json_data['image_url']
    else:
        return jsonify({'error': 'image_url not found in request'}) 
    

    print("image_url->", image_url)
    img_data = requests.get(image_url).content

    image_path = os.path.join(os.getcwd(), 'images', str(uuid.uuid4()))+".jpg"
    
    with open(image_path, 'wb') as handler:   
        handler.write(img_data)

    plant_species_pred = plant_identification.plant_species(image_path, model)
    plant_disease_pred = plant_disease.plant_disease(image_path)

    response =  jsonify({'plant_species': plant_species_pred, 'plant_disease': plant_disease_pred})
    response.status_code = 200
    return response


if __name__ == '__main__':
    # debug run
    # app.run(debug=True)
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)
