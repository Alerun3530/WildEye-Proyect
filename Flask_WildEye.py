from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import base64
import pandas as pd

app = Flask(__name__)

# Cargar tu modelo aquí
model = load_model('WildEyev1.h5')

# Cargar el archivo Excel con la información de los animales
animal_info = pd.read_excel('animals-details.xlsx')

# Definición de las etiquetas
labels = [
    'antelope', 'badger', 'bat', 'bear', 'bee', 'beetle', 'bison', 'boar', 'butterfly', 'cat',
    'caterpillar', 'chimpanzee', 'cockroach', 'cow', 'coyote', 'crab', 'crow', 'deer', 'dog', 'dolphin',
    'donkey', 'dragonfly', 'duck', 'eagle', 'elephant', 'flamingo', 'fly', 'fox', 'goat',
    'goldfish', 'goose', 'gorilla', 'grasshopper', 'hamster', 'hare', 'hedgehog', 'hippopotamus', 
    'hornbill', 'horse', 'hummingbird', 'hyena', 'jellyfish', 'kangaroo', 'koala', 'ladybugs', 
    'leopard', 'lion', 'lizard', 'lobster', 'mosquito', 'moth', 'mouse', 'octopus', 'okapi', 
    'orangutan', 'otter', 'owl', 'ox', 'oyster', 'panda', 'parrot', 'pelecaniformes', 
    'penguin', 'pig', 'pigeon', 'porcupine', 'possum', 'raccoon', 'rat', 'reindeer', 
    'rhinoceros', 'sandpiper', 'seahorse', 'seal', 'shark', 'sheep', 'snake', 'sparrow', 
    'squid', 'squirrel', 'starfish', 'swan', 'tiger', 'turkey', 'turtle', 'whale', 
    'wolf', 'wombat', 'woodpecker', 'zebra'
]

# Ruta para la página principal
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para la predicción
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Recibir datos JSON
    img_data = data['image'].split(',')[1]  # Obtener solo la parte base64
    img = base64.b64decode(img_data)  # Decodificar la imagen

    # Convertir la imagen a un formato que OpenCV pueda manejar
    nparr = np.frombuffer(img, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Redimensionar la imagen a 224x224
    img = cv2.resize(img, (224, 224)) / 255.0  # Normalizar entre 0 y 1
    img = np.expand_dims(img, axis=0)  # Expandir dimensiones para el batch

    # Realiza la predicción
    preds = model.predict(img)
    predicted_index = np.argmax(preds)
    predicted_label = labels[predicted_index]

    # Buscar la información adicional del animal en el DataFrame
    animal_data = animal_info.iloc[predicted_index].to_dict()

    return jsonify({
        'prediction': predicted_label,
        'info': animal_data
    })

if __name__ == '__main__':
    app.run(debug=True)
