from flask import Flask, request, jsonify
from joblib import dump, load
import numpy as np
from PIL import Image
import io ,os

# flask api file 
# create flask app object
app = Flask(__name__)

# decorator for base  route
@app.route("/")
def hello_world1():
    return "<p>Hello, World!</p>"

# decorator with base post function
@app.route("/", methods=["POST"])
def hello_world_post():    
    return {"op" : "Hello, World POST " + request.json["suffix"]}


# predict route decorator to get inferencing from model
@app.route("/prediction", methods = ['POST'])
def imagePrediction():
    input_json = request.get_json( )
    img1 = input_json['image']
    load_best_model = load('./models/svm_gamma_0.001_C_1_kernel_rbf.joblib')
    image1_1d_reshaped = np.array(img1).reshape(1, -1)
    prediction1 = load_best_model.predict(image1_1d_reshaped)
    print(prediction1)
    return str(prediction1)