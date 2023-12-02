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
@app.route("/predict/<model_type>", methods = ['POST'])
def imagePrediction(model_type):
    input_json = request.get_json( )
    img1 = input_json['image']
    
    # load best one lr, svm and tree model from load model function
    best_lr_model, best_svm_model, best_tree_model=load_models()

    # Based on the model_type in route load the model of that type 

    if model_type == 'lr':
        loaded_best_model=best_lr_model
    elif model_type == 'svm':
        loaded_best_model=best_svm_model
    else:
        loaded_best_model=best_tree_model

    
    image1_1d_reshaped = np.array(img1).reshape(1, -1)

    # do prediction from the best load model based on the route input
    prediction1 = loaded_best_model.predict(image1_1d_reshaped)
    print(prediction1)
    return str(prediction1)

def load_models():

    #load best one logistic, svm, tree model from model folder and return

    load_best_lr_model = load('./models/M22AIE247_lr_solver_lbfgs.joblib')
    load_best_svm_model = load('./models/M22AIE247_svm_gamma_1_C_100_kernel_rbf.joblib')
    load_best_tree_model = load('./models/M22AIE247_tree_max_depth_100.joblib')

    return load_best_lr_model, load_best_svm_model, load_best_tree_model
    