import os
import numpy as np
import tensorflow as tf
import pickle
import config as cfg
from inference import inference_with_color_filter
from flask import Flask, request, render_template, send_from_directory

# set root dir
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# define model
model = tf.keras.models.load_model(os.path.join("saver", "model_epoch_6"))

# load training set vector
with open("hamming_train.pickle", "rb") as f:
    train_vector = pickle.load(f)

# load color vector
with open("color_featuer.pickle", "rb") as f:
    color_vector = pickle.load(f)

# load train set path
with open("training_image_pickle.pickle", "rb") as f:
    train_image_path = pickle.load(f)

# define flask app
app = Flask(__name__, static_url_path="/static")

# define home route
@app.route("/")
def index():
    return render_template("index.html")

# define upload function
@app.route("/upload", methods=["POST"])
def upload():
    upload_dir = os.path.join(APP_ROOT, "upload/")

    if not os.path.isdir(upload_dir):
        os.mkdir(upload_dir)

    for img in request.files.getlist("file"):
        img_name = img.filename
        destination = "/".join([upload_dir, img_name])

        img.save(destination)

    # inference
    result = np.array(train_image_path)[inference_with_color_filter(model,
                                                                    train_vector,
                                                                    os.path.join(upload_dir, img_name),
                                                                    color_vector,
                                                                    cfg.IMAGE_SIZE)]

    final_result = []

    for img in result:
        final_result.append("images/" + img.split("/")[-1]) # grab file name

    return render_template("result.html", image_name=img_name, result_path=final_result)

# define helper function
@app.route("/upload/<filename>")
def send_image(filename):
    return send_from_directory("upload", filename)

#Start application
if __name__ == "__main__":
    app.run(port=5000, debug=True)