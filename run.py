from flask import Flask, request, render_template, current_app
import pickle
import json
import os
import base64
import torch
from ultralytics import YOLO
import shutil

app = Flask(__name__)


@app.route("/")
def main():
    return render_template("index.html")


def load_model():
    model = YOLO('best.pt', verbose=True)
    return model

model = load_model()

# API
@app.route("/model", methods=['POST'])
def root():

    # Save the image
    print(request.files['img'])
    file = request.files['img']
    file.save(os.path.join(current_app.root_path, 'inputImage.jpg'))

    if os.path.exists("runs"):
        shutil.rmtree("runs")
    result = model.predict('inputImage.jpg', save_txt=True)
    print(result[0])
    for i, r in enumerate(result):
        r.save(filename=f'result{i}.jpg')

    with open("result0.jpg", "rb") as img_file:
        my_string = base64.b64encode(img_file.read()).decode("utf-8")


    data_label_file = "runs/detect/predict/labels/inputImage.txt"


    preds = ""
    if os.path.exists(data_label_file):
        preds = "Found"

    return json.dumps({"success": 1, "image": my_string, "preds":preds})


if __name__ == "__main__":
    app.run(debug=True)
