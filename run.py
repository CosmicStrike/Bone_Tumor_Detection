from flask import Flask, request, render_template, current_app
import pickle
import json
import os
import base64
app = Flask(__name__)


@app.route("/")
def main():
    return render_template("index.html")


def load_model(f):
    with open(f, 'rb') as file:
        return pickle.load(file)


# API
@app.route("/model", methods=['POST'])
def model():

    # Save the image
    print(request.files['img'])
    file = request.files['img']
    file.save(os.path.join(current_app.root_path, 'inputImage.jpg'))

    # Load the model
    model = load_model('bone_cancer_model.pickle')
    result = model('inputImage.jpg')
    for i, r in enumerate(result):
        r.save(filename=f'result{i}.jpg')

    with open("result0.jpg", "rb") as img_file:
        my_string = base64.b64encode(img_file.read()).decode("utf-8")

    return json.dumps({"success": 1, "image": my_string})


if __name__ == "__main__":
    app.run(debug=True)
