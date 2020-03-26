from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import cv2
from tensorflow.keras.preprocessing import image
import numpy as np

from flask_cors import CORS, cross_origin

names = ["normal", "COVID"]

# Process image and predict label
def processImg(IMG_PATH):
    #Read image
    model = load_model("covid19.model")

    # Preprocess image
    img = cv2.imread(IMG_PATH)
    img = cv2.resize(img, (224, 224))
    img = img.astype("float") / 255.0
    img = img.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    res = model.predict(img)
    label = np.argmax(res)

    print("Label", label)
    labelName = name[label]
    print("Label name:", labelName)
    return labelName

# Initializing flask application
app = Flask(__name__)
cors = CORS(app)

@app.route("/")
def main():
    return """
        Application is working
    """

# About page with render template
@app.route("/about")
def postsPage():
    return render_template("about.html")

# Process images
@app.route("/process", methods=["POST"])
def processReq():
    data = request.files["img"]
    data.save("img.jpg")

    resp = processImg("img.jpg")


    return resp

if __name__ == "__main__":
    app.run(debug=True)
