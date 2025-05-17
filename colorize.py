import numpy as np
import argparse
import cv2
import os
from flask import Flask, render_template, request, send_file
import io
from PIL import Image

app = Flask(__name__)

# Paths to load the model
DIR = r"C:\Users\spoorti sangalad\Downloads\Colorizing-black-and-white-images-using-Python-master"
PROTOTXT = os.path.join(DIR, "model", "colorization_deploy_v2.prototxt")
POINTS = os.path.join(DIR, "model", "pts_in_hull.npy")
MODEL = os.path.join(DIR, "model", "colorization_release_v2.caffemodel")

# Load the Model
print("Load model")
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
pts = np.load(POINTS)

# Load centers for ab channel quantization used for rebalancing.
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/colorize', methods=['POST'])
def colorize_image():
    if 'image' not in request.files:
        return "No image file found in request", 400
    
    file = request.files['image']
    if file.filename == '':
        return "No image file selected", 400

    # Read the input image
    in_memory_file = file.read()
    npimg = np.frombuffer(in_memory_file, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Convert the image to float32 and scale it
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    # Resize for processing
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    # Colorizing the image
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")

    # Concatenate the black and colorized images horizontally
    gray_display = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_display = cv2.cvtColor(gray_display, cv2.COLOR_GRAY2BGR)
    gray_display = cv2.resize(gray_display, (colorized.shape[1], colorized.shape[0]))
    
    combined = cv2.hconcat([gray_display, colorized])

    # Convert the combined image to a format that can be displayed in the browser
    _, buffer = cv2.imencode('.jpg', combined)
    io_buf = io.BytesIO(buffer)
    
    return send_file(io_buf, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
