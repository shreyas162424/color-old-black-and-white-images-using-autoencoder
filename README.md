# Image Colorization Using Autoencoders

This project demonstrates automatic colorization of black and white images using a pretrained convolutional neural network (CNN) and deploys the model using a Flask web application.

---

## 🔬 Overview

Colorizing grayscale images enhances their realism and historical appeal. This project uses a pretrained model from OpenCV that was trained on the ImageNet dataset to predict color (ab channels) for grayscale input images (L channel). A user-friendly web interface is provided to upload images and view results.

---

## 📁 Project Structure

```
ImageColorizationProject/
|--Images/                     #Images Folder
├── colorize.py                # Flask backend
├── templates/index.html                 # Frontend UI
└── model/
    ├── colorization_release_v2.caffemodel
    ├── colorization_deploy_v2.prototxt
    └── pts_in_hull.npy
```

---

## 🧠 Features

* Upload black and white (grayscale) images via web interface
* Automatic prediction of realistic colors using pretrained DNN
* Display of original and colorized image side-by-side
* Runs locally using Flask

---

## ⚙️ Model Details

* **Model**: Caffe-based, trained on ImageNet
* **Files used**:

  * `colorization_release_v2.caffemodel`: pretrained weights
  * `colorization_deploy_v2.prototxt`: model architecture
  * `pts_in_hull.npy`: 313 cluster centers for ab channel quantization

---

## 🛠 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-repo/image-colorization
cd image-colorization
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

*Make sure OpenCV includes DNN support:*

```bash
pip install opencv-python opencv-contrib-python flask numpy pillow
```

### 3. Download Model Files

Place the following files in a `model/` directory:

* `colorization_release_v2.caffemodel`
* `colorization_deploy_v2.prototxt`
* `pts_in_hull.npy`
download the below model from the below githib link 
colorization_release_v2.caffemodel
(https://github.com/richzhang/colorization/tree/caffe/colorization/models)
---

## 🚀 Running the App

```bash
python colorize.py
```

Visit [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser. Upload a grayscale image to view the colorized result.

---

## 📊 Sample Output

* **Input**: Grayscale image
* **Output**: Side-by-side grayscale and colorized image returned as JPEG

---

## 📋 How It Works

1. Image uploaded via browser
2. Image converted to LAB color space
3. L (lightness) channel extracted
4. L channel passed to CNN model
5. Predicted ab channels concatenated with L
6. Converted back to BGR image
7. Returned to browser for visualization

---

## 🚧 Future Improvements

* Integrate GAN-based colorization models for more vivid output
* Allow user-guided color hints
* Add drag-and-drop and preview support
* Deploy to cloud (e.g., Heroku, Render, or AWS)

---

## 📚 References

* OpenCV DNN Colorization Example: [https://github.com/richzhang/colorization](https://github.com/richzhang/colorization)
* Caffe Pretrained Models: [https://github.com/BVLC/caffe/wiki/Model-Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)

---

## 👨‍💼 Author

* Developed by: \[Shreyas Sangalad]
* Under the guidance of: Mrs. Roopa M J, SJBIT, Bangalore

---

## ✉️ License

This project is for academic and educational use only. Attribution required if reused.
