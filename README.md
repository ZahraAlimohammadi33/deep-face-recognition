# Face Recognition with OpenCV, YuNet & SFace

A real-time face recognition project using **OpenCV**, **YuNet** for face detection, and **SFace** for face feature extraction and recognition.

The system compares faces captured from a webcam with a predefined reference image and determines whether they belong to the same person.

* 🟢 **MATCH** — The detected face matches the reference face
* 🔴 **NOT MATCH** — The detected face does not match the reference face
* ⚪ **NO FACE DETECTED** — No face is detected in the current frame

## Features

* Real-time face detection from webcam
* Face detection using **YuNet**
* Face alignment using facial landmarks
* Face feature extraction using **SFace**
* Face comparison using:

  * L2 Distance
  * Cosine Similarity
* Real-time recognition result displayed on the video frame
* Pre-extraction of the reference face feature for better performance

## Model Setup

This project uses two pretrained ONNX models from the official **OpenCV Zoo** repository.
پپ
Since the model files are relatively large, they are **not included in this repository**. Download them from the official OpenCV sources and place them in the following directory:

```text
model/
└── face/
    ├── face_detection_yunet_2022mar.onnx
    └── face_recognition_sface_2021dec.onnx
```

### 1. YuNet — Face Detection

Download:

[YuNet — face_detection_yunet_2022mar.onnx](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet?utm_source=chatgpt.com)

The YuNet model is used for detecting faces and locating facial landmarks.

### 2. SFace — Face Recognition

Download:

[SFace — face_recognition_sface_2021dec.onnx](https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx?utm_source=chatgpt.com)

The SFace model is used to extract facial feature vectors and compare face identities. The model is stored using Git LFS because of its size.

### Final Directory Structure

After downloading the models, your project should look like this:

```text
face-recognition/
│
├── main.py
├── README.md
├── .gitignore
│
├── images/
│   └── akhavan.jpg
│
└── model/
    └── face/
        ├── face_detection_yunet_2022mar.onnx
        └── face_recognition_sface_2021dec.onnx
```

> **Important:** Do not rename the downloaded model files unless you also update their paths in `main.py`.


## Installation

Clone the repository:

```bash
git clone https://github.com/ZahraAlimohammadi33/face-recognition.git
cd face-recognition
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Requirements

The main dependencies are:

```text
opencv-python
numpy
matplotlib
```

You can install them manually with:

```bash
pip install opencv-python numpy matplotlib
```

## Usage

Make sure the reference image exists at:

```text
images/akhavan.jpg
```

Then run:

```bash
python main.py
```

The webcam window will open automatically.

Press:

```text
q
```

to exit the application.

## Face Comparison

The project uses two similarity metrics provided by OpenCV's `FaceRecognizerSF`:

### L2 Distance

A lower L2 distance indicates greater similarity between two face feature vectors.

The project uses:

```python
L2_SIMILARITY_THRESHOLD = 1.128
```

A value below this threshold is considered a potential match.

### Cosine Similarity

A higher cosine similarity indicates greater similarity between the two feature vectors.

The project uses:

```python
COSINE_SIMILARITY_THRESHOLD = 0.363
```

A value above this threshold is considered a potential match.

The final decision is based on both metrics.

## Technologies

* **Python**
* **OpenCV**
* **YuNet**
* **SFace**
* **NumPy**
* **ONNX**

## Future Improvements

Possible extensions for this project include:

* Multiple known identities
* Automatic name recognition
* `Unknown` identity detection
* Face recognition from images and videos
* Face database management
* Recognition confidence visualization
* Performance/FPS monitoring
* Improved handling of multiple faces
* Real-time attendance system

## Note

This project is intended for educational and demonstration purposes. Recognition thresholds may need to be adjusted depending on the camera, lighting conditions, image quality, and application requirements.
