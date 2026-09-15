import cv2

YU_NET_MODEL = "model/face/face_detection_yunet_2022mar.onnx"
SFACE_MODEL = "model/face/face_recognition_sface_2021dec.onnx"

REFERENCE_IMAGE = "images/akhavan.jpg"

L2_SIMILARITY_THRESHOLD = 1.128
COSINE_SIMILARITY_THRESHOLD = 0.363


def detect_faces(image, detector):

    height, width = image.shape[:2]
    detector.setInputSize((width, height))
    faces = detector.detect(image)

    return faces

def draw_face(frame, face, color):

    coords = face[:-1].astype(int)
    x, y, w, h = coords[:4]

    # Bounding box
    cv2.rectangle(
        frame,
        (x, y),
        (x + w, y + h),
        color,
        2
    )

    # Facial landmarks
    landmarks = [
        (coords[4], coords[5]),
        (coords[6], coords[7]),
        (coords[8], coords[9]),
        (coords[10], coords[11]),
        (coords[12], coords[13])
    ]

    for point in landmarks:
        cv2.circle(
            frame,
            point,
            2,
            color,
            -1
        )


def extract_face_feature(image, face, recognizer):

    aligned_face = recognizer.alignCrop(image, face)
    feature = recognizer.feature(aligned_face)

    return feature


def compare_faces(
    reference_feature,
    current_feature,
    recognizer
):

    l2_score = recognizer.match(
        reference_feature,
        current_feature,
        cv2.FaceRecognizerSF_FR_NORM_L2
    )

    cosine_score = recognizer.match(
        reference_feature,
        current_feature,
        cv2.FaceRecognizerSF_FR_COSINE
    )

    # Both metrics should confirm the identity
    is_match = (
        l2_score <= L2_SIMILARITY_THRESHOLD
        and
        cosine_score >= COSINE_SIMILARITY_THRESHOLD
    )

    return l2_score, cosine_score, is_match


def main():

    detector = cv2.FaceDetectorYN.create(
        YU_NET_MODEL,
        "",
        (320, 320),
        0.8,
        0.3,
        5000
    )

    recognizer = cv2.FaceRecognizerSF.create(SFACE_MODEL,"")
    reference_image = cv2.imread(REFERENCE_IMAGE)

    if reference_image is None:
        raise FileNotFoundError(
            f"Could not load reference image: "
            f"{REFERENCE_IMAGE}"
        )

    reference_faces = detect_faces(reference_image,detector)

    if reference_faces[1] is None:
        raise ValueError(
            "No face found in the reference image."
        )

    if len(reference_faces[1]) > 1:
        print(
            "Warning: Multiple faces detected "
            "in reference image."
        )

    # Use first detected face
    reference_face = reference_faces[1][0]
    reference_feature = extract_face_feature(reference_image,reference_face,recognizer)

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError(
            "Could not open webcam."
        )

    while True:

        success, frame = camera.read()

        if not success:
            print("Could not read frame.")
            break

        faces = detect_faces(frame,detector)

        if faces[1] is None:

            cv2.putText(
                frame,
                "NO FACE DETECTED",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )

        else:

            # Process every detected face
            for face in faces[1]:

                current_feature = extract_face_feature(frame,face,recognizer)
                (
                    l2_score,
                    cosine_score,
                    is_match
                ) = compare_faces(reference_feature,current_feature,recognizer)
    
                if is_match:
                    result_text = "MATCH"
                    box_color = (0, 255, 0)
                else:
                    result_text = "NOT MATCH"
                    box_color = (0, 0, 255)

                draw_face(frame,face,box_color)

                # Get bounding box coordinates
                coords = face[:-1].astype(int)
                x, y, w, h = coords[:4]

                cv2.putText(
                    frame,
                    result_text,
                    (x, max(y - 10, 30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    box_color,
                    2
                )

                # L2 score
                cv2.putText(
                    frame,
                    f"L2: {l2_score:.2f}",
                    (x, y + h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    box_color,
                    2
                )

                # Cosine score
                cv2.putText(
                    frame,
                    f"Cosine: {cosine_score:.2f}",
                    (x, y + h + 48),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    box_color,
                    2
                )

        cv2.imshow(
            "Face Recognition",
            frame
        )

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
