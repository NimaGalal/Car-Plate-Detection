import streamlit as st
from PIL import Image
import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import onnxruntime as ort

try:
    ort_session = ort.InferenceSession("model.onnx")
except Exception as e:
    st.error(f"Could not load model.onnx: {e}")

label_map = list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

# =========================
# Page Title
# =========================
st.title("Car Plate Detection System")
st.markdown("Digital Image Processing Project")

# =========================
# Upload Section
# =========================
st.subheader("Upload Car Image")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

# =========================
# Image Processing
# =========================
if uploaded_file is not None:

    # Read image using OpenCV
    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    img = cv2.imdecode(file_bytes, 1)

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.success("Image Uploaded Successfully")

    # =========================
    # Original Image
    # =========================
    st.subheader("Original Image")
    st.image(img_rgb, width='stretch')

    # =========================
    # GrayScale
    # =========================
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    st.subheader("GrayScale Image")
    st.image(gray, channels="GRAY")

    # =========================
    # Filter Selection
    # =========================
    st.subheader("Noise Removal")

    filter_type = st.selectbox(
        "Choose Filter",
        ["Gaussian Filter", "Median Filter"]
    )

    if filter_type == "Gaussian Filter":
        filtered = cv2.GaussianBlur(gray, (5, 5), 0)

    else:
        filtered = cv2.medianBlur(gray, 5)

    st.image(filtered, channels="GRAY")

    # =========================
    # Histogram Equalization
    # =========================
    equalized = cv2.equalizeHist(filtered)

    st.subheader("Histogram Equalization")
    st.image(equalized, channels="GRAY")
    # =========================
    # Before vs After
    # =========================
    st.subheader("Before vs After Enhancement")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("Before")
        st.image(gray, channels="GRAY")

    with col2:
        st.markdown("After")
        st.image(equalized, channels="GRAY")

    # =========================
    # Thresholding
    # =========================
    _, thresh = cv2.threshold(
        equalized,
        150,
        255,
        cv2.THRESH_BINARY
    )

    st.subheader("Threshold Image")
    st.image(thresh, channels="GRAY")



    # Edge Detection and Plate Segmentation
    st.subheader("Edge Detection(Canny)")

    # Step 1: Canny Edge Detection
    edges = cv2.Canny(equalized, 100, 200)
    st.image(edges, channels="GRAY", caption="Canny Edges")

    # Step 2: Find and draw all contours
    st.subheader("Contour Detection")
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = img_rgb.copy()
    cv2.drawContours(contour_img, contours, -1, (255, 0, 0), 1)
    st.image(contour_img, caption=f"All Contours Found: {len(contours)}", width='stretch')

    # Step 3: manually finding plate location
    img_area = img.shape[0] * img.shape[1]
    plate_candidates = []
    for cont in contours:
        perimeter = cv2.arcLength(cont,True)
        approx = cv2.approxPolyDP(cont,0.01*perimeter,True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            area = w * h
            if (2.0 < aspect_ratio < 5.5) and (0.01 * img_area < area < 0.20 * img_area):
                plate_candidates.append(approx)

    
    if plate_candidates:
        best_plate = min(plate_candidates, key=lambda x: abs((cv2.boundingRect(x)[2]/cv2.boundingRect(x)[3]) - 3.5))
        x, y, w, h = cv2.boundingRect(best_plate)
        plate_img = img_rgb[y:y+h, x:x+w]
        st.image(plate_img, caption="Filtered Plate Detection")

    # Step 4: Read plate location from xml
    st.subheader("Plate Segmentation")

    def get_plate_coords(image_name):
        xml_name = image_name.replace('.png', '.xml').replace('.jpg', '.xml')
        xml_path = f"Dataset/annotations/{xml_name}"
        if not os.path.exists(xml_path):
            return None
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bndbox = root.find('.//bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        return xmin, ymin, xmax, ymax

    # Step 5: Drawing the box & cropping the plate
    coords = get_plate_coords(uploaded_file.name)
    img_copy = img_rgb.copy()

    if coords:
        xmin, ymin, xmax, ymax = coords
        cv2.rectangle(img_copy, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        plate_img = img_rgb[ymin:ymax, xmin:xmax]
        st.image(img_copy, caption="Detected Plate Location", width='stretch')
        st.subheader("Cropped Plate Region")
        st.image(plate_img, caption="Extracted Plate", width='stretch')
        st.success("Plate Successfully Segmented")
    else:
        st.warning("No annotation found ")
    gray_plate = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    ret, thresh_plate = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # finding characters
    st.subheader("Threshold Image")
    st.image(thresh_plate, channels="GRAY")
    contours_plate, _ = cv2.findContours(thresh_plate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = plate_img.copy()
    cv2.drawContours(contour_img, contours_plate, -1, (255, 0, 0), 1)
    st.image(contour_img)

    # filter
    plate_height, plate_width = plate_img.shape[:2]
    chars = []
    for cnt in contours_plate:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if 0.6 < h / plate_height < 0.9 and w / plate_width < 0.3:
            chars.append((x, y, w, h))
    chars = sorted(chars, key=lambda x: x[0])
    for x, y, w, h in chars:
        cv2.rectangle(plate_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        st.image(plate_img[y:y+h, x:x+w])
    st.image(plate_img, caption="Characters")
    
    text = ""
    for char in chars:
        x, y, w, h = char
        char_crop = thresh_plate[y:y+h, x:x+w]
        char_resized = cv2.resize(char_crop, (32, 32))
        char_normalized = char_resized.astype(np.float32) / 255.0
        char_input = char_normalized.reshape(1, 32, 32, 1)
        ort_inputs = {ort_session.get_inputs()[0].name: char_input}
        ort_outs = ort_session.run(None, ort_inputs)
        pred_idx = np.argmax(ort_outs[0])
        text += label_map[pred_idx]

    st.subheader("Plate Text (OCR Result)")
    st.success(f"Detected Plate: {text}")



# =========================
# Dataset Preview
# =========================
st.subheader("Sample Dataset Images")

dataset_path = "dataset/images"

if os.path.exists(dataset_path):

    files = [
        f for f in os.listdir(dataset_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if files:

        cols = st.columns(3)

        for i, file in enumerate(files[:3]):

            img = Image.open(f"{dataset_path}/{file}")

            cols[i].image(
                img,
                caption=file,
                width='stretch'
            )

    else:
        st.warning("No images found in dataset folder")

else:
    st.error("Dataset folder not found")