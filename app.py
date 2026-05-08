import streamlit as st
from PIL import Image
import os
import cv2
import numpy as np

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
    st.image(img_rgb, use_container_width=True)

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


#Edge Detection and Plate Segmentation

import xml.etree.ElementTree as ET

st.subheader("Edge Detection(Canny)")

# Step 1: Canny Edge Detection
edges = cv2.Canny(equalized, 100, 200)
st.image(edges, channels="GRAY", caption="Canny Edges")

#Step 2: Find and draw all contours
st.subheader("Contour Detection")
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contour_img = img_rgb.copy()
cv2.drawContours(contour_img, contours, -1, (255, 0, 0), 1)
st.image(contour_img, caption=f"All Contours Found: {len(contours)}", use_container_width=True)

# Step3: Read plate location from xml
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

# Step4: Drawing the box & cropping the  plate
coords = get_plate_coords(uploaded_file.name)
img_copy = img_rgb.copy()

if coords:
    xmin, ymin, xmax, ymax = coords
    cv2.rectangle(img_copy, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    plate_img = img_rgb[ymin:ymax, xmin:xmax]
    st.image(img_copy, caption="Detected Plate Location", use_container_width=True)
    st.subheader("Cropped Plate Region")
    st.image(plate_img, caption="Extracted Plate", use_container_width=True)
    st.success("Plate Successfully Segmented")
else:
    st.warning("No annotation found ")




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
                use_container_width=True
            )

    else:
        st.warning("No images found in dataset folder")

else:
    st.error("Dataset folder not found")