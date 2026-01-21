import streamlit as st
from PIL import Image
import os
import glob
import numpy as np
from ultralytics import YOLO
import torch
import cv2
from io import BytesIO
import base64

# --------------------------
# 💧 Header
# --------------------------
st.markdown("# Water Preventive Maintenance Classification 💧")  # Level-1 heading

# --------------------------
# 🔍 Load YOLO model
# --------------------------
@st.cache_resource
def load_model():
    model = YOLO("best8_100_16.pt")
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model

model = load_model()

# --------------------------
# 📁 Input path and settings
# --------------------------
base_dir = r"D:\Code\PMCMApp-win32-x64\PMCMApp-win32-x64\resources\app\downloads\2025-06-01_to_2025-06-01"
valid_tank_types = ["ถังน้ำดื่ม", "ถังน้ำใช้"]

# Bounding box color map (BGR)
color_map = {
    "correct": (0, 255, 0),
    "incorrect": (0, 0, 255),
}

# Helper: Convert image to base64 for HTML embedding
def pil_to_base64(img):
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

# --------------------------
# 🚀 Main detection loop
# --------------------------
if os.path.exists(base_dir):
    for store_id in os.listdir(base_dir):
        store_path = os.path.join(base_dir, store_id)
        if not os.path.isdir(store_path):
            continue

        for tank_type in os.listdir(store_path):
            if tank_type not in valid_tank_types:
                continue

            tank_path = os.path.join(store_path, tank_type)
            if not os.path.isdir(tank_path):
                continue

            for image_group in os.listdir(tank_path):
                group_path = os.path.join(tank_path, image_group)
                if not os.path.isdir(group_path):
                    continue

                image_paths = glob.glob(os.path.join(group_path, "*.jpg")) + glob.glob(os.path.join(group_path, "*.png"))
                best_result = None

                for image_path in image_paths:
                    image_pil = Image.open(image_path).convert("RGB")
                    resized_image = image_pil.resize((500, 500))
                    results = model.predict(np.array(resized_image))
                    detections = results[0].boxes
                    class_count = {}
                    max_confidence = {}

                    if detections is not None:
                        for box in detections:
                            class_name = results[0].names[int(box.cls)]
                            confidence = box.conf.item() * 100
                            class_count[class_name] = class_count.get(class_name, 0) + 1
                            max_confidence[class_name] = max(max_confidence.get(class_name, 0), confidence)

                    if not class_count:
                        final_class = "undetected"
                    else:
                        detected_classes = set(class_count.keys())
                        if "correct" in detected_classes and len(detected_classes) > 1:
                            final_class = "check"
                        elif "correct" in detected_classes and max_confidence["correct"] < 80:
                            final_class = "check"
                        elif "incorrect" in detected_classes and max_confidence["incorrect"] < 80:
                            final_class = "check"
                        elif len(detected_classes) > 1:
                            final_class = "incorrect"
                        else:
                            final_class = list(detected_classes)[0]

                    if final_class == "correct":
                        best_result = ("correct", image_path, image_pil, results, max_confidence.get("correct", 0.0))
                        break
                    elif final_class == "incorrect" and (best_result is None or best_result[0] != "correct"):
                        best_result = ("incorrect", image_path, image_pil, results, max_confidence.get("incorrect", 0.0))
                    elif final_class == "check" and best_result is None:
                        best_result = ("check", image_path, image_pil, results, 0.0)

                # === DISPLAY RESULT FOR THIS TANK GROUP ===
                if best_result:
                    final_class, image_path, image_pil, results, confidence = best_result

                    st.markdown("---")
                    st.markdown(f"### 🏬 Store: `{store_id}`")
                    tank_icon = "💧" if tank_type == "ถังน้ำดื่ม" else "🚿"
                    st.markdown(f"### {tank_icon} Tank Type: `{tank_type}`")
                    

                    # Resize both images
                    resized_image = image_pil.resize((500, 500))
                    image_cv = np.array(resized_image)
                    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

                    detections = results[0].boxes
                    if detections is not None:
                        for box in detections:
                            class_id = int(box.cls)
                            class_name = results[0].names[class_id]
                            conf = box.conf.item()
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            color = color_map.get(class_name, (0, 255, 255))
                            label = f"{class_name} {conf:.1%}"
                            cv2.rectangle(image_cv, (x1, y1), (x2, y2), color, 6)
                            cv2.putText(image_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # Convert result image for HTML rendering
                    image_result = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
                    result_pil = Image.fromarray(image_result)
                    result_base64 = pil_to_base64(result_pil)

                    # Border color by class
                    border_color = {
                        "correct": "green",
                        "incorrect": "red",
                        "check": "orange"
                    }.get(final_class, "gray")

                    # Use columns layout
                    col1, col2 = st.columns(2)

                    with col1:
                        st.image(resized_image, caption="Original")

                    with col2:
                        st.markdown(
                            f"""
                            <div style="border: 4px solid {border_color}; width: fit-content; padding: 4px; border-radius: 8px; margin: auto;">
                                <img src="data:image/png;base64,{result_base64}" width="500"/>
                                <p style="text-align:center; margin-top: 8px;">Detection → {final_class.upper()}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    # Classification result
                    if final_class == "correct":
                        st.success(f"✅ Result: CORRECT ({confidence:.1f}%)")
                    elif final_class == "incorrect":
                        st.error(f"❌ Result: INCORRECT ({confidence:.1f}%)")
                    else:
                        st.warning(f"⚠️ Result: CHECK")
else:
    st.warning("❌ Invalid base path. Please check again.")
