import json

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

import meal_log

st.set_page_config(page_title="Food Calories from YOLO", page_icon="🍽️", layout="wide")
meal_log.init_db()


@st.cache_data
def load_calorie_db(path: str = "calorie_db.json") -> dict:
    with open(path) as f:
        db = json.load(f)
    return {k: v for k, v in db.items() if not k.startswith("_")}


@st.cache_resource
def load_yolo_model(weights_path: str):
    return YOLO(weights_path)


def normalize_class_name(name: str) -> str:
    return name.lower().replace(" ", "_")


st.title("🍽️ Food Calories Estimator")
st.warning(
    "Calorie values are rough estimates based on standard servings. "
    "YOLO cannot determine ingredients, recipe, weight, or exact portion size."
)

CALORIES_PER_SERVING = load_calorie_db()

with st.sidebar:
    st.header("Model")
    weights_path = st.text_input(
        "Weights file",
        value="yolo26s.pt",
        help="Path to your trained weights, or 'yolo11n.pt'/'yolo26n.pt' for "
        "the pretrained 10-class fallback.",
    )

    st.header("Detection settings")
    conf_threshold = st.slider("Confidence threshold", 0.05, 1.00, 0.25, 0.05)
    imgsz = st.select_slider(
        "Detection resolution (imgsz)", options=[640, 960, 1280, 1536], value=960
    )
    serving_multiplier = st.number_input(
        "Serving size multiplier", min_value=0.25, max_value=5.0, value=1.0, step=0.25
    )
    st.caption("1.0 assumes the detected item is one standard serving.")

    st.markdown(f"### Supported foods ({len(CALORIES_PER_SERVING)})")
    st.write(", ".join(sorted(CALORIES_PER_SERVING)))

detect_tab, history_tab = st.tabs(["Detect", "History"])

with detect_tab:
    if "captured_photo" not in st.session_state:
        st.session_state.captured_photo = None

    upload_tab, camera_tab = st.tabs(["Upload image", "Use camera"])

    with upload_tab:
        uploaded_file = st.file_uploader(
            "Choose a JPG, JPEG, or PNG image", type=["jpg", "jpeg", "png"]
        )

    with camera_tab:
        if st.session_state.captured_photo is None:
            camera_file = st.camera_input("Take a food photo")
            if camera_file is not None:
                st.session_state.captured_photo = camera_file.getvalue()
                st.rerun()
        else:
            st.image(st.session_state.captured_photo, caption="Captured photo", use_container_width=True)
            if st.button("Retake photo"):
                st.session_state.captured_photo = None
                st.rerun()

    image_file = st.session_state.captured_photo or uploaded_file

    if image_file is None:
        st.info("Upload an image or take a photo to begin.")
    else:
        input_image = Image.open(image_file).convert("RGB")
        if st.session_state.captured_photo is None:
            st.image(input_image, caption="Food image", use_container_width=True)

        if st.button("Detect food and estimate calories", type="primary", use_container_width=True):
            with st.spinner("Detecting food..."):
                try:
                    model = load_yolo_model(weights_path)
                except Exception as e:
                    st.error(f"Couldn't load weights at '{weights_path}': {e}")
                    st.stop()

                image_array = np.array(input_image)
                model_names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))

                results = model.predict(
                    source=image_array, conf=conf_threshold, imgsz=imgsz, verbose=False,
                )

            result = results[0]
            boxes = result.boxes
            st.image(Image.fromarray(result.plot()), caption="All detections", use_container_width=True)

            # count instances per class first, so a meal with 3 bananas logs
            # as one line item x3, not three separate rows
            counts = {}
            confidences = {}
            unsupported = []
            for box in boxes:
                raw_name = model_names[int(box.cls[0])]
                class_name = normalize_class_name(raw_name)
                confidence = float(box.conf[0])
                if class_name in CALORIES_PER_SERVING:
                    counts[class_name] = counts.get(class_name, 0) + 1
                    confidences.setdefault(class_name, []).append(confidence)
                else:
                    unsupported.append((raw_name, confidence))

            known_rows = []
            log_items = []
            for class_name, count in counts.items():
                entry = CALORIES_PER_SERVING[class_name]
                per_item = entry["kcal_per_100g"] * entry["avg_portion_g"] / 100 * serving_multiplier
                subtotal = round(per_item * count)
                avg_conf = sum(confidences[class_name]) / len(confidences[class_name])
                known_rows.append(
                    {
                        "Food": class_name.replace("_", " ").title(),
                        "Count": count,
                        "Avg confidence": f"{avg_conf * 100:.1f}%",
                        "Estimated calories": subtotal,
                    }
                )
                log_items.append({"food": class_name, "count": count, "calories": subtotal})

            if known_rows:
                detections = pd.DataFrame(known_rows)
                total_kcal = int(detections["Estimated calories"].sum())

                meal_log.log_meal(log_items, total_kcal)
                st.toast(f"Logged {total_kcal} kcal to today's history", icon="✅")

                st.subheader("Nutrition estimate")
                metric_col, count_col = st.columns(2)
                metric_col.metric("Estimated total", f"{total_kcal} kcal")
                count_col.metric("Food types detected", len(detections))
                st.dataframe(detections, hide_index=True, use_container_width=True)
            else:
                st.warning(
                    "Nothing matched a supported food class. Check the debug "
                    "expander below to see what YOLO actually detected."
                )

            with st.expander(f"Debug: everything YOLO detected ({len(boxes)} total boxes)"):
                if unsupported:
                    st.write("Detected but no calorie data:")
                    st.dataframe(
                        pd.DataFrame(unsupported, columns=["Class", "Confidence"]),
                        hide_index=True, use_container_width=True,
                    )
                if not boxes:
                    st.write("Zero boxes at all. Try lowering confidence or increasing imgsz.")

with history_tab:
    st.subheader("Meal history")

    today_total = meal_log.get_today_total()
    st.metric("Today's total", f"{today_total} kcal")

    days = st.slider("Days to show", 7, 30, 7)
    daily = meal_log.get_daily_totals(days=days)
    daily_df = pd.DataFrame(daily).set_index("date")
    st.bar_chart(daily_df)

    st.markdown("### Recent meals")
    recent = meal_log.get_recent_meals(limit=20)
    if not recent:
        st.info("No meals logged yet — detect some food in the Detect tab first.")
    else:
        for meal in recent:
            item_summary = ", ".join(
                f"{item['count']}x {item['food'].replace('_', ' ')}" for item in meal["items"]
            )
            col1, col2 = st.columns([5, 1])
            with col1:
                st.write(f"**{meal['timestamp'][:16].replace('T', ' ')}** — {meal['total_calories']} kcal — {item_summary}")
            with col2:
                if st.button("Delete", key=f"del_{meal['id']}"):
                    meal_log.delete_meal(meal["id"])
                    st.rerun()