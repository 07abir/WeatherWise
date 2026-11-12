from datetime import date
import requests
import streamlit as st

st.set_page_config(
    page_title="WeatherWise | Historical Weather Classifier",
    page_icon="🌤️",
    layout="centered",
)

API_URL = "http://127.0.0.1:8000"

st.title("🌤️ WeatherWise")
st.caption("Historical-pattern weather classification")

st.info(
    "This version is trained on Kolkata historical data. "
    "It estimates the historically learned weather category for a calendar date; "
    "it does not provide a real meteorological forecast."
)

with st.container(border=True):
    st.subheader("Choose a date")
    location = st.text_input("Location", value="Kolkata", disabled=True)
    selected_date = st.date_input(
        "Select prediction date",
        value=date.today(),
        min_value=date(2000, 1, 1),
        max_value=date(2100, 12, 31),
    )

    predict_clicked = st.button("Predict weather", type="primary", use_container_width=True)

if predict_clicked:
    request_data = {
        "prediction_date": selected_date.isoformat(),
        "location": location,
    }

    with st.spinner("Running your trained Decision Tree model..."):
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json=request_data,
                timeout=20,
            )
            if response.ok:
                result = response.json()
                st.success("Prediction completed")

                st.metric("Predicted weather category", result["prediction"])
                st.write(f"**Location:** {result['location']}")
                st.write(f"**Date:** {result['date']}")
                st.caption(f"Model: {result['model']}")

                scores = result.get("class_scores", {})
                if scores:
                    st.subheader("Decision-tree class scores")
                    st.caption(
                        "These are the model's leaf class proportions, not calibrated "
                        "probabilities of what the actual weather will be."
                    )
                    for label, score in scores.items():
                        st.write(f"**{label}** — {score:.1%}")
                        st.progress(float(score))

                st.warning(result["interpretation"])
            else:
                try:
                    detail = response.json().get("detail", response.text)
                except ValueError:
                    detail = response.text
                st.error(f"API error ({response.status_code}): {detail}")

        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot connect to FastAPI. Start the backend first in another terminal:\n\n"
                "`uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000`"
            )
        except requests.exceptions.RequestException as exc:
            st.error(f"Request failed: {exc}")

st.divider()
st.subheader("About this model")
st.write(
    "Your saved artifacts include a Decision Tree Classifier, SimpleImputer, "
    "StandardScaler, LabelEncoder, and feature-column list. "
    "The application uses the four features expected by your model: month, "
    "day_of_year, latitude, and longitude."
)
st.caption(
    "Model limitation: latitude and longitude were constant in the training data, "
    "so this model should only be used for Kolkata unless it is retrained with "
    "representative data from additional locations."
)
