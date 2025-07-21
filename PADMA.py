import threading
import base64
import requests
from io import BytesIO
from PIL import Image
import streamlit as st
import numpy as np
import cv2


def process_image(image):
    image = image.convert("RGB").resize((224, 224))
    img_np = np.array(image)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    b, g, r = cv2.split(img_np)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    b, g, r = clahe.apply(b), clahe.apply(g), clahe.apply(r)
    processed_img = cv2.merge((b, g, r))
    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)

    return image, Image.fromarray(processed_img)


def predict_image(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

    try:
        response = requests.post("http://127.0.0.1:5000/predict", json={"file": img_data})
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


# Generate file content for download
def generate_file_content(results):
    file_content = "Prediction Results:\n"
    for label, prob in zip(results["labels"], results["probabilities"]):
        file_content += f"{label}: {float(prob):.2f}%\n"
    return file_content


# Main application logic
def main():
    # Display the logo at the top
    st.image("logo.png", use_container_width=True)

    st.title("Diabetic Foot Ulcer Monitoring and Severity Assessment - PADMA")

    def change_label_style(label, font_size='12px', font_color='black', font_family='sans-serif'):
        # JavaScript to update the label style
        html = f"""
        <script>
            const elems = window.parent.document.querySelectorAll('p');
            const elem = Array.from(elems).find(x => x.innerText.trim() === '{label}');
            if (elem) {{
                elem.style.margin = '0'; /* Remove extra margin */
                elem.style.padding = '0'; /* Remove extra padding */
                elem.style.fontSize = '{font_size}';
                elem.style.color = '{font_color}';
                elem.style.fontFamily = '{font_family}';
            }}
        </script>
        """
        # Inject the HTML into Streamlit
        st.components.v1.html(html, height=0)

        # Initialize session state for page navigation
    if "page" not in st.session_state:
        st.session_state.page = "Registration"

    def navigate_to(page_name):
        st.session_state.page = page_name

    # Page 1: Registration
    if st.session_state.page == "Registration":
        st.subheader("User Registration")

        label = "Full Name"
        st.text_input(label, key="full_name")
        change_label_style(label, '20px')

        label1 = "Email"
        st.text_input(label1, key="email")
        change_label_style(label1, '20px')


        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Next"):
                if st.session_state.full_name and st.session_state.email:
                    navigate_to("Personal Details")
                else:
                    st.error("Please enter your full name and email.")

    # Page 2: Personal Details
    elif st.session_state.page == "Personal Details":
        st.subheader("Personal Details")
        st.text_input("Name:", key="name")
        st.date_input("Date of Birth:", key="dob")
        st.number_input("Age:", min_value=1, max_value=120, step=1, key="age")
        st.selectbox("Gender:", ["Male", "Female", "Other"], key="gender")
        st.text_input("Address:", key="address")
        st.number_input("Pincode:", min_value=0, step=1, key="pincode")

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Back"):
                navigate_to("Registration")
        with col2:
            if st.button("Next"):
                navigate_to("Medical Questionnaire")

    # Page 3: Medical Questionnaire
    elif st.session_state.page == "Medical Questionnaire":
        st.title("Medical Questionnaire")
        high_bp = st.radio("Do you have high blood pressure?", options=["Yes", "No"], key="high_bp")
        diabetes = st.radio("Do you have diabetes?", options=["Yes", "No"], key="diabetes")

        if diabetes == "Yes":
            st.number_input("How many years have you had diabetes?", min_value=0, step=1, key="diabetes_years")
        else:
            st.radio("Do you have ulcers on your foot?", options=["Yes", "No"], key="ulcers")

        st.text_area("Please list any other current medical conditions:", key="medical_conditions")
        st.text_area("Are you taking any medications?", key="medications")

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Back"):
                navigate_to("Personal Details")
        with col2:
            if st.button("Next"):
                navigate_to("Image Upload")

    # Page 4: Image Upload and Prediction
    elif st.session_state.page == "Image Upload":
        st.title("Image Upload and Prediction")

        uploaded_file = st.file_uploader("Upload an image (JPG, JPEG, PNG):", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            image = Image.open(uploaded_file)
            _, processed_image = process_image(image)

            # Show processed image
            st.image(processed_image, caption="Processed Image for Prediction", use_container_width=True)

            if st.button("Predict"):
                predictions = predict_image(processed_image)

                if predictions and "probabilities" in predictions:
                    st.subheader("Predictions:")
                    for label, prob in zip(predictions["labels"], predictions["probabilities"]):
                        st.write(f"{label}: {float(prob):.2f}%")

                    # Generate file content
                    file_content = generate_file_content(predictions)

                    # Provide Save Results button for direct download
                    st.download_button(
                        label="Save Results",
                        data=file_content,
                        file_name="prediction_results.txt",
                        mime="text/plain",
                    )

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Back"):
                navigate_to("Medical Questionnaire")


def start_fastapi():
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=5000)


if __name__ == "__main__":
    threading.Thread(target=start_fastapi, daemon=True).start()
    main()
