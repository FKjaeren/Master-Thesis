import os
import json
import requests
import SessionState
import streamlit as st


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "masterthesis-366109-eeac07888cf3.json" # change for your GCP key
PROJECT = "MasterThesis" # change for your GCP project
REGION = "eu-central2" # change for your GCP region (where your model is hosted)

st.title("Make a customer recommendation")
st.header("Type the id of the custommer which recommendations is wanted.")

def preprocess_recommendatio_data(customer_id):
    return data
def Predict_id(data, model, project, region):
    return prediction

classes_and_models = {
    "model_1": {
        "model_name": "efficientnet_model_1_10_classes" # change to be your model name
    },
    "model_2": {
        "model_name": "efficientnet_model_2_11_classes"
    },
    "model_3": {
        "model_name": "efficientnet_model_3_12_classes"
    }
}

@st.cache # cache the function so predictions aren't always redone (Streamlit refreshes every click)
def make_prediction(customer_id, model):
    """
    Takes an image and uses model (a trained TensorFlow model) to make a
    prediction.
    Returns:
     image (preproccessed)
     pred_class (prediction class from class_names)
     pred_conf (model confidence)
    """
    data = preprocess_recommendatio_data(customer_id)
    # Turn tensors into int16 (saves a lot of space, ML Engine has a limit of 1.5MB per request)
    # image = tf.expand_dims(image, axis=0)
    preds = Predict_id(project=PROJECT,
                         region=REGION,
                         model=model,
                         instances=data)
    pred_class = preds[0]
    pred_conf = preds[1]
    return customer_id, pred_class, pred_conf

choose_model = st.sidebar.selectbox(
    "Pick model you'd like to use",
    ("Model 1 (10 food classes)", # original 10 classes
     "Model 2 (11 food classes)", # original 10 classes + donuts
     "Model 3 (11 food classes + non-food class)") # 11 classes (same as above) + not_food class
)

# Model choice logic
if choose_model == "Model 1":
    MODEL = classes_and_models["model_1"]["model_name"]
elif choose_model == "Model 2":
    MODEL = classes_and_models["model_2"]["model_name"]
else:
    MODEL = classes_and_models["model_3"]["model_name"]

# Display info about model and classes
if st.checkbox("Show models"):
    st.write(f"You chose {MODEL}")

session_state = SessionState.get(pred_button=False)

# Create logic for app flow
"""
if not uploaded_file:
    st.warning("Please upload a customer ID.")
    st.stop()
else:
    session_state.uploaded_image = uploaded_file.read()
    st.image(session_state.uploaded_image, use_column_width=True)
    
"""
pred_button = st.button("Recommend")
# Did the user press the predict button?
if pred_button:
    session_state.pred_button = True 

# And if they did...
if session_state.pred_button:
    session_state.image, session_state.pred_class, session_state.pred_conf = make_prediction(session_state.uploaded_image, model=MODEL)
    st.write(f"Prediction: {session_state.pred_class}, \
               Confidence: {session_state.pred_conf:.3f}")
    """
    # Create feedback mechanism (building a data flywheel)
    session_state.feedback = st.selectbox(
        "Is this correct?",
        ("Select an option", "Yes", "No"))
    if session_state.feedback == "Select an option":
        pass
    elif session_state.feedback == "Yes":
        st.write("Thank you for your feedback!")
        # Log prediction information to terminal (this could be stored in Big Query or something...)
        print(update_logger(image=session_state.image,
                            model_used=MODEL,
                            pred_class=session_state.pred_class,
                            pred_conf=session_state.pred_conf,
                            correct=True))
    elif session_state.feedback == "No":
        session_state.correct_class = st.text_input("What should the correct label be?")
        if session_state.correct_class:
            st.write("Thank you for that, we'll use your help to make our model better!")
            # Log prediction information to terminal (this could be stored in Big Query or something...)
            print(update_logger(image=session_state.image,
                                model_used=MODEL,
                                pred_class=session_state.pred_class,
                                pred_conf=session_state.pred_conf,
                                correct=False,
                                user_label=session_state.correct_class))
    """