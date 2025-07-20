

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)


def calculate_bmr(gender_code, weight, height, age):
    return (10 * weight) + (6.25 * height) - (5 * age) + (5 if gender_code == 1 else -161)

@st.cache_resource
def load_assets():
    try:
        model = joblib.load('assets/catboost_model.pkl')
        explainer = joblib.load('assets/shap_explainer.pkl')
        feature_columns = joblib.load('assets/feature_columns.pkl')
        return model, explainer, feature_columns
    except FileNotFoundError:
        st.error("Model assets not found. Please run the training script first.")
        return None, None, None

model, explainer, feature_columns = load_assets()


@st.cache_resource
def get_github_client():
    token = st.secrets["OPENAI_API_KEY"]  # GitHub PAT
    endpoint = "https://models.github.ai/inference"
    try:
        client = ChatCompletionsClient(endpoint=endpoint, credential=AzureKeyCredential(token))
        return client
    except Exception as e:
        st.error(f"Failed to initialize GitHub inference client: {e}")
        return None

github_client = get_github_client()
github_model = "openai/gpt-4.1"

def ask_github_gpt(messages, temperature=0.5, top_p=1):
    if github_client is None:
        return "GitHub inference client not initialized."
    try:
        response = github_client.complete(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            model=github_model,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"GitHub inference call error: {e}")
        return "Error retrieving response from AI."


if 'page' not in st.session_state:
    st.session_state.page = 'welcome'


if st.session_state.page == 'welcome':
    st.title("🩺 Welcome to the Personalized Diabetes Risk Predictor")
    st.markdown("""
    This tool uses a machine learning model to estimate your diabetes risk based on key health and lifestyle factors.

    **How it works:**
    1. Enter your data anonymously.
    2. Get an instant risk score.
    3. Receive personalized insights and chat with an AI health assistant.

    *Disclaimer: This is an informational tool and not a substitute for professional medical advice.*
    """)
    if st.button("Get Started", type="primary"):
        st.session_state.page = 'input_form'
        st.rerun()

elif st.session_state.page == 'input_form':
    if not model:
        st.stop()

    st.title("👤 Enter Your Information")
    with st.form("user_input_form"):
        st.subheader("Personal & Lifestyle")
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("First Name (for personalization)", "Alex")
            age = st.number_input("Age", 18, 80, 45)
            sex = st.selectbox("Sex", ("Male", "Female"))
            weight_kg = st.number_input("Weight (kg)", 30.0, 200.0, 85.0)
            height_cm = st.number_input("Height (cm)", 100.0, 250.0, 175.0)
        with col2:
            physical_activity = st.selectbox("Physical Activity",
                                             ("Active (Regular Exercise)", "Sedentary (Little to No Exercise)"))
            smoker = st.selectbox("Do you smoke?", ("No", "Yes"))
            family_history = st.selectbox("Family History of Diabetes?", ("No", "Yes"))
            high_bp = st.selectbox("Ever told you have High Blood Pressure?", ("No", "Yes"))

        st.subheader("Health & Dietary Metrics")
        col3, col4 = st.columns(2)
        with col3:
            waist_circumference = st.number_input("Waist Circumference (cm)", 50.0, 200.0, 98.0)
            calories = st.number_input("Average Daily Calorie Intake (kcal)", 500, 6000, 2500)
            sugar = st.number_input("Average Daily Sugar Intake (grams)", 0.0, 500.0, 110.0)
        with col4:
            numbness = st.selectbox("Experience numbness in hands/feet?", ("No", "Yes"))
            foot_problems = st.selectbox("Have non-healing foot sores?", ("No", "Yes"))
            kidney_problems = st.selectbox("Ever told you have Kidney Problems?", ("No", "Yes"))

        submitted = st.form_submit_button("Analyze My Risk")

        if submitted:
            user_input = {
                'Name': name,
                'Age': age,
                'Sex': 1 if sex == 'Male' else 0,
                'Weight': weight_kg,
                'Height': height_cm,
                'PhysicalActivity': 1 if physical_activity == 'Active (Regular Exercise)' else 0,
                'Smoker': 1 if smoker == 'Yes' else 0,
                'FamilyHistoryDiabetes': 1 if family_history == 'Yes' else 0,
                'HadHighBP': 1 if high_bp == 'Yes' else 0,
                'Numbness': 1 if numbness == 'Yes' else 0,
                'FootProblems': 1 if foot_problems == 'Yes' else 0,
                'KidneyProblems': 1 if kidney_problems == 'Yes' else 0,
                'WaistCircumference': waist_circumference,
                'Calories': calories,
                'Sugar': sugar,
            }
            user_input['BMI'] = user_input['Weight'] / ((user_input['Height'] / 100) ** 2)
            user_input['BMR'] = calculate_bmr(
                user_input['Sex'], user_input['Weight'], user_input['Height'], user_input['Age'])
            pa_for_eer = 1 if user_input['PhysicalActivity'] == 1 else 2
            user_input['EER'] = user_input['BMR'] * (1.55 if pa_for_eer == 1 else 1.2)

            input_df = pd.DataFrame([user_input])[feature_columns]

            st.session_state.user_data = user_input
            st.session_state.input_df = input_df
            st.session_state.page = 'report'
            st.rerun()

elif st.session_state.page == 'report':
    user_data = st.session_state.user_data
    input_df = st.session_state.input_df

    st.title(f"📊 Your Personalized Diabetes Risk Report, {user_data.get('Name')}")

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][int(prediction)]
    shap_values = explainer(input_df)

    pred_label = "High Risk" if prediction == 1 else "Low Risk"
    prob_percent = probability * 100

    if prediction == 1:
        st.error(f"**Prediction: {pred_label} of Diabetes** (Confidence: {prob_percent:.1f}%)")
    else:
        st.success(f"**Prediction: {pred_label} of Diabetes** (Confidence: {prob_percent:.1f}%)")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Your Profile")
        st.metric("Age", f"{user_data['Age']} years")
        st.metric("Body Mass Index (BMI)", f"{user_data['BMI']:.1f}")
        st.metric("Waist Circumference", f"{user_data['WaistCircumference']} cm")
    with col2:
        st.subheader("Estimated Energy Needs")
        st.metric("Basal Metabolic Rate (BMR)", f"{user_data['BMR']:.0f} kcal/day")
        st.metric("Estimated Energy Requirement (EER)", f"{user_data['EER']:.0f} kcal/day")

    st.markdown("---")
    st.subheader("Key Factors Influencing Your Risk")
    st.markdown("Our AI analyzed how each of your inputs contributed to your risk score. Factors pushing the score higher are shown below.")

    shap_feature_pairs = list(zip(shap_values.values[0], shap_values.feature_names))
    risk_features = sorted([(val, feat) for val, feat in shap_feature_pairs if val > 0], key=lambda x: x[0], reverse=True)[:5]

    raw_recommendations = "Factors increasing risk:\n"
    if risk_features:
        for val, feature in risk_features:
            raw_recommendations += f"- {feature} (Impact score: {val:.2f})\n"
    else:
        raw_recommendations = "No significant factors found increasing your risk."


    num_features = len(shap_values[0].values)
    fig_height = min(2.8, max(2, num_features * 0.15))  # super compact!
    st.subheader("SHAP Waterfall Plot Explaining Your Prediction")
    shap_values_instance = shap.Explanation(
        values=shap_values[0].values,
        base_values=explainer.expected_value,
        data=input_df.iloc[0].values,
        feature_names=input_df.columns,
    )
    fig, ax = plt.subplots(figsize=(8, fig_height))
    shap.plots.waterfall(shap_values_instance, max_display=8, show=False)
    plt.tight_layout()
    st.pyplot(fig)



    prompt = (
        f"You are a friendly health AI assistant. A user named {user_data['Name']} has received a diabetes risk assessment.\n"
        f"Their data is: Age={user_data['Age']}, Sex={'Male' if user_data['Sex'] == 1 else 'Female'}, "
        f"BMI={user_data['BMI']:.1f}, Waist={user_data['WaistCircumference']}, Smoker={'Yes' if user_data['Smoker'] == 1 else 'No'}.\n"
        f"Their prediction is: '{pred_label}' of diabetes.\n"
        f"The model identified these factors as increasing their risk:\n{raw_recommendations}\n"
        f"Explain these results simply and positively for {user_data['Name']}."
    )

    messages = [
        SystemMessage(content="You are a helpful health AI assistant."),
        UserMessage(content=prompt),
    ]

    st.subheader("Personalized Recommendations")
    with st.spinner("Generating personalized advice..."):
        advice = ask_github_gpt(messages)
        st.info(advice)


    if "chat_messages" not in st.session_state:
        assistant_msg = (
            f"Here is the diabetes risk prediction and recommendation for the user:\n\n"
            f"Prediction: {pred_label} (Confidence: {prob_percent:.1f}%)\n\n"
            f"Recommendations:\n{advice}"
        )
        st.session_state.chat_messages = [
            {"role": "assistant", "content": assistant_msg},
        ]

    st.markdown("---")
    st.subheader("💬 Chat with Your Health Assistant")
    st.markdown("Ask for diet plans, workout ideas, or more details about your results.")

    # Display chat history (skip any "system" role messages)
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask me anything about your health:"):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)


        messages_for_ai = [
            SystemMessage(content="You are a helpful health AI assistant. Speak informatively, positively, and continue the existing conversation with the user after their diabetes prediction and recommendations.")
        ]
        for m in st.session_state.chat_messages:
            messages_for_ai.append(UserMessage(content=m["content"]) if m["role"] != "user" else UserMessage(content=m["content"]))

        response_text = ask_github_gpt(messages_for_ai)
        st.session_state.chat_messages.append({"role": "assistant", "content": response_text})
        with st.chat_message("assistant"):
            st.markdown(response_text)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Start a New Prediction"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
