from pathlib import Path
import sys

import pandas as pd
import streamlit as st

from src.inference import InferenceModel

sys.path.append(str(Path(__file__).resolve().parent.parent))



model = InferenceModel()

st.set_page_config(layout='wide')

st.markdown(
        f"<div style='background-color: #86BC24; padding: 20px; border_radius: 10px; color: black;'>"
        f"<h1>CHURN PREDICTION</h1>"
        f"</div>",
        unsafe_allow_html=True,
        text_alignment='center'
    )
st.markdown('---')

col1, col2 = st.columns(2, gap='large', border=True)

with col1:
    credit_score = st.slider('Credit Score', min_value=1, max_value=1000)
    st.space('xxsmall')
    geography = st.selectbox('Geography', options=model.one_hot_encoder.categories_[0])
    st.space('xxsmall')
    gender = st.selectbox('Gender', options=model.label_encoder.classes_)
    st.space('xxsmall')
    age = st.slider('Age', min_value=18, max_value=100)
    st.space('xxsmall')
    tenure = st.slider('Tenure', min_value=0, max_value=10)

with col2:
    balance = st.number_input('Balance', min_value=0)
    st.space('xxsmall')
    num_of_products = st.slider('Number of Products', min_value=1, max_value=5)
    st.space('xxsmall')
    has_credit_card = st.selectbox('Has Credit Card', options=[0, 1])
    st.space('xxsmall')
    is_active_member = st.selectbox('Is Active Member', options=[0, 1])
    st.space('xxsmall')
    estimated_salary = st.number_input('Estimated Salary', min_value=1)

input_data = {
    'CreditScore': [credit_score],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_credit_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
}

input_data['Gender'] = model.label_encoder.transform(input_data['Gender'])
input_data_df = pd.DataFrame(input_data)

encoded_geo_df = pd.DataFrame(model.one_hot_encoder.transform([[geography]]).toarray(), columns=model.one_hot_encoder.get_feature_names_out(['Geography']))
input_data_df = pd.concat([input_data_df, encoded_geo_df], axis=1)

input_data_scaled = model.standard_scaler.transform(input_data_df)

prediction = model.model.predict(input_data_scaled)
prediction_probability = prediction[0][0]

st.markdown('---')
if (prediction_probability > 0.5):
    st.markdown(
        f"<div style='color: #86BC24;'>"
        f"<h3>The customer will be exited.</h3>"
        f"<p>Churn Probability: {prediction_probability:.2f}</p>"
        f"</div>",
        unsafe_allow_html=True,
        text_alignment='center'
    )
else:
    st.markdown(
        f"<div style='color: #86BC24;'>"
        f"<h3>The customer will not be exited.</h3>"
        f"<p>Churn Probability: {prediction_probability:.2f}</p>"
        f"</div>",
        unsafe_allow_html=True,
        text_alignment='center'
    )
