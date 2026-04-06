import streamlit as st
from predict import predict_comment

st.title("Toxic Comment Classifier")

st.write("Enter a comment to check if it is toxic.")

text = st.text_area("Comment")

if st.button("Predict"):

    if text.strip() == "":
        st.warning("Please enter a comment.")
    else:
        result = predict_comment(text)

        st.subheader("Prediction")

        for label, value in result.items():
            st.write(f"{label}: {value}")