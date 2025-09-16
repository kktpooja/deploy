import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("student_model_lr.pkl")

def main():
    st.title("Student Performance Prediction")
    st.write("Predict a student's final grade based on age and previous exam scores (G1, G2).")

    # -----------------------------
    # Collect user input (3 features)
    # -----------------------------
    age = st.number_input("Age", min_value=10, max_value=20)
    G1 = st.number_input("G1", min_value=0, max_value=20)
    G2 = st.number_input("G2", min_value=0, max_value=20)

    if st.button("Predict"):
        # Prepare input for the model
        input_data = np.array([[age, G1, G2]])
        
        # Make prediction
        prediction = model.predict(input_data)
        
        st.success(f"Predicted Final Grade (G3): {prediction[0]:.2f}")

# Run the app
if __name__ == "__main__":
    main()


