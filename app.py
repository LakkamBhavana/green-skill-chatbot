import streamlit as st
import joblib

# Load the trained model
model = joblib.load("green_skill_chatbot_model.pkl")

# Function to get the chatbot's response
def get_bot_response(user_input):
    return model.predict([user_input])[0]

# Streamlit interface
st.title("Green Skill Technology Chatbot")

# Get user input
user_input = st.text_input("Ask me anything about Green Skill Technology:")

if user_input:
    # Get the chatbot's response
    response = get_bot_response(user_input)
    st.write(f"Chatbot Response: {response}")
