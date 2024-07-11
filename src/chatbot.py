import streamlit as st

import sys
import os

# Add the directory containing this script to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def chatbot_say(message):
    with st.chat_message("assistant", avatar="🔮"):
        st.markdown(message)

        # Add chatbot response to chat history
        st.session_state.chat_history.append(("assistant", message))

def run_chatbot():
    from perplexity_api import chat_completion

    # Initialise chat history in session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for role, content in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(content)

    # If user has entered a message, add it to chat history and get chatbot response
    if prompt := st.chat_input("Say something: "):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Add user message to chat history
        st.session_state.chat_history.append(("user", prompt))

        info = "none"
        if st.session_state.latest: 
            info = st.session_state.latest
            
        response = chat_completion(prompt, info, mode="normal")

        # Display chatbot response
        with st.chat_message("assistant", avatar="🔮"):
            st.markdown(response)

        # Add chatbot response to chat history
        st.session_state.chat_history.append(("assistant", response))

    
