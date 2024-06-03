import streamlit as st
import os
from dotenv import load_dotenv  # Import load_dotenv to load the .env file
import time
from groq import Groq
from langchain.chains import ConversationChain, LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import speech_recognition as sr

# Load environment variables from the .env file
load_dotenv()

def display_typing_effect(text, delay):
    """
    Display text with a typing effect in Streamlit.
    Args:
    - text (str): The text to display.
    - delay (float): The delay between each letter in seconds.
    """
    placeholder = st.empty()
    typed_text = ""
    for letter in text:
        typed_text += letter
        placeholder.markdown(f'<div style="word-wrap: break-word;">{typed_text}</div>', unsafe_allow_html=True)
        time.sleep(delay)

def speech_to_text():
    """
    Function to convert speech to text
    """
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        st.write("Sorry, I could not understand your speech.")
        return None
    except sr.RequestError as e:
        st.write(f"Sorry, an error occurred: {e}")
        return None

def display_connect_with_me():
    """
    Function to display "Connect with Me" section
    """
    st.sidebar.markdown("""
        <h3 align="left">Connect with me:</h3>
        <p align="left">
        <a href="https://twitter.com/Adityak22723056" target="_blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/twitter.svg" alt="Twitter" height="30" width="40" /></a>
        <a href="https://linkedin.com/in/aditya-kumar-tiwari-a14547232" target="_blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/linked-in-alt.svg" alt="LinkedIn" height="30" width="40" /></a>
        <a href="https://kaggle.com/aditya0kumar0tiwari" target="_blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/kaggle.svg" alt="Kaggle" height="30" width="40" /></a>
        <a href="https://instagram.com/_aadi_anant" target="_blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/instagram.svg" alt="Instagram" height="30" width="40" /></a>
        <a href="https://www.leetcode.com/_aadi_anant" target="_blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/leet-code.svg" alt="LeetCode" height="30" width="40" /></a>
        </p>
    """, unsafe_allow_html=True)

def evaluate_response(response, expected_answer):
    """
    Function to evaluate the response based on predefined criteria.
    Args:
    - response (str): The response given by the user.
    - expected_answer (str): The expected correct answer.
    
    Returns:
    - dict: Scores and feedback based on the evaluation criteria.
    """
    # Sample scoring criteria
    clarity = 1 if "clear" in response else 0
    correctness = 1 if response == expected_answer else 0
    depth_of_knowledge = 1 if len(response.split()) > 10 else 0
    relevance = 1 if "relevant" in response else 0
    
    total_score = clarity + correctness + depth_of_knowledge + relevance
    
    feedback = {
        "Clarity": clarity,
        "Correctness": correctness,
        "Depth of Knowledge": depth_of_knowledge,
        "Relevance": relevance,
        "Total Score": total_score,
        "Feedback": f"Your answer scored {total_score}/4. Clarity: {clarity}, Correctness: {correctness}, Depth of Knowledge: {depth_of_knowledge}, Relevance: {relevance}"
    }
    
    return feedback

def main():
    """
    This function is the main entry point of the application. It sets up the Groq client, the Streamlit interface, and handles the chat interaction.
    """
    st.set_page_config(page_title='FlexiBot', layout='wide')

    # Apply custom CSS to improve text layout
    st.markdown(
        """
        <style>
        .chat-response {
            word-wrap: break-word;
            white-space: pre-wrap;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )

    # Get Groq API key
    groq_api_key = os.getenv('GROQ_API_KEY')
    if not groq_api_key:
        st.error("GROQ_API_KEY is not set. Please ensure the .env file is correctly set up.")
        return

    # Display the Groq logo
    spacer, col = st.columns([5, 1])  
    with col:  
        st.image('FlexiBot.jpg')

    # The title and greeting message of the Streamlit application
    st.title("Chat with FlexiBot!")
    st.write("Hi! I'm FlexiBot, your responsive and friendly chatbot. I can help with questions, provide information, or just chat for fun. And I'm super quick! Let's begin our conversation!")

    # Add customization options to the sidebar
    st.sidebar.title('Customization')
    system_prompt = st.sidebar.text_input("System prompt:")
    model = st.sidebar.selectbox(
        'Choose a model',
        ['llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it']
    )
    conversational_memory_length = st.sidebar.slider('Conversational memory length:', 1, 10, value=5)
    
    # Add a slider for typing speed in milliseconds
    typing_speed = st.sidebar.slider('Typing speed (ms per letter):', 1, 100, value=50)
    typing_delay = typing_speed / 1000  # Convert milliseconds to seconds

    memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)

    # Check for chat history in session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    else:
        for message in st.session_state.chat_history:
            memory.save_context(
                {'input': message['human']},
                {'output': message['AI']}
            )

    # Initialize Groq Langchain chat object and conversation
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key, 
        model_name=model
    )

    # If the user has asked a question,
    user_question = st.text_input("Ask a question:")
    if user_question:
        # Construct a chat prompt template using various components
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{human_input}"),
            ]
        )

        # Create a conversation chain using the LangChain LLM (Language Learning Model)
        conversation = LLMChain(
            llm=groq_chat,
            prompt=prompt,
            verbose=True,
            memory=memory,
        )

        # The chatbot's answer is generated by sending the full prompt to the Groq API.
        response = conversation.predict(human_input=user_question)
        message = {'human': user_question, 'AI': response}
        st.session_state.chat_history.append(message)
        display_typing_effect(response, typing_delay)  # Use the typing effect function with delay
        
        # Provide evaluation and feedback
        st.subheader("Evaluation and Feedback")
        expected_answer = "sample expected answer"  # Placeholder for the expected correct answer
        feedback = evaluate_response(response, expected_answer)
        st.write(feedback["Feedback"])

    # Speech-to-text section
    if st.button("Use Speech Input"):
        user_question = speech_to_text()
        if user_question:
            st.write(f"User Question: {user_question}")
            # Construct a chat prompt template using various components
            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=system_prompt),
                    MessagesPlaceholder(variable_name="chat_history"),
                    HumanMessagePromptTemplate.from_template("{human_input}"),
                ]
            )

            # Create a conversation chain using the LangChain LLM (Language Learning Model)
            conversation = LLMChain(
                llm=groq_chat,
                prompt=prompt,
                verbose=True,
                memory=memory,
            )

            # The chatbot's answer is generated by sending the full prompt to the Groq API.
            response = conversation.predict(human_input=user_question)
            message = {'human': user_question, 'AI': response}
            st.session_state.chat_history.append(message)
            display_typing_effect(response, typing_delay)  # Use the typing effect function with delay
            
            # Provide evaluation and feedback
            st.subheader("Evaluation and Feedback")
            expected_answer = "sample expected answer"  # Placeholder for the expected correct answer
            feedback = evaluate_response(response, expected_answer)
            st.write(feedback["Feedback"])

    # Display "About the Author" section
    st.sidebar.title("About the Author")
    st.sidebar.markdown("""
    Hi there! I'm Aditya, an aspiring ML engineer with a passion for machine learning, deep learning, and problem-solving. I love working on diverse projects and exploring new technologies. Connect with me to stay updated on my latest projects and endeavors!
    """)

    # Display "Connect with Me" section
    display_connect_with_me()

if __name__ == "__main__":
    main()
