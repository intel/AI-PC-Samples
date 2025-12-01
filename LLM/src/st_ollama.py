import ollama
import streamlit as st

# Set the title of the Streamlit app
st.title("Let's Chat....🐼")


def load_models():
    """
    Load the list of available Ollama models.

    Returns:
        list: A list of model names if successful, otherwise an empty list.
    """
    try:
        response = ollama.list()
        model_list = [model.model for model in response.models]
        return model_list
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return []


def generate_response(user_input, model):
    """
    Generate a response from the selected Ollama model based on user input.

    Args:
        user_input (str): The input text from the user.
        model (str): The name of the selected model.

    Yields:
        str: The generated response content.
    """
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {
                    'role': 'user',
                    'content': user_input,
                },
            ],
            stream=True,
        )
        for res in response:
            yield res["message"]["content"]
    except Exception as e:
        st.error(f"Error generating response: {e}")


# Load Ollama models
model_list = load_models()

# Validate if models are loaded
if model_list:
    model = st.selectbox("Choose a model from the list", model_list)
else:
    st.stop()  # Stop execution if models are not loaded

# Get user input from chat
chat_input = st.chat_input("Hi, How are you?")

# Validate user input
if chat_input:
    with st.spinner("Running....🐎"):
        with st.chat_message("user"):
            st.markdown(chat_input)

        # Stream the generated response
        st.write_stream(generate_response(chat_input, model))
