import streamlit as st
import llm_openrouter as llm


@st.cache_data
def get_models():
    models = llm.available_models()
    models = sorted(models, key=lambda x: x.name)
    return models


st.set_page_config(page_title="LLM Selector", layout="wide")


def configuration():
    with st.expander("Click to configure prompt and models"):
        if "key" not in st.session_state:
            st.session_state.key = ""
        if "prompt" not in st.session_state:
            st.session_state.prompt = ""
        if "temperature" not in st.session_state:
            st.session_state.temperature = 0.0
        if "models" not in st.session_state:
            st.session_state.models = []
        models = get_models()
        st.session_state.prompt = st.text_area("Prompt", height=200)
        st.session_state.temperature = st.slider("Select temperature", 0.0, 2.0, 0.0)
        st.session_state.models = st.multiselect("Choose LLM(s)", models)


configuration()

# Input field for the request and send button
user_input = st.text_area("Enter your request to the LLM", height=150)
send_button = st.button("Send Request")

st.write(f"Selected models: {', '.join([model.name for model in st.session_state.models])}")

# Send request and display response
if send_button:
    with st.spinner("Sending request..."):
        models = st.session_state.models
        if not isinstance(models, list):
            models = [models]
        response = llm.chat_completion_multiple(
            models, st.session_state.prompt, user_input, st.session_state.temperature
        )
    with st.spinner("Calculating cost and stats..."):
        cost_and_stats = llm.cost_and_stats_multiple(response)

    for model, response in response.items():
        st.markdown(f"### {model.name} response")
        with st.expander("Click to see the raw response"):
            st.write("LLM raw response")
            st.json(response.raw_response, expanded=False)
            st.write("Cost and stats raw response")
            st.json(cost_and_stats[model].raw_response, expanded=False)
        st.markdown(response.response)
