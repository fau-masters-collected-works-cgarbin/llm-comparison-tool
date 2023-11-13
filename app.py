import streamlit as st
import llm_openrouter as llm


@st.cache_data
def get_models():
    models = llm.available_models()
    models = sorted(models, key=lambda x: x.name)
    return models


st.set_page_config(page_title="LLM Selector", layout="wide")

# Sidebar for model selection and temperature
models = get_models()
selected_models = st.sidebar.multiselect("Choose LLM(s)", models)
temperature = st.sidebar.slider("Select temperature", 0.0, 2.0, 0.0)

with st.expander("Prompt"):
    prompt = st.text_area("Prompt", height=200)

# Input field for the request and send button
user_input = st.text_area("Enter your request to the LLM", height=150)
send_button = st.button("Send Request")

# Send request and display response
if send_button:
    with st.spinner("Sending request..."):
        model = selected_models[0]
        response = llm.chat_completion(model.id, prompt, user_input, temperature)
    with st.spinner("Calculating cost and stats..."):
        cost_and_stats = llm.cost_and_stats(response)

    st.markdown(f"### {model.name} response")
    with st.expander("Click to see the raw response"):
        st.write("#### LLM raw response")
        st.json(response.raw_response, expanded=False)
        st.write("#### Cost and stats raw response")
        st.json(cost_and_stats.raw_response, expanded=False)
    st.markdown(response.response)
