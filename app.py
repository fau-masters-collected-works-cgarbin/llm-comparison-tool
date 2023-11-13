import streamlit as st
import llm_openrouter as llm

st.set_page_config(page_title="Multi LLM Test Tool", layout="wide")
st.title("Multi LLM Test Tool")


@st.cache_data
def get_models():
    models = llm.available_models()
    models = sorted(models, key=lambda x: x.name)
    return models


def prepare_session_state():
    if "key" not in st.session_state:
        st.session_state.key = ""
    if "prompt" not in st.session_state:
        st.session_state.prompt = ""
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.0
    if "models" not in st.session_state:
        st.session_state.models = []


def configuration():
    with st.expander("Click to configure prompt and models"):
        models = get_models()
        st.session_state.prompt = st.text_area("Prompt", height=200)
        st.session_state.temperature = st.slider("Select temperature", 0.0, 2.0, 0.0)
        st.session_state.models = st.multiselect("Choose LLM(s)", models)


def get_llm_response(user_input: str) -> dict[llm.Model, llm.LLMResponse]:
    with st.spinner("Sending request..."):
        models = st.session_state.models
        if not isinstance(models, list):
            models = [models]
        response = llm.chat_completion_multiple(
            models, st.session_state.prompt, user_input, st.session_state.temperature
        )
    return response


def get_cost_and_stats(response: dict[llm.Model, llm.LLMResponse]) -> dict[llm.Model, llm.LLMCostAndStats]:
    with st.spinner("Calculating cost and stats..."):
        cost_and_stats = llm.cost_and_stats_multiple(response)
    return cost_and_stats


def show_response(response: dict[llm.Model, llm.LLMResponse], cost_and_stats: dict[llm.Model, llm.LLMCostAndStats]):
    for m, r in response.items():
        st.markdown(f"### {m.name} response")
        with st.expander("Click to see the raw response"):
            st.write("LLM raw response")
            st.json(r.raw_response, expanded=False)
            st.write("Cost and stats raw response")
            st.json(cost_and_stats[m].raw_response, expanded=False)
        st.markdown(r.response)


prepare_session_state()
configuration()
st.write(f"Selected models: {', '.join([model.name for model in st.session_state.models])}")

user_input = st.text_area("Enter your request to the LLM", height=150)
send_button = st.button("Send Request")

if send_button:
    response = get_llm_response(user_input)
    cost_and_stats = get_cost_and_stats(response)
    show_response(response, cost_and_stats)
