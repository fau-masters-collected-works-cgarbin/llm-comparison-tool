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
    session_vars = { # Default values
        "prompt": "",
        "temperature": 0.0,
        "max_tokens": 2048,
        "models": [],
    }
    for var in session_vars:
        if var not in st.session_state:
            st.session_state[var] = session_vars[var]


def configuration():
    with st.expander("Click to to show/hide configuration options (system prompt, temperature, models)"):
        models = get_models()
        st.session_state.prompt = st.text_area("System prompt", placeholder="Enter here the system prompt", height=150)
        cols = st.columns([5, 2, 1])
        with cols[0]:
            st.session_state.models = st.multiselect("Model(s)", models, placeholder="Select one or more models")
        with cols[1]:
            st.session_state.temperature = st.slider("Temperature", 0.0, 2.0, 0.0, )
        with cols[2]:
            st.session_state.max_tokens = st.number_input("Max completion tokens", 1, 20_480, 2048, step=10)

        # Order models by name to make them easier to find in the results
        st.session_state.models = sorted(st.session_state.models, key=lambda x: x.name)

        # Show all modes in a markdown table
        st.markdown("**Models details**")
        model_list = (
            "| Model | ID | Prompt price | Completion price |"
            "Context length | Max completion tokens | Tokenizer | Instruct type\n"
            "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |\n"
        )
        for model in models:
            model_list += (
                f"| {model.name} | {model.id} | {model.pricing_prompt:.10f} | {model.pricing_completion:.10f} |"
                f"{model.context_length:,} | {model.max_completion_tokens:,} | {model.tokenizer} |"
                f"{model.instruct_type} |\n"
            )
        st.markdown(model_list)
    selected_models = (
        ", ".join([model.name for model in st.session_state.models])
        if st.session_state.models
        else "_Click above to select models_"
    )
    st.write(f"Selected models: {selected_models}")


def get_llm_response(user_input: str) -> dict[llm.Model, llm.LLMResponse]:
    with st.spinner("Sending request..."):
        models = st.session_state.models
        if not isinstance(models, list):
            models = [models]
        response = llm.chat_completion_multiple(
            models, st.session_state.prompt, user_input, st.session_state.temperature, st.session_state.max_tokens
        )
    return response


def get_cost_and_stats(response: dict[llm.Model, llm.LLMResponse]) -> dict[llm.Model, llm.LLMCostAndStats]:
    with st.spinner("Calculating cost and stats..."):
        cost_and_stats = llm.cost_and_stats_multiple(response)
    return cost_and_stats


def show_response(response: dict[llm.Model, llm.LLMResponse], cost_and_stats: dict[llm.Model, llm.LLMCostAndStats]):
    # Sort the response by model name to keep the order consistent
    response = dict(sorted(response.items(), key=lambda x: x[0].name))

    # Show the response side by side by model
    cols = st.columns(len(response))
    for i, (m, r) in enumerate(response.items()):
        with cols[i]:
            st.markdown(f"### {m.name}")
            with st.expander("Click to show/hide the raw response"):
                st.write("LLM raw request data")
                st.json(r.raw_request, expanded=False)
                st.write("LLM raw response data")
                st.json(r.raw_response, expanded=False)
                st.write("Cost and stats raw response")
                st.json(cost_and_stats[m].raw_response, expanded=False)
            c = cost_and_stats[m]
            st.markdown(
                (
                    "GPT tokens | Native tokens | Cost | Elapsed time |\n"
                    "| --- | --- | --- | --- |\n"
                    "| _(prompt/completion)_ | _(prompt/completion)_ | _(US $)_ | _(seconds)_ |\n"
                    f"| {c.gpt_tokens_prompt}/{c.gpt_tokens_completion} |"
                    f"{c.native_tokens_prompt}/{c.native_tokens_completion} |"
                    f"{c.cost:.10f} | {r.elapsed_time:.1f}s |"
                )
            )
            st.info(r.response)


prepare_session_state()
configuration()

user_input = st.text_area("Enter your request", placeholder="Enter here the user request", height=100)
st.error(":no_entry_sign: Do not enter private or sensitive information. What you type here is going to external servers.")

read_and_agreed = st.checkbox("There is no private or sensitive information in my request")
send_button = st.button("Send Request")

if send_button:
    if not read_and_agreed:
        st.error("Please confirm that there is no private or sensitive information in your request")
        st.stop()
    if not st.session_state.models:
        st.error("Please select at least one model")
        st.stop()
    if not user_input:
        st.error("Please enter a request")
        st.stop()

    response = get_llm_response(user_input)
    cost_and_stats = get_cost_and_stats(response)
    show_response(response, cost_and_stats)
