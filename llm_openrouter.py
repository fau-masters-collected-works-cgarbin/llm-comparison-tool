"""Interface to OpenRouter.

The main source for this code is https://openrouter.ai/docs.

The main steps to use OpenRouter are:

1. Create an API key at https://openrouter.ai/keys
2. Add the key to the request: "Authorization": `Bearer ${OPENROUTER_API_KEY}`
3. Understand prompt transformation: https://openrouter.ai/docs#transforms

Although OpenRouter suports the OpenAI Python client we use an HTTP request to understand the API better and to
have more control over the request.
"""
import os
import time
from dataclasses import dataclass
from dataclasses import field

import dotenv
import requests


_OPENROUTER_API = "https://openrouter.ai/api/v1"
_REFERER = "http://localhost:3000"


@dataclass
class LLMResponse:
    """Class to hold the LLM response.

    We use our class instead of returning the native LLM response to make it easier to adapt to different LLMs later.
    """

    id: str = ""
    model: str = ""
    prompt: str = ""
    user_input: str = ""
    response: str = ""

    raw_response: dict = field(default_factory=dict)
    elapsed_time: float = 0.0


@dataclass
class LLMCostAndStats:
    """Class to hold the LLM cost and stats.

    We use our class instead of returning the native LLM response to make it easier to adapt to different LLMs later.
    """

    id: str = ""  # Should match the request ID - we store to correlate the response and the cost/stats

    # It's not clear what the differences between tokens_ and natives_tokens_ are
    # According to this post https://twitter.com/OpenRouterAI/status/1704862401022009773, it seems that tokens_ is
    # calculated with the GPT tokenizer and native_tokens_ is calculated with the native tokenizer for the model
    # We will make that assumption and name the fields accordingly
    gpt_tokens_prompt: int = 0
    gpt_tokens_completion: int = 0
    native_tokens_prompt: int = 0
    native_tokens_completion: int = 0
    cost: float = 0.0

    raw_response: dict = field(default_factory=dict)
    elapsed_time: float = 0.0

    @property
    def gpt_tokens_total(self):
        return self.gpt_tokens_prompt + self.gpt_tokens_completion

    @property
    def native_tokens_total(self):
        return self.native_tokens_prompt + self.native_tokens_completion


def _get_api_key() -> str:
    """Get the API key from the environment."""
    # Try an OpenAI key, fall back to OpenRouter key if not found
    dotenv.load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise EnvironmentError("API key environment variable not set -- see README.md for instructions")

    return api_key


def cost_and_stats(response: LLMResponse) -> LLMCostAndStats:
    """Retrieve costs and stats for the LLM response."""

    # Always get the API key from the environment in case it changed
    headers = {
        "Authorization": f"Bearer {_get_api_key()}",
    }

    # Implementation notes:
    #   - We let `requests` manage the connection pool for us.
    #   - The default configuration allows up to 10 simultaneous connections to the same host, which should be enough
    #     for our needs.
    #   - We use a large timeout because some LLMs take a long time to respond.
    start_time = time.time()
    http_response = requests.get(f"{_OPENROUTER_API}/generation?id={response.id}", headers=headers, timeout=120)
    response_time = time.time() - start_time
    # We let exceptions propagate for now because this is a developement tool
    # When/if we let end users (or less technical users) use this code, we handle exceptions more gracefully
    # Note that the exception will stop the parallel execution of the LLMs, which is ok in our case
    http_response.raise_for_status()

    payload = http_response.json()["data"]

    cost_stats = LLMCostAndStats()
    cost_stats.elapsed_time = response_time
    cost_stats.id = response.id
    cost_stats.gpt_tokens_prompt = payload["tokens_prompt"]
    cost_stats.gpt_tokens_completion = payload["tokens_completion"]
    cost_stats.native_tokens_prompt = payload["native_tokens_prompt"]
    cost_stats.native_tokens_completion = payload["native_tokens_completion"]
    cost_stats.cost = payload["usage"]
    cost_stats.raw_response = payload

    return cost_stats


def chat_completion(model: str, prompt: str, user_input: str) -> LLMResponse:
    """Get a chat completion from the specified model."""
    # TODO: add temperature
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input},
        ],
    }

    headers = {
        # Always get the API key from the environment in case it changed
        "Authorization": f"Bearer {_get_api_key()}",
        "HTTP-Referer": _REFERER,
    }

    url = f"{_OPENROUTER_API}/chat/completions"

    start_time = time.time()
    http_response = requests.post(url=url, json=payload, headers=headers, timeout=120)
    elapsed_time = time.time() - start_time
    # We let exceptions propagate for now because this is a developement tool
    # When/if we let end users (or less technical users) use this code, we handle exceptions more gracefully
    # Note that the exception will stop the parallel execution of the LLMs, which is ok in our case
    http_response.raise_for_status()

    payload = http_response.json()

    # Record the request and the response
    # We use the keys without checking if they exist because we want to know if the API changes (it will result
    # in a hard failure that makes the change obvious)
    response = LLMResponse()
    response.elapsed_time = elapsed_time
    response.id = payload["id"]
    response.model = model
    response.prompt = prompt
    response.user_input = user_input
    response.response = payload["choices"][0]["message"]["content"]

    response.raw_response = payload

    return response


def _test():
    """Test function - add breakpoints and start under the debugger."""
    response = chat_completion(
        "mistralai/mistral-7b-instruct",
        "You are a helpful assistant and an expert in MATLAB.",
        "What is the latest matlab version?",
    )
    print(f"Response:\n{response}")

    cost_stats = cost_and_stats(response)
    print(f"\nCost and stats:\n{cost_stats}")


if __name__ == "__main__":
    _test()
