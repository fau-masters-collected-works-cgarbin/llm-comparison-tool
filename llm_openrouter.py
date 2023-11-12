"""Interface to OpenRouter.

The main source for this code is https://openrouter.ai/docs.

The main steps to use OpenRouter are:

1. Create an API key at https://openrouter.ai/keys
2. Add the key to the request: "Authorization": `Bearer ${OPENROUTER_API_KEY}`
3. Understand prompt transformation: https://openrouter.ai/docs#transforms

We use the OpenAI SDK with OpenRouter, as described at https://github.com/alexanderatallah/openrouter-streamlit. It
takes care of some of the steos above.
"""
import os
import time
from dataclasses import dataclass

import dotenv
from openai import OpenAI


@dataclass
class LLMResponse:
    """Class to hold the LLM response.

    We use our class instead of returning the native LLM response to make it easier to adapt to different LLMs later.
    """

    model: str = ""
    prompt: str = ""
    user_input: str = ""
    llm_response: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    raw_response: dict = {}
    elapsed_time: float = 0.0

    @property
    def total_tokens(self):
        """Calculate the total number of tokens used."""
        return self.input_tokens + self.output_tokens


def _get_openai_client() -> OpenAI:
    """Get a client for OpenAI."""
    # Try an OpenAI key, fall back to OpenRouter key if not found
    dotenv.load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise EnvironmentError("API key environment variable not set -- see README.md for instructions")

    # Instantiate a new client but point it to the OpenRouter API
    # OpenRouter is compatible with the OpenAI API
    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        # Headers required by OpenRouter
        # See https://openrouter.ai/docs#quick-start
        default_headers={"HTTP-Referer": "http://localhost:3000"},
    )


def _cost() -> float:
    """Calculate the cost of the completion."""
    return 0


def chat_completion(model: str, prompt: str, user_input: str) -> LLMResponse:
    """Get a chat completion from the specified model."""
    # Always instantiate a new client to pick up configuration changes without restarting the program
    client = _get_openai_client()

    start_time = time.time()
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input},
        ],
        temperature=0.0,  # We want precise and repeatable results
    )
    elapsed_time = time.time() - start_time

    # Record the request and the response
    response = LLMResponse()
    response.elapsed_time = elapsed_time
    response.model = model
    response.prompt = prompt
    response.user_input = user_input
    response.llm_response = completion.choices[0].message.content or ""

    # This is not exactly the raw response, but it's close enough
    # It assumes the completion object is a pydantic.BaseModel class, which has the `dict()`
    # method we need here
    response.raw_response = completion.model_dump()

    # Record the number of tokens used for input and output
    response.input_tokens = 0  # TODO
    response.output_tokens = 0  # TODO

    # Records costs (depends on the tokens and model - set them first)
    response.cost = _cost()

    return response


def _test():
    """Test function - add breakpoints and start under the debugger."""
    response = chat_completion(
        "mistralai/mistral-7b-instruct",
        "You are a helpful assistant and an expert in MATLAB.",
        "What is the latest matlab version?",
    )
    print(response)


if __name__ == "__main__":
    _test()
