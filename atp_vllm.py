"""
Streamlit streaming chat demo
"""

# imports
import argparse
import asyncio
from pathlib import Path
from typing import Iterable

# packages
import streamlit as st
import openai


# setup the model artifact details
BASE_MODEL_URI = "https://atp.273v.io/models/"
MODEL_VERSION = "kl3m-170m-patent-v001.tar.gz"
MODEL_PATH = Path(__file__).parent / "model"
OPENAI_API_KEY = "EMPTY"
OPENAI_BASE_URL = "http://localhost:8000/v1"
MODEL_ID = "model/"

# keep description handy up top
HEADER_DESCRIPTION = """# All the Patents

##### Version: 2024-03-12, proof of concept v001

 * **Project**: [All the Patents on GitHub](https://github.com/allthemusicllc/atp-tools)
 * **Model**: kl3m-1.7b-patents-v002
 * **Base Model**: 🍊 [kl3m-1.7b](https://273ventures.com/kl3m-the-first-legal-large-language-model/) (Kelvin Legal LLM)  
 * **Fine-Tune Data**: USPTO Full Text, 1971-1978, 2023
 * **Team**: M. Bommarito, D. Riehl & N. Rubin (All the Music LLC), and friends
"""


def stream_generation(
    prompt: str,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 10,
    repetition_penalty: float = 1.25,
    show_prompt: bool = False,
) -> Iterable[str]:
    """
    Stream the generation of a prompt.

    Args:
        prompt (str): the prompt
        max_new_tokens (int, optional): the maximum number of tokens to generate. Defaults to 32.
        temperature (float, optional): the temperature of the generation. Defaults to 0.7.
        top_p (float, optional): the top-p value of the generation. Defaults to 0.9.
        top_k (int, optional): the top-k value of the generation. Defaults to 100.
        repetition_penalty (float, optional): the repetition penalty of the generation. Defaults to 1.1.
        show_prompt (bool, optional): whether to show the prompt or not. Defaults to False.

    Yields:
        str: the generated text
    """
    # get openai client
    client = openai.OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

    # create completion with streaming
    completions = client.completions.create(
        model=MODEL_ID,
        prompt=prompt,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        extra_body={
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
        },
        echo=False,
        stream=True
    )

    # check for show prompt
    if show_prompt:
        yield prompt

    # yield the prompt
    for token in completions:
        yield token.choices[0].text


async def amain():
    """
    Async main for the functional streamlit setup.  Do not use any class constructs.

    Args:
        tokenizer (PreTrainedTokenizerFast): the tokenizer
        model (AutoModelForCausalLM): the model
    """
    # setup streamlit with wide mode
    st.set_page_config()
    st.markdown(HEADER_DESCRIPTION)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # container with left-right columns for each button
    top_columns = st.columns(2)

    # add the other generation params next
    with top_columns[0].expander("Generation Parameters"):
        # title-first or claims-first dropdown
        generation_type = st.selectbox(
            "Generation Type", ["Top Down", "Bottom Up"], index=0
        )

        # add temperature slider to pass to gen in third column
        temperature = st.slider("Temperature", 0.0, 1.0, 0.5)
        max_new_tokens = st.number_input("max_new_tokens", 1, 4096, 1024)
        top_k = st.number_input("top_k", 1, 256, 128)
        top_p = st.number_input("top_p", 0.0, 1.0, 0.9)
        repetition_penalty = st.number_input("Repetition Penalty", 0.0, 2.0, 1.5)

    # button to generate directly on left
    if top_columns[1].button("Generate a Patent", type="primary"):
        with st.chat_message("assistant"):
            # prompt type from generation
            if generation_type == "Top Down":
                prompt = "# Patent\n\n## Title\n"
            elif generation_type == "Bottom Up":
                prompt = "# Patent\n\n## Claims\n"

            st.write_stream(
                stream_generation(
                    prompt,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    show_prompt=True,
                )
            )


if __name__ == "__main__":
    # setup args
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="Download the model")
    parser.add_argument(
        "--model-version",
        type=str,
        help="The model version to download",
        default=MODEL_VERSION,
    )
    parser.add_argument(
        "--model-path", type=Path, help="Path to the model", default=MODEL_PATH
    )
    args = parser.parse_args()

    # run the main
    asyncio.run(
        amain()
    )
