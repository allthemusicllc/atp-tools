"""
Streamlit streaming chat demo
"""

# imports
import argparse
import asyncio
import sys
import threading
from pathlib import Path
from typing import Iterable

# packages
import streamlit as st
import httpx

# setup the model artifact details
BASE_MODEL_URI = "https://atp.273v.io/models/"
MODEL_VERSION = "kl3m-170m-patent-v001.tar.gz"
MODEL_PATH = Path(__file__).parent / "model"
OPENAI_BASE_URL = "http://localhost:8000/"

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
    penalty_alpha: float = 0.25,
    no_repeat_ngram_size: int = 3,
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
        penalty_alpha (float, optional): the penalty alpha of the generation. Defaults to 0.25.
        no_repeat_ngram_size (int, optional): the no repeat ngram size of the generation. Defaults to 3.

    Yields:
        str: the generated text
    """
    # get an httpx client for base path
    with httpx.Client(base_url=OPENAI_BASE_URL) as client:
        # setup kwargs to pass to the /generate endpoint
        generate_kwargs = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "penalty_alpha": penalty_alpha,
            "no_repeat_ngram_size": no_repeat_ngram_size,
        }

        # make the GET request to /generate and then stream the tokens back
        with client.stream("GET", "/generate", params=generate_kwargs) as response:
            for chunk in response.iter_text():
                yield chunk


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
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
        max_new_tokens = st.number_input("max_new_tokens", 1, 4096, 1024)
        top_k = st.number_input("top_k", 1, 256, 128)
        top_p = st.number_input("top_p", 0.0, 1.0, 0.95)
        penalty_alpha = st.number_input("Penalty Alpha", 0.0, 2.0, 0.5)
        repetition_penalty = st.number_input("Repetition Penalty", 1.0, 2.0, 1.2)
        nop_repeat_ngram_size = st.number_input("no_repeat_ngram_size", 1, 100, 10)

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
                    penalty_alpha=penalty_alpha,
                    max_new_tokens=max_new_tokens,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=nop_repeat_ngram_size,
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
