"""
Dead-simple FastAPI endpoint to stream tokens back from generation.
"""

# imports
import argparse
import threading
from pathlib import Path
from typing import AsyncIterable

# packages
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


def api_factory(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
) -> FastAPI:
    """
    Create a FastAPI instance with a streaming endpoint.

    Args:
        model (AutoModelForCausalLM): the model
        tokenizer (PreTrainedTokenizerFast): the tokenizer
        max_tokens (int, optional): the maximum number of tokens to generate. Defaults to 2048.
        temperature (float, optional): the temperature of the generation. Defaults to 0.7.
        top_p (float, optional): the top-p value of the generation. Defaults to 0.9.
        top_k (int, optional): the top-k value of the generation. Defaults to 100.
        repetition_penalty (float, optional): the repetition penalty of the generation. Defaults to 1.1.

    Returns:
        FastAPI: the FastAPI instance
    """
    # create the FastAPI instance
    app = FastAPI()

    # async streaming endpoint
    @app.get("/generate")
    async def generate(
            prompt: str,
            max_new_tokens: int = 2048,
            temperature: float = 0.7,
            top_p: float = 0.9,
            top_k: int = 128,
            repetition_penalty: float = 1.2,
            penalty_alpha: float = 0.25,
            no_repeat_ngram_size: int = 3,
    ):
        """
        Generate text from a prompt.

        Args:
            prompt (str): the prompt
            max_new_tokens (int, optional): the maximum number of tokens to generate. Defaults to 2048.
            temperature (float, optional): the temperature of the generation. Defaults to 0.7.
            top_p (float, optional): the top-p value of the generation. Defaults to 0.9.
            top_k (int, optional): the top-k value of the generation. Defaults to 100.
            repetition_penalty (float, optional): the repetition penalty of the generation. Defaults to 1.2.
            penalty_alpha (float, optional): the penalty alpha of the generation. Defaults to 0.25.
            no_repeat_ngram_size (int, optional): the no repeat ngram size of the generation. Defaults to 3.

        Returns:
            str: the generated text
        """
        # create local method to stream from
        def streamer() -> AsyncIterable[str]:
            # create the streamer
            streamer = TextIteratorStreamer(tokenizer)

            # get tokens
            tokens = tokenizer(prompt, return_tensors="pt")["input_ids"]
            try:
                tokens = tokens.to(model.device)
            except:
                pass

            # setup kwargs for generation
            generation_kwargs = dict(
                input_ids=tokens,
                streamer=streamer,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                penalty_alpha=penalty_alpha,
                no_repeat_ngram_size=no_repeat_ngram_size,
                max_new_tokens=max_new_tokens,
            )

            # start the generation in a separate thread
            generation_thread = threading.Thread(
                target=model.generate, kwargs=generation_kwargs  # type: ignore
            )
            generation_thread.start()

            for new_text in streamer:
                yield new_text

            # wait for the generation to finish
            generation_thread.join()

        # return the streaming response
        return StreamingResponse(streamer())

    return app


if __name__ == "__main__":
    # argparse to get model path
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, help="path to model")
    args = parser.parse_args()

    if not args.model:
        model_path = Path("model/").absolute()
    else:
        model_path = args.model

    print(f"Using model: {model_path}")

    # load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # create the FastAPI instance
    app = api_factory(model, tokenizer)

    # run the app
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
