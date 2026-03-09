#!/usr/bin/env python3
"""Simple chatbot using openbmb/MiniCPM-V-4_5 and OpenVINO.

This script loads the MiniCPM-V-4_5 causal language model from Hugging
Face and exports it to OpenVINO format (if necessary) using Optimum. Once
converted, the OpenVINO runtime is used for inference in a simple
interactive loop.

Usage:
    python chatbot/app.py

Requirements:
    - transformers
    - optimum
    - openvino-dev
    - numpy

"""
import os
import sys
from typing import Optional

import numpy as np
from transformers import AutoTokenizer

try:
    from optimum.openvino import OVModelForCausalLM, OVTokenizer
except ImportError:
    raise RuntimeError(
        "optimum-openvino is required. Install via `pip install optimum[openvino]` or `pip install openvino-dev`"
    )

MODEL_NAME = "openbmb/MiniCPM-V-4_5"  # Hugging Face model ID


def load_or_export_model(model_name: str = MODEL_NAME, device: str = "CPU") -> OVModelForCausalLM:
    """Load an OpenVINO model. If conversion is required, it will run
    automatically and cache the results.

    Args:
        model_name: Hugging Face model identifier.
        device: OpenVINO target device ("CPU", "GPU", "MYRIAD", etc.).
    """
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "ov_models")
    os.makedirs(cache_dir, exist_ok=True)

    model = OVModelForCausalLM.from_pretrained(
        model_name,
        from_transformers=True,
        export=True,
        cache_dir=cache_dir,
    )
    model.to(device)
    return model


def chat_loop(model: OVModelForCausalLM, tokenizer: OVTokenizer) -> None:
    """Interactive conversation loop."""
    print("Type 'exit' or 'quit' to end.")
    while True:
        prompt = input("You: ")
        if prompt.strip().lower() in {"exit", "quit"}:
            break
        # encode prompt
        inputs = tokenizer(prompt, return_tensors="np")
        out_tokens = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            top_p=0.95,
            temperature=0.8,
        )
        text = tokenizer.decode(out_tokens[0], skip_special_tokens=True)
        print(f"Bot: {text}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="MiniCPM chat with OpenVINO")
    parser.add_argument(
        "--device",
        type=str,
        default="CPU",
        help="OpenVINO device to use (CPU, GPU, MYRIAD, etc.)",
    )
    args = parser.parse_args()

    tokenizer = OVTokenizer.from_pretrained(MODEL_NAME)
    model = load_or_export_model(MODEL_NAME, device=args.device)
    chat_loop(model, tokenizer)


if __name__ == "__main__":
    main()
