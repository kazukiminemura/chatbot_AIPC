#!/usr/bin/env python3
"""Simple chatbot using OpenVINO GenAI.

This script runs text generation with `openvino_genai.VLMPipeline`.
If the target OpenVINO model directory does not exist yet, it is exported
automatically from Hugging Face on first run.

Usage:
    python chatbot/app.py
    python chatbot/app.py --model-id some/model --device GPU
"""
from pathlib import Path
import importlib
import os
import sys
import sysconfig

def _configure_windows_dll_search_paths() -> None:
    """Ensure OpenVINO native libraries are discoverable on Windows."""
    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return

    purelib = Path(sysconfig.get_paths()["purelib"])
    candidate_dirs = [
        purelib / "openvino" / "libs",
        purelib / "openvino_genai",
    ]
    for dll_dir in candidate_dirs:
        if dll_dir.is_dir():
            os.add_dll_directory(str(dll_dir))


_configure_windows_dll_search_paths()


def _populate_openvino_namespace() -> None:
    """Backfill top-level OpenVINO symbols for namespace-package installs."""
    ov_module = importlib.import_module("openvino")
    pyopenvino = importlib.import_module("openvino._pyopenvino")

    for name in dir(pyopenvino):
        if name.startswith("_") or hasattr(ov_module, name):
            continue
        setattr(ov_module, name, getattr(pyopenvino, name))

    for module_name, attr_name in [
        ("openvino.utils", "utils"),
        ("openvino.frontend", "frontend"),
        ("openvino.frontend.frontend", None),
    ]:
        module = importlib.import_module(module_name)
        if attr_name and not hasattr(ov_module, attr_name):
            setattr(ov_module, attr_name, module)

    frontend_pkg = importlib.import_module("openvino.frontend")
    frontend_module = importlib.import_module("openvino.frontend.frontend")
    if not hasattr(frontend_pkg, "frontend"):
        setattr(frontend_pkg, "frontend", frontend_module)


_populate_openvino_namespace()

import openvino as ov
import openvino_genai as ov_genai


MODEL_NAME = "openbmb/MiniCPM-V-4_5"


def default_model_dir(model_id: str) -> Path:
    return Path("models") / model_id.replace("/", "--")


def export_model(model_id: str, model_dir: Path) -> None:
    """Export a Hugging Face VLM into an OpenVINO GenAI-ready directory."""
    try:
        from openvino_tokenizers import convert_tokenizer
        from transformers import AutoProcessor
        from optimum.intel import OVModelForVisualCausalLM
    except ImportError as exc:
        raise RuntimeError(
            "Automatic export requires optimum-intel and OpenVINO tokenizer support."
        ) from exc

    model_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = OVModelForVisualCausalLM.from_pretrained(
        model_id,
        export=True,
        trust_remote_code=True,
    )
    model.save_pretrained(model_dir)
    processor.save_pretrained(model_dir)

    ov_tokenizer, ov_detokenizer = convert_tokenizer(processor.tokenizer, with_detokenizer=True)
    ov.save_model(ov_tokenizer, model_dir / "openvino_tokenizer.xml")
    ov.save_model(ov_detokenizer, model_dir / "openvino_detokenizer.xml")


def ensure_model_exported(model_id: str, model_dir: Path) -> Path:
    """Export the model on first run if the target directory is missing artifacts."""
    legacy_pairs = [
        (model_dir / "tokenizer.xml", model_dir / "openvino_tokenizer.xml"),
        (model_dir / "tokenizer.bin", model_dir / "openvino_tokenizer.bin"),
        (model_dir / "detokenizer.xml", model_dir / "openvino_detokenizer.xml"),
        (model_dir / "detokenizer.bin", model_dir / "openvino_detokenizer.bin"),
    ]
    for legacy_path, current_path in legacy_pairs:
        if legacy_path.exists() and not current_path.exists():
            legacy_path.replace(current_path)

    llm_required_files = [
        model_dir / "openvino_model.xml",
        model_dir / "openvino_model.bin",
        model_dir / "openvino_tokenizer.xml",
        model_dir / "openvino_detokenizer.xml",
    ]
    vlm_required_files = [
        model_dir / "openvino_language_model.xml",
        model_dir / "openvino_language_model.bin",
        model_dir / "openvino_text_embeddings_model.xml",
        model_dir / "openvino_text_embeddings_model.bin",
        model_dir / "openvino_vision_embeddings_model.xml",
        model_dir / "openvino_vision_embeddings_model.bin",
        model_dir / "openvino_resampler_model.xml",
        model_dir / "openvino_resampler_model.bin",
        model_dir / "openvino_tokenizer.xml",
        model_dir / "openvino_detokenizer.xml",
    ]
    if all(path.exists() for path in llm_required_files) or all(path.exists() for path in vlm_required_files):
        return model_dir

    print(f"Exporting {model_id} to OpenVINO format at {model_dir} ...")
    export_model(model_id, model_dir)
    return model_dir


def load_pipeline(model_dir: Path, device: str = "CPU") -> ov_genai.VLMPipeline:
    """Load an OpenVINO GenAI VLM pipeline from an exported model directory."""
    if not model_dir.exists():
        raise FileNotFoundError(f"OpenVINO model directory not found: {model_dir}")

    return ov_genai.VLMPipeline(str(model_dir), device)


def build_generation_config() -> ov_genai.GenerationConfig:
    """Return the default generation settings for interactive chat."""
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = 128
    config.do_sample = True
    config.top_p = 0.95
    config.temperature = 0.8
    return config


def chat_loop(pipeline: ov_genai.VLMPipeline) -> None:
    """Interactive text-only conversation loop for a visual language model."""
    generation_config = build_generation_config()
    pipeline.start_chat()

    print("Type 'exit' or 'quit' to end.")
    try:
        while True:
            prompt = input("You: ")
            if prompt.strip().lower() in {"exit", "quit"}:
                break
            if not prompt.strip():
                continue

            result = pipeline.generate(prompt, generation_config=generation_config)
            texts = result.texts if hasattr(result, "texts") else []
            text = texts[0] if texts else str(result)
            print(f"Bot: {text}\n")
    finally:
        pipeline.finish_chat()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Chat with MiniCPM-V-4_5 via OpenVINO GenAI")
    parser.add_argument(
        "--model-id",
        type=str,
        default=MODEL_NAME,
        help="Hugging Face model ID to export on first run",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Path to store or load the exported OpenVINO model directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="CPU",
        help="OpenVINO device to use (CPU, GPU, NPU, etc.)",
    )
    args = parser.parse_args()

    model_dir = args.model_dir or default_model_dir(args.model_id)
    ensure_model_exported(args.model_id, model_dir)
    pipeline = load_pipeline(model_dir, device=args.device)
    chat_loop(pipeline)


if __name__ == "__main__":
    main()
