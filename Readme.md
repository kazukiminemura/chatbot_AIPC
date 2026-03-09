# Chatbot with OpenVINO GenAI 2026

This repository contains a minimal Python chatbot that uses
`openvino-genai==2026.0.0` for inference.

On first run, the app can export a Hugging Face visual language model to
OpenVINO format automatically, then run it through
`openvino_genai.VLMPipeline`.

## Prerequisites

- Python 3.10+
- A working internet connection for the first run

Install dependencies:

```bash
pip install -r requirements.txt
```

`MiniCPM-V-4_5` uses remote model code that depends on Pillow and torchvision.

## Running the chatbot

```bash
python chatbot/app.py
python chatbot/app.py --model-id openbmb/MiniCPM-V-4_5 --device GPU
python chatbot/app.py --model-id openbmb/MiniCPM-V-4_5 --model-dir path/to/openvino_model
```

Type a message and press Enter. Enter `exit` or `quit` to finish.

## Notes

- The first run exports the model into `models/<model-id>/`.
- The default model is `openbmb/MiniCPM-V-4_5`.
- This app currently uses the model in text-chat mode, even though the
  backend pipeline supports image inputs.
