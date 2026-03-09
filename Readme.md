# Chatbot with MiniCPM-V-4_5 and OpenVINO

This repository contains a minimal Python chatbot that uses the
`openbmb/MiniCPM-V-4_5` causal language model and runs inference via
[OpenVINO](https://www.openvino.ai/).

The project demonstrates how to export a Hugging Face transformer model to
OpenVINO format using the `optimum` library and then execute a simple
interactive prompt.

---

## Prerequisites

- Python 3.8+ (tested on Windows)
- A working internet connection to download model weights

Install dependencies:

```bash
pip install -r requirements.txt
```

> The first run will export the model from Transformers to OpenVINO.  The
> converted files are cached under `~/.cache/ov_models`.

## Running the chatbot

```bash
# default runs on CPU
python chatbot/app.py

# specify another OpenVINO device
python chatbot/app.py --device GPU
```
Type a message and press Enter; the bot will reply.  Enter `exit` or
`quit` to finish.

## Project layout

```
.
├── Readme.md        # this file
├── requirements.txt # python dependencies
└── chatbot
    └── app.py       # main script
```

## Notes

- This is a minimal example; error handling and conversation context are
  omitted for clarity.
- You can customize generation parameters or add a web front‑end as needed.

