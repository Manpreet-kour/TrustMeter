# HILT Conversation Recorder

A Streamlit application for recording, editing, and managing multi-turn human↔AI conversations with trust ratings and conversation management features.

## Features

- Multi-turn Conversation Recording
- AI Reply Editing and Accept/Modify workflow
- Trust Rating (1–5) per turn
- Persistent NDJSON storage at `./data/conversations.ndjson`
- Admin interface to browse, search, and export conversations
- Placeholder LLM with commented OpenAI example

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python -m streamlit run streamlit_hilt_app.py
```

## Deploy to Streamlit Community Cloud

1. Push this folder to a GitHub repo (root should contain `streamlit_hilt_app.py` and `requirements.txt`).
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/).
3. "New app" → select your repo/branch → set the main file path to `streamlit_hilt_app.py` → Deploy.

Note: The app writes data to `./data/conversations.ndjson`. On Streamlit Cloud, writes are ephemeral (reset on redeploy).


