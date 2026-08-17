# Shogun AI — Game Auto-Play System

A Python AI that **automatically plays Shogun Showdown**, a turn-based tactical roguelike. It captures
the game screen, parses the game state with **OpenCV computer vision**, makes tactical decisions, and
executes keyboard actions.

## Pipeline

**Screenshot → CV Parse → Decision → Execute → Loop**

## AI Backends

- **Rule Engine (default)** — a 600+ line prioritized decision system. Free, instant, no AI model required.
- **Ollama (local LLM)** — runs a local language model for decisions.
- **Claude API (optional)** — highest decision quality via Anthropic's API.

## Key Features

- Computer-vision state parsing (player HP, enemy positions, skill tiles, attack queue).
- Lethal combo calculator (1/2/3-skill combinations).
- Playbook system that records and replays successful tactics.
- Stuck detection, auto-restart on death, and background mode (no focus stealing).

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Copy the example environment file and fill in your settings:

   ```bash
   cp .env.example .env
   ```

3. Set `AI_BACKEND` to `rule`, `ollama`, or `anthropic` in `.env`.
4. Run:

   ```bash
   python main.py
   ```

> ⚠️ Never commit your `.env` file — it may contain API keys.

## Tech Stack

Python · OpenCV · pyautogui · pydirectinput · mss · numpy · Pillow · httpx · Ollama
