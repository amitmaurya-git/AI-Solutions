# OpenRouter ReAct Agent

A **ReAct (Reasoning + Acting)** agent built with Python, the OpenAI SDK, and
[OpenRouter](https://openrouter.ai).

---

## What Is the ReAct Pattern?

ReAct combines **chain-of-thought reasoning** with **tool use** in an
iterative loop. Instead of answering in one shot, the model:

1. **Thinks** about what information it needs.
2. **Acts** by calling one or more tools (weather lookup, calculator, …).
3. **Observes** the tool results that Python returns.
4. **Repeats** steps 1–3 until it has enough information.
5. **Answers** the user with a final, grounded response.

This keeps the model honest — it never invents data and always shows its work
through real tool executions.

---

## Agent Execution Flow

```
User question
  → model decides whether a tool is required
  → model requests one or more tools
  → Python executes the requested tools
  → tool results are returned to the model
  → model continues reasoning through tool calls
  → final answer is returned to the user
```

---

## Project Structure

```
openrouter-react-agent/
├── .env.example        # Template for environment variables
├── .gitignore          # Files excluded from version control
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── react_agent.py      # The ReAct agent implementation
```

---

## Prerequisites

- **Python 3.12** (required)
- An **OpenRouter API key** — get one at <https://openrouter.ai/keys>

---

## Setup (Windows PowerShell)

### 1. Create and activate a virtual environment

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Upgrade core packaging tools and install dependencies

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

### 3. Verify the OpenAI SDK import

```powershell
python -c "from openai import OpenAI; print('OpenAI import successful')"
```

### 4. Configure environment variables

Copy the example file and add your real API key:

```powershell
Copy-Item .env.example .env
```

Then edit `.env`:

```
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=openrouter/auto
```

> **Tip:** You can change `OPENROUTER_MODEL` to any model available on
> OpenRouter that supports function calling (e.g. `google/gemini-2.5-flash`).

### 5. Select the `.venv` interpreter in VS Code

1. Open the Command Palette — `Ctrl + Shift + P`.
2. Type **Python: Select Interpreter**.
3. Choose the interpreter inside `.venv\Scripts\python.exe`.

---

## Running the Agent

```powershell
python react_agent.py
```

You will see an interactive prompt:

```
OpenRouter ReAct Agent
Type 'exit' to stop.

You:
```

Type a question and press Enter. Type `exit` or `quit` to stop.

---

## Example Questions

Try these to see the ReAct loop in action:

| Question | Tools Used |
|---|---|
| `What is the weather in Delhi?` | `get_weather` |
| `What is the weather in Mumbai and convert the temperature to Fahrenheit?` | `get_weather` → `convert_celsius_to_fahrenheit` |
| `What is the weather in Delhi, convert it to Fahrenheit, and multiply the Fahrenheit value by 2?` | `get_weather` → `convert_celsius_to_fahrenheit` → `calculate` |
| `Calculate (25 * 4) + 10.` | `calculate` |

---

## How the Tool-Call Loop Works

1. The user's question is added to the **message history**.
2. The history (including a system prompt) is sent to the model along with
   the available **tool schemas**.
3. If the model's response contains **tool_calls**, each call is executed
   locally in Python and the result is appended as a `role="tool"` message.
4. The updated history is sent back to the model so it can continue
   reasoning or issue more tool calls.
5. Steps 2–4 repeat until the model returns a **normal text answer** or the
   **maximum iteration count** is reached.

---

## Linting with Ruff

```powershell
ruff check react_agent.py --fix
ruff format react_agent.py
```

---

## License

This project is provided for educational purposes.
