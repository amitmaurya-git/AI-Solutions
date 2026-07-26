"""OpenRouter ReAct Agent — Reasoning + Acting with tool use.

Uses the OpenAI-compatible Python SDK to talk to OpenRouter and implements a
full ReAct loop: the model reasons, requests tools, Python executes them, and
the results are fed back until a final answer emerges.
"""

# ---------------------------------------------------------------------------
# Imports (stdlib → third-party, alphabetical within each group)
# ---------------------------------------------------------------------------
import ast
import json
import operator
import os
import sys
from collections.abc import Callable
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

# ---------------------------------------------------------------------------
# Fix Windows terminal encoding (charmap/cp1252 cannot handle all Unicode)
# ---------------------------------------------------------------------------
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

OPENROUTER_API_KEY: str | None = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "openrouter/auto")

if not OPENROUTER_API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY is not set. "
        "Create a .env file with your key (see .env.example)."
    )

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

# ---------------------------------------------------------------------------
# System prompt — instructs the model on ReAct behaviour
# ---------------------------------------------------------------------------
SYSTEM_PROMPT: str = (
    "You are a helpful assistant with access to tools.\n"
    "When external information or exact calculations are required, use the "
    "available tools — never invent or guess tool results.\n"
    "After receiving tool results, answer the user clearly and concisely.\n"
    "Do not expose or repeat any private chain-of-thought reasoning to the user."
)

# ===========================================================================
# Tool implementations
# ===========================================================================

# ---- A. get_weather -------------------------------------------------------

# Static demo weather data for supported cities.
_WEATHER_DATA: dict[str, dict[str, Any]] = {
    "delhi": {
        "city": "Delhi",
        "temperature_celsius": 42,
        "condition": "Hot and sunny",
    },
    "mumbai": {
        "city": "Mumbai",
        "temperature_celsius": 34,
        "condition": "Humid and partly cloudy",
    },
    "bengaluru": {
        "city": "Bengaluru",
        "temperature_celsius": 27,
        "condition": "Pleasant with light rain",
    },
}


def get_weather(city: str) -> dict[str, Any]:
    """Return static demo weather data for a supported city."""
    key = city.strip().lower()
    data = _WEATHER_DATA.get(key)
    if data is None:
        return {
            "success": False,
            "error": f"Weather data is not available for '{city}'. "
            f"Supported cities: Delhi, Mumbai, Bengaluru.",
        }
    return {"success": True, **data}


# ---- B. convert_celsius_to_fahrenheit -------------------------------------


def convert_celsius_to_fahrenheit(celsius: float) -> dict[str, Any]:
    """Convert a Celsius temperature to Fahrenheit."""
    fahrenheit = (celsius * 9 / 5) + 32
    return {
        "celsius": celsius,
        "fahrenheit": round(fahrenheit, 2),
    }


# ---- C. calculate (safe arithmetic using ast) -----------------------------

# Allowed binary and unary operators for the safe calculator.
_ALLOWED_BINOPS: dict[type, Callable[..., Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_ALLOWED_UNARYOPS: dict[type, Callable[..., Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float:
    """Recursively evaluate an AST node using only allowed operators."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)

    if isinstance(node, ast.BinOp):
        op_func = _ALLOWED_BINOPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        if isinstance(node.op, ast.Div) and right == 0:
            raise ZeroDivisionError("Division by zero")
        return float(op_func(left, right))

    if isinstance(node, ast.UnaryOp):
        op_func = _ALLOWED_UNARYOPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return float(op_func(_safe_eval(node.operand)))

    raise ValueError(
        f"Unsupported expression node: {type(node).__name__}. "
        "Only arithmetic operations are allowed."
    )


def calculate(expression: str) -> dict[str, Any]:
    """Safely evaluate an arithmetic expression without using eval()."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return {"success": False, "error": f"Invalid expression syntax: '{expression}'"}

    try:
        result = _safe_eval(tree)
    except ZeroDivisionError:
        return {"success": False, "error": "Division by zero."}
    except ValueError as exc:
        return {"success": False, "error": str(exc)}

    return {"success": True, "expression": expression, "result": result}


# ===========================================================================
# Tool schemas (OpenAI-compatible function-calling format)
# ===========================================================================

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Get current weather information for a supported Indian city. "
                "Supported cities: Delhi, Mumbai, Bengaluru."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name (e.g. 'Delhi', 'Mumbai', 'Bengaluru').",
                    },
                },
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "convert_celsius_to_fahrenheit",
            "description": "Convert a temperature from Celsius to Fahrenheit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "celsius": {
                        "type": "number",
                        "description": "Temperature in Celsius to convert.",
                    },
                },
                "required": ["celsius"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Evaluate a safe arithmetic expression. Supports +, -, *, /, "
                "%, ** (power), and unary +/-. Does not support variables or "
                "function calls."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The arithmetic expression to evaluate (e.g. '(25 * 4) + 10').",
                    },
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
        },
    },
]

# ===========================================================================
# Tool registry — maps tool names to Python callables
# ===========================================================================

TOOL_REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {
    "get_weather": get_weather,
    "convert_celsius_to_fahrenheit": convert_celsius_to_fahrenheit,
    "calculate": calculate,
}

# ===========================================================================
# Tool execution
# ===========================================================================


def execute_tool(tool_name: str, tool_arguments: dict[str, Any]) -> dict[str, Any]:
    """Look up *tool_name* in the registry and call it with *tool_arguments*.

    Returns a structured error dict for unknown tools, bad arguments, or
    expected runtime errors — never raises to the caller.
    """
    # Reject unknown tools.
    func = TOOL_REGISTRY.get(tool_name)
    if func is None:
        return {"success": False, "error": f"Unknown tool: '{tool_name}'"}

    # Validate that arguments is a dict.
    if not isinstance(tool_arguments, dict):
        return {
            "success": False,
            "error": f"Tool arguments must be a dict, got {type(tool_arguments).__name__}.",
        }

    # Execute and catch only specific, expected exceptions.
    try:
        return func(**tool_arguments)
    except TypeError as exc:
        return {
            "success": False,
            "error": f"Invalid arguments for '{tool_name}': {exc}",
        }
    except (ValueError, ZeroDivisionError, KeyError) as exc:
        return {"success": False, "error": f"Tool '{tool_name}' error: {exc}"}


# ===========================================================================
# ReAct agent loop
# ===========================================================================


def run_react_agent(user_question: str, *, max_iterations: int = 8) -> str:
    """Run the ReAct loop for *user_question* and return the final answer.

    The loop sends messages (including tool results) to the model until it
    produces a plain text answer or *max_iterations* is reached.
    """
    # --- Build the initial message history ---
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_question},
    ]

    for iteration in range(1, max_iterations + 1):
        # --- Model request ---
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0,
        )

        assistant_message = response.choices[0].message

        # --- Tool-call detection ---
        # If the model did NOT request any tool calls, we have the final answer.
        if not assistant_message.tool_calls:
            # --- Final-answer return ---
            return assistant_message.content or "(No response from model.)"

        # Append the assistant's message (with its tool_calls) to history.
        messages.append(assistant_message.model_dump())

        # --- Tool execution (supports multiple parallel tool calls) ---
        for tool_call in assistant_message.tool_calls:
            fn_name: str = tool_call.function.name
            raw_args: str = tool_call.function.arguments

            # Parse the tool arguments from JSON.
            try:
                parsed_args = json.loads(raw_args)
            except (json.JSONDecodeError, TypeError):
                parsed_args = None

            # Ensure parsed arguments are a dictionary.
            if not isinstance(parsed_args, dict):
                tool_result: dict[str, Any] = {
                    "success": False,
                    "error": f"Could not parse arguments for '{fn_name}': {raw_args}",
                }
            else:
                # Execute the requested tool.
                tool_result = execute_tool(fn_name, parsed_args)

            # --- Observation / tool-result insertion ---
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": fn_name,
                    "content": json.dumps(tool_result),
                }
            )

        print(f"  [ReAct loop iteration {iteration}: executed tool(s)]")

    # Maximum iterations reached without a final answer.
    return "(Reached maximum iterations without a final answer.)"


# ===========================================================================
# Interactive terminal loop
# ===========================================================================


def main() -> None:
    """Run the interactive terminal REPL."""
    print("\nOpenRouter ReAct Agent")
    print("Type 'exit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        # Handle exit commands and empty input.
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        try:
            answer = run_react_agent(user_input)
            print(f"\nAgent: {answer}\n")
        except OpenAIError as exc:
            print(f"\n[OpenAI API error] {exc}\n")
        except OSError as exc:
            print(f"\n[Network / IO error] {exc}\n")
        except ValueError as exc:
            print(f"\n[Value error] {exc}\n")


if __name__ == "__main__":
    main()
