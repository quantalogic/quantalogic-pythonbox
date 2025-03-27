# Quantalogic Python Sandbox with CodeAct Agent

Welcome to the **Quantalogic Python Sandbox**, a secure and extensible environment for executing Python code asynchronously, paired with the **CodeAct Agent**, a ReAct-based framework for task-solving using language models and tools. This project, developed as part of the `quantalogic_pythonbox` package, provides a robust sandbox for interpreting Python Abstract Syntax Trees (AST) and a demonstration of an autonomous agent capable of reasoning and acting on tasks.

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [The Python Sandbox](#the-python-sandbox)
5. [CodeAct Agent Explained](#codeact-agent-explained)
6. [Demo: `code_act_agent.py`](#demo-code_act_agentpy)
7. [Usage Examples](#usage-examples)
8. [References](#references)
9. [Contributing](#contributing)
10. [License](#license)

---

## Overview

The Quantalogic Python Sandbox is designed to safely execute Python code by interpreting its AST, providing fine-grained control over execution, security, and resource usage. Built with `asyncio` for asynchronous operations, it supports a wide range of Python constructs, from basic arithmetic to complex comprehensions and async functions.

The **CodeAct Agent**, showcased in `demo/code_act_agent.py`, leverages the ReAct (Reasoning + Acting) paradigm to solve tasks by combining language model reasoning with a suite of callable tools. This agent iteratively reasons about a task, selects actions, and executes them until a solution is reached, making it a powerful demonstration of AI-driven problem-solving within the sandbox.

---

## Features

- **Secure Execution**: Restricts access to dangerous modules (e.g., `os`, `sys`) and enforces memory/operation limits.
- **AST Interpretation**: Executes code by traversing its AST, supporting Python 3.12+ features like async functions, comprehensions, and pattern matching.
- **Asynchronous Support**: Built with `asyncio` for non-blocking execution of coroutines and generators.
- **CodeAct Framework**: Integrates a ReAct-based agent for task-solving with dynamic tool usage.
- **Extensible Tools**: Easily define and integrate custom tools for the agent to use.
- **Logging**: Detailed execution logs via `loguru` for debugging and monitoring.

---

## Installation

To get started, you'll need Python 3.12 or higher and the `uv` package manager (optional but recommended for dependency management). Clone the repository and install dependencies:

```bash
git clone https://github.com/your-org/quantalogic-python-sandbox.git
cd quantalogic-python-sandbox
uv pip install -r requirements.txt
```

Alternatively, install the required packages manually:

```bash
pip install litellm typer loguru aiohttp
```

---

## The Python Sandbox

The `quantalogic_pythonbox` package provides a custom AST interpreter (`ASTInterpreter`) that executes Python code in a controlled environment. Key components include:

- **Execution Engine (`execution.py`)**: Manages async execution with timeouts, memory limits, and constant folding optimizations.
- **Visitors (`*.py`)**: Handle specific AST nodes (e.g., `visit_ListComp` for list comprehensions, `visit_AsyncFunctionDef` for async functions).
- **Security**: Restricts imports to an allowlist (e.g., `asyncio`) and blocks OS-level operations by default.
- **Function Utilities (`function_utils.py`)**: Supports custom function types like `AsyncFunction` and `LambdaFunction`.

The sandbox ensures that code runs safely and efficiently, making it ideal for untrusted inputs or educational purposes.

---

## CodeAct Agent Explained

The **CodeAct Agent** is an implementation of the ReAct framework, which combines **reasoning** (via a language model) and **acting** (via tool execution) to solve tasks iteratively. Introduced by Yao et al. in their 2022 paper ["ReAct: Synergizing Reasoning and Acting in Language Models"](https://arxiv.org/abs/2210.03629), ReAct enables agents to:
1. **Reason**: Analyze the task and plan the next step.
2. **Act**: Execute actions using predefined tools.
3. **Observe**: Incorporate results into the reasoning process.
4. **Repeat**: Continue until the task is solved or a limit is reached.

### How CodeAct Works
- **Task Input**: The agent receives a task (e.g., "Calculate 5 + 3").
- **Toolset**: A collection of callable tools (e.g., `add_tool`, `wiki_tool`) is provided.
- **ReAct Loop**: For up to `max_steps` iterations:
  - The agent generates a prompt with the task, history, and tool descriptions.
  - A language model (e.g., `gemini-2.0-flash`) responds with either:
    - `Action: tool_name(arg1=value1, ...)` to execute a tool.
    - `Stop: [final answer]` to conclude.
  - The agent executes the action, updates the history, and repeats.
- **Output**: The final answer or an error if the task isn't solved within `max_steps`.

### Benefits
- **Flexibility**: Tools can be simple (e.g., arithmetic) or complex (e.g., Wikipedia search).
- **Traceability**: Each step is logged and displayed, making the process transparent.
- **Scalability**: Easily extend with new tools or integrate with different models.

---

## Demo: `code_act_agent.py`

The `demo/code_act_agent.py` script demonstrates the CodeAct Agent within the Python Sandbox. It defines a CLI application using `typer` and integrates tools with a ReAct loop.

### Key Components
1. **Tool Creation (`make_tool`)**:
   - Converts Python functions into tools with metadata (name, description, argument types).
   - Example: `add_tool` from `async def add(a: int, b: int) -> int`.

2. **Defined Tools**:
   - `add_tool`: Adds two integers.
   - `multiply_tool`: Multiplies two integers.
   - `concat_tool`: Concatenates two strings.
   - `wiki_tool`: Searches Wikipedia asynchronously using `aiohttp`.
   - `agent_tool`: Wraps the language model for text generation.

3. **ReAct Loop (`generate_core`)**:
   - Takes a task, model, and step/token limits as input.
   - Iteratively prompts the model, parses responses, and executes tools.
   - Logs actions and results using `loguru`.

4. **CLI Interface**:
   - Command: `generate "task description" --model "model_name" --max-tokens 4000 --max-steps 10`.
   - Outputs steps and the final answer in a colored `typer` interface.

### Code Highlights
```python
async def generate_core(task: str, model: str, max_tokens: int, max_steps: int = 10):
    tools = [
        make_tool(add, "add_tool", "Adds two numbers and returns the sum."),
        make_tool(wikipedia_search, "wiki_tool", "Performs a Wikipedia search.")
    ]
    history = f"Task: {task}\n"
    for step in range(max_steps):
        prompt = f"You are an AI agent tasked with solving: {task}\nHistory:\n{history}\nTools:\n{tool_descriptions}"
        response = await litellm.acompletion(model=model, messages=[{"role": "user", "content": prompt}])
        action_text = response.choices[0].message.content.strip()
        if action_text.startswith("Action:"):
            tool_name, args = parse_action(action_text)
            result = await execute_tool(tool_name, args, tools)
            history += f"Action: {action_text}\nResult: {result}\n"
        elif action_text.startswith("Stop:"):
            return action_text[len("Stop:"):].strip()
```

### Running the Demo
```bash
uv run demo/code_act_agent.py generate "What is 5 + 3?" --model "gemini/gemini-2.0-flash"
```
**Output**:
```
Starting ReAct Agent Loop:
Step 1: Action: add_tool(a=5, b=3)
Result: 8
Step 2: Stop: 8

Final Answer: 8
```

This shows the agent reasoning that it needs to add 5 and 3, executing `add_tool`, and stopping with the answer.

---

## Usage Examples

### Simple Arithmetic
```bash
uv run demo/code_act_agent.py generate "Calculate 5 * 4 + 2"
```
The agent might:
1. Multiply 5 and 4 (`multiply_tool`) → 20.
2. Add 2 (`add_tool`) → 22.
3. Stop with "22".

### Wikipedia Search
```bash
uv run demo/code_act_agent.py generate "What is the capital of France?"
```
The agent uses `wiki_tool(query='France')` to fetch the Wikipedia intro, extracts "Paris," and stops.

### Custom Tools
Add a new tool in `code_act_agent.py`:
```python
async def subtract(a: int, b: int) -> int:
    return a - b
tools.append(make_tool(subtract, "subtract_tool", "Subtracts b from a."))
```
Then run:
```bash
uv run demo/code_act_agent.py generate "What is 10 - 7?"
```

---

## References

- **ReAct Paper**: Yao, S., et al. (2022). "ReAct: Synergizing Reasoning and Acting in Language Models." [arXiv:2210.03629](https://arxiv.org/abs/2210.03629).
- **Python AST**: [Python Official Documentation](https://docs.python.org/3/library/ast.html).
- **asyncio**: [Python AsyncIO Documentation](https://docs.python.org/3/library/asyncio.html).
- **LiteLLM**: [LiteLLM GitHub](https://github.com/BerriAI/litellm) - For model integration.
- **Typer**: [Typer Documentation](https://typer.tiangolo.com/) - CLI framework.

---

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/new-tool`).
3. Commit changes (`git commit -m "Add new tool"`).
4. Push to the branch (`git push origin feature/new-tool`).
5. Open a pull request.

Please include tests and update documentation as needed.

---

## License

This project is licensed under the Apache License, Version 2.0 (the "License"). You may not use this project except in compliance with the License. A copy of the License is included in the `LICENSE` file, or you can obtain it at [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0).

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
