#!/usr/bin/env -S uv run

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "litellm",
#     "typer",
#     "loguru"
# ]
# ///


import ast
import asyncio
from asyncio import TimeoutError
from contextlib import AsyncExitStack
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Any, Optional

import litellm
import typer
from loguru import logger

# Configure logging
logger.add("action_gen.log", rotation="10 MB", level="DEBUG")
app = typer.Typer()

# Define ToolArgument dataclass
@dataclass
class ToolArgument:
    name: str
    arg_type: str
    description: str
    required: bool
    default: Optional[str] = None

# Define base Tool class
class Tool:
    def __init__(self, name: str, description: str, arguments: List[ToolArgument], return_type: str):
        self.name = name
        self.description = description
        self.arguments = arguments
        self.return_type = return_type

    async def async_execute(self, **kwargs) -> str:
        """Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement async_execute")

    def to_docstring(self) -> str:
        """Convert the tool definition into a Google-style docstring with function signature."""
        signature_parts = []
        for arg in self.arguments:
            arg_str = f"{arg.name}: {arg.arg_type}"
            if arg.default is not None:
                arg_str += f" = {arg.default}"
            signature_parts.append(arg_str)
        signature = f"async def {self.name}({', '.join(signature_parts)}) -> {self.return_type}"
        args_doc = "\n".join(f"    {arg.name} ({arg.arg_type}): {arg.description}" for arg in self.arguments)
        return f'"""\n{signature}\n\n{self.description}\n\nArgs:\n{args_doc}\n\nReturns:\n    {self.return_type}: {self.description}\n"""'

# Define tool implementations
class AddTool(Tool):
    def __init__(self):
        super().__init__(
            name="add_tool",
            description="Adds two numbers and returns the sum.",
            arguments=[
                ToolArgument(name="a", arg_type="int", description="First number", required=True),
                ToolArgument(name="b", arg_type="int", description="Second number", required=True)
            ],
            return_type="int"
        )
    
    async def async_execute(self, **kwargs) -> str:
        logger.info(f"Starting tool execution: {self.name}")
        a, b = int(kwargs["a"]), int(kwargs["b"])
        logger.info(f"Adding {a} and {b}")
        result = str(a + b)
        logger.info(f"Finished tool execution: {self.name}")
        return result

class MultiplyTool(Tool):
    def __init__(self):
        super().__init__(
            name="multiply_tool",
            description="Multiplies two numbers and returns the product.",
            arguments=[
                ToolArgument(name="x", arg_type="int", description="First number", required=True),
                ToolArgument(name="y", arg_type="int", description="Second number", required=True)
            ],
            return_type="int"
        )
    
    async def async_execute(self, **kwargs) -> str:
        logger.info(f"Starting tool execution: {self.name}")
        x, y = int(kwargs["x"]), int(kwargs["y"])
        logger.info(f"Multiplying {x} and {y}")
        result = str(x * y)
        logger.info(f"Finished tool execution: {self.name}")
        return result

class ConcatTool(Tool):
    def __init__(self):
        super().__init__(
            name="concat_tool",
            description="Concatenates two strings.",
            arguments=[
                ToolArgument(name="s1", arg_type="string", description="First string", required=True),
                ToolArgument(name="s2", arg_type="string", description="Second string", required=True)
            ],
            return_type="string"
        )
    
    async def async_execute(self, **kwargs) -> str:
        logger.info(f"Starting tool execution: {self.name}")
        s1, s2 = kwargs["s1"], kwargs["s2"]
        logger.info(f"Concatenating '{s1}' and '{s2}'")
        result = s1 + s2
        logger.info(f"Finished tool execution: {self.name}")
        return result

class AgentTool(Tool):
    def __init__(self, model: str = "gemini/gemini-2.0-flash"):
        super().__init__(
            name="agent_tool",
            description="Generates text using a language model based on a system prompt and user prompt.",
            arguments=[
                ToolArgument(name="system_prompt", arg_type="string", description="System prompt to guide the model", required=True),
                ToolArgument(name="prompt", arg_type="string", description="User prompt to generate a response for", required=True),
                ToolArgument(name="temperature", arg_type="float", description="Temperature for generation (0 to 1)", required=True)
            ],
            return_type="string"
        )
        self.model = model
    
    async def async_execute(self, **kwargs) -> str:
        logger.info(f"Starting tool execution: {self.name}")
        system_prompt = kwargs["system_prompt"]
        prompt = kwargs["prompt"]
        temperature = float(kwargs["temperature"])

        if not 0 <= temperature <= 1:
            logger.error(f"Temperature {temperature} is out of range (0-1)")
            raise ValueError("Temperature must be between 0 and 1")

        logger.info(f"Generating text with model {self.model}, temperature {temperature}")
        try:
            async with AsyncExitStack():
                response = await litellm.acompletion(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=1000
                )
                result = response.choices[0].message.content.strip()
                logger.debug(f"Generated text: {result}")
                logger.info(f"Finished tool execution: {self.name}")
                return result
        except TimeoutError as e:
            logger.error(f"API call to {self.model} timed out")
            raise TimeoutError(f"API call timed out: {str(e)}") from e
        except Exception as e:
            logger.error(f"Failed to generate text: {str(e)}")
            raise RuntimeError(f"Text generation failed: {str(e)}")

async def generate_program(task_description: str, tools: List[Tool], model: str, max_tokens: int) -> str:
    logger.debug(f"Generating program for task: {task_description}")
    tool_docstrings = "\n\n".join([tool.to_docstring() for tool in tools])

    prompt = f"""
You are a Python code generator. Your task is to create a Python program that solves the following task:
"{task_description}"

You have access to the following pre-defined async tool functions:

{tool_docstrings}

Instructions:
1. Generate a Python program as a single string.
2. Include only 'import asyncio' as the import statement.
3. Define an async function named 'main()' that solves the task.
4. Use the pre-defined tool functions (e.g., add_tool, multiply_tool, concat_tool, agent_tool) directly by calling them with 'await' and the appropriate arguments.
5. Do not redefine the tool functions; assume they are available in the namespace.
6. Return the program as a Markdown code block (```python ... ```).
7. Do not include asyncio.run(main()) or any 'if __name__ == "__main__":' block; the runtime will handle execution.
8. Use multiline strings for all string variables, starting each string at the beginning of a line.
9. Always print the result or key outputs at the end of the program.

Example task: "Add 5 and 7 and print the result"
Example output:
```python
import asyncio

async def main():
    result = await add_tool(a=5, b=7)
    print(result)
```
"""
    logger.debug(f"Prompt sent to litellm:\n{prompt}")

    try:
        response = await litellm.acompletion(
            model=model,
            messages=[
                {"role": "system", "content": "You are a Python code generator."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.3
        )
        generated_code = response.choices[0].message.content.strip()
        logger.debug(f"Generated code:\n{generated_code}")
    except Exception as e:
        logger.error(f"Failed to generate code: {str(e)}")
        raise typer.BadParameter(f"Failed to generate code: {str(e)}")

    # Clean up Markdown markers
    if generated_code.startswith("```python"):
        generated_code = generated_code[len("```python"):].strip()
    if generated_code.endswith("```"):
        generated_code = generated_code[:-3].strip()

    return generated_code

async def generate_core(task: str, model: str, max_tokens: int) -> None:
    logger.info(f"Starting generate command for task: {task}")
    if not task.strip():
        logger.error("Task description is empty")
        raise typer.BadParameter("Task description cannot be empty")
    if max_tokens <= 0:
        logger.error("max-tokens must be positive")
        raise typer.BadParameter("max-tokens must be a positive integer")

    # Initialize tools
    tools = [
        AddTool(),
        MultiplyTool(),
        ConcatTool(),
        AgentTool(model=model)
    ]

    # Generate the program
    try:
        program = await generate_program(task, tools, model, max_tokens)
    except Exception as e:
        logger.error(f"Program generation failed: {str(e)}")
        typer.echo(typer.style(f"Error: {str(e)}", fg=typer.colors.RED))
        raise typer.Exit(code=1)

    logger.debug(f"Generated program:\n{program}")
    typer.echo(typer.style("Generated Python Program:", fg=typer.colors.GREEN, bold=True))
    typer.echo(program)

    # Validate syntax
    try:
        ast_tree = ast.parse(program)
        has_async_main = any(
            isinstance(node, ast.AsyncFunctionDef) and node.name == "main"
            for node in ast.walk(ast_tree)
        )
        if not has_async_main:
            logger.warning("Generated code lacks an async main() function")
            typer.echo(typer.style("Warning: Generated code lacks an async main() function", fg=typer.colors.YELLOW))
            return
    except SyntaxError as e:
        logger.error(f"Syntax error in generated code: {str(e)}")
        typer.echo(typer.style(f"Syntax error: {str(e)}", fg=typer.colors.RED))
        return

    # Prepare namespace
    namespace: Dict[str, Callable] = {
        "asyncio": asyncio,
        "add_tool": partial(AddTool().async_execute),
        "multiply_tool": partial(MultiplyTool().async_execute),
        "concat_tool": partial(ConcatTool().async_execute),
        "agent_tool": partial(AgentTool(model=model).async_execute),
    }

    # Execute the program with timeout
    typer.echo("\n" + typer.style("Executing the program:", fg=typer.colors.GREEN, bold=True))
    try:
        exec(program, namespace)
        main_func = namespace.get("main")
        if not main_func or not asyncio.iscoroutinefunction(main_func):
            logger.error("No valid async main() function found")
            typer.echo(typer.style("Error: No valid async main() function found", fg=typer.colors.RED))
            return

        async with asyncio.timeout(30):  # 30-second timeout
            start_time = asyncio.get_event_loop().time()
            result = await main_func()
            execution_time = asyncio.get_event_loop().time() - start_time
            logger.info(f"Execution completed in {execution_time:.2f} seconds")
            typer.echo(typer.style(f"Execution completed in {execution_time:.2f} seconds", fg=typer.colors.GREEN))
            if result is not None:
                typer.echo("\n" + typer.style("Result:", fg=typer.colors.BLUE, bold=True))
                typer.echo(str(result))
    except TimeoutError:
        logger.error("Execution timed out after 30 seconds")
        typer.echo(typer.style("Error: Execution timed out after 30 seconds", fg=typer.colors.RED))
    except Exception as e:
        logger.error(f"Execution error: {str(e)}")
        typer.echo(typer.style(f"Error: {str(e)}", fg=typer.colors.RED))

@app.command()
def generate(
    task: str = typer.Argument(...),
    model: str = typer.Option("gemini/gemini-2.0-flash", "--model", "-m"),
    max_tokens: int = typer.Option(4000, "--max-tokens", "-t")
) -> None:
    asyncio.run(generate_core(task, model, max_tokens))

def main() -> None:
    logger.debug("Starting script execution")
    app()

if __name__ == "__main__":
    main()
