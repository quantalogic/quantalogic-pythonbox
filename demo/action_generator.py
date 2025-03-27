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
from functools import partial
from typing import Callable, Dict, List
import inspect
import litellm
import typer
from loguru import logger

# Configure logging
logger.add("action_gen.log", rotation="10 MB", level="DEBUG")
app = typer.Typer()

# Tool creation function
def make_tool(func: Callable, name: str, description: str, model: str = None) -> Callable:
    """Create a callable tool from a function with metadata and docstring generation."""
    # Extract signature and type hints
    from typing import get_type_hints
    type_hints = get_type_hints(func)
    type_map = {int: "int", str: "string", float: "float", bool: "boolean"}

    # Parse function AST for arguments
    tree = ast.parse(inspect.getsource(func).strip())
    func_def = tree.body[0]
    if isinstance(func_def, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = func_def.args
        defaults = [None] * (len(args.args) - len(args.defaults)) + [
            ast.unparse(d) if isinstance(d, ast.AST) else str(d) for d in args.defaults
        ]
        arguments = [
            {"name": arg.arg, "arg_type": type_map.get(type_hints.get(arg.arg, str), "string"),
             "description": f"Argument {arg.arg}", "required": defaults[i] is None}
            for i, arg in enumerate(args.args)
        ]
    elif isinstance(func_def, ast.Expr) and isinstance(func_def.value, ast.Lambda):
        lambda_args = func_def.value.args
        arguments = [
            {"name": arg.arg, "arg_type": type_map.get(type_hints.get(arg.arg, str), "string"),
             "description": f"Argument {arg.arg}", "required": True}
            for arg in lambda_args.args
        ]
        defaults = [None] * len(lambda_args.args)
    else:
        raise ValueError("Unsupported function type")

    return_type = type_map.get(type_hints.get("return", str), "string")

    # Generate docstring
    signature_parts = [f"{arg['name']}: {arg['arg_type']}" + (f" = {defaults[i]}" if i < len(defaults) and defaults[i] else "")
                      for i, arg in enumerate(arguments)]
    docstring = f"""async def {name}({", ".join(signature_parts)}) -> {return_type}
\n\n{description}\n\nArgs:\n""" + \
                "\n".join(f"    {arg['name']} ({arg['arg_type']}): {arg['description']}" for arg in arguments) + \
                f"\n\nReturns:\n    {return_type}: {description}\n"

    # Define the tool as a callable with metadata
    async def tool(**kwargs) -> str:
        logger.info(f"Starting tool execution: {name}")
        try:
            result = await func(**kwargs)
            result = str(result)
            logger.info(f"Finished tool execution: {name}")
            return result
        except Exception as e:
            logger.error(f"Tool {name} failed: {str(e)}")
            raise

    # Attach metadata
    tool.name = name
    tool.description = description
    tool.arguments = arguments
    tool.return_type = return_type
    tool.to_docstring = lambda: docstring
    return tool

# Define tools as callables
async def add(a: int, b: int) -> int:
    return a + b

async def multiply(x: int, y: int) -> int:  # Removed stray hyphen
    return x * y

async def concat(s1: str, s2: str) -> str:
    return s1 + s2

async def agent_func(system_prompt: str, prompt: str, temperature: float, model: str) -> str:
    """Generate text using a language model."""
    if not 0 <= temperature <= 1:
        raise ValueError("Temperature must be between 0 and 1")
    logger.info(f"Generating text with model {model}, temperature {temperature}")
    async with AsyncExitStack():
        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()

async def generate_program(task_description: str, tools: List[Callable], model: str, max_tokens: int) -> str:
    logger.debug(f"Generating program for task: {task_description}")
    tool_docstrings = "\n\n".join([tool.to_docstring() for tool in tools])

    prompt = f"""
You are a Python code generator. Your task is to create a Python program that solves:
"{task_description}"

Available async tool functions:
{tool_docstrings}

Instructions:
1. Return a Python program as a single string in a Markdown code block (```python ... ```).
2. Include only 'import asyncio' as the import statement.
3. Define an async function 'main()' to solve the task.
4. Use the tools (e.g., add_tool, multiply_tool) with 'await' and appropriate arguments.
5. Do not redefine tools; assume they are in the namespace.
6. Exclude asyncio.run(main()) or 'if __name__ == "__main__":' blocks.
7. Use multiline strings starting at the line beginning.
8. Print key outputs at the end.

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
            messages=[{"role": "system", "content": "You are a Python code generator."}, {"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.3
        )
        generated_code = response.choices[0].message.content.strip()
        logger.debug(f"Generated code:\n{generated_code}")
    except Exception as e:
        logger.error(f"Failed to generate code: {str(e)}")
        raise typer.BadParameter(f"Failed to generate code: {str(e)}")

    # Clean Markdown markers
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

    # Define a wrapper for agent_func to bind the model
    async def agent_tool_wrapper(system_prompt: str, prompt: str, temperature: float) -> str:
        return await agent_func(system_prompt, prompt, temperature, model)

    # Initialize tools
    tools = [
        make_tool(add, "add_tool", "Adds two numbers and returns the sum."),
        make_tool(multiply, "multiply_tool", "Multiplies two numbers and returns the product."),
        make_tool(concat, "concat_tool", "Concatenates two strings."),
        make_tool(agent_tool_wrapper, "agent_tool", "Generates text using a language model.")
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
        if not any(isinstance(node, ast.AsyncFunctionDef) and node.name == "main" for node in ast.walk(ast_tree)):
            logger.warning("Generated code lacks an async main() function")
            typer.echo(typer.style("Warning: Generated code lacks an async main() function", fg=typer.colors.YELLOW))
            return
    except SyntaxError as e:
        logger.error(f"Syntax error in generated code: {str(e)}")
        typer.echo(typer.style(f"Syntax error: {str(e)}", fg=typer.colors.RED))
        return

    # Prepare namespace
    namespace: Dict[str, Callable] = {tool.name: partial(tool) for tool in tools}
    namespace["asyncio"] = asyncio

    # Execute the program
    typer.echo("\n" + typer.style("Executing the program:", fg=typer.colors.GREEN, bold=True))
    try:
        exec(program, namespace)
        main_func = namespace.get("main")
        if not main_func or not asyncio.iscoroutinefunction(main_func):
            logger.error("No valid async main() function found")
            typer.echo(typer.style("Error: No valid async main() function found", fg=typer.colors.RED))
            return

        async with asyncio.timeout(30):
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
