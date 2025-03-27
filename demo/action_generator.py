#!/usr/bin/env -S uv run

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "quantalogic
#     "pathspec",
# ]
# ///


import ast
import asyncio
import inspect
from dataclasses import dataclass
from functools import partial
from typing import Callable, List, Dict, Any, Optional
import litellm
import typer
from loguru import logger

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
    example: Optional[str] = None

# Define base Tool class
class Tool:
    def __init__(self, name: str, description: str, arguments: List[ToolArgument], return_type: str, func: Callable):
        self.name = name
        self.description = description
        self.arguments = arguments
        self.return_type = return_type
        self.func = func

    async def async_execute(self, **kwargs) -> Any:
        """Execute the tool asynchronously."""
        if asyncio.iscoroutinefunction(self.func):
            return await self.func(**kwargs)
        return self.func(**kwargs)

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

# Tool creation function
def create_tool(func: Callable) -> Tool:
    if not callable(func):
        raise ValueError("Input must be a callable function")

    try:
        source = inspect.getsource(func).strip()
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError) as e:
        raise ValueError(f"Failed to parse function source: {e}")

    if not tree.body or not isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise ValueError("Source must define a single function")
    func_def = tree.body[0]

    name = func_def.name
    docstring = ast.get_docstring(func_def) or ""
    description = docstring or f"Tool generated from {name}"
    
    from typing import get_type_hints
    type_hints = get_type_hints(func)
    type_map = {int: "int", str: "string", float: "float", bool: "boolean"}

    args = func_def.args
    defaults = [None] * (len(args.args) - len(args.defaults)) + [
        ast.unparse(d) if isinstance(d, ast.AST) else str(d) for d in args.defaults
    ]
    arguments: List[ToolArgument] = []

    for i, arg in enumerate(args.args):
        arg_name = arg.arg
        default = defaults[i]
        hint = type_hints.get(arg_name, str)
        arg_type = type_map.get(hint, "string")
        arguments.append(ToolArgument(
            name=arg_name,
            arg_type=arg_type,
            description=f"Argument {arg_name}",
            required=default is None,
            default=default
        ))

    return_type = type_map.get(type_hints.get("return", str), "string")
    return Tool(name=name, description=description, arguments=arguments, return_type=return_type, func=func)

# Define tool functions
async def add(a: int, b: int) -> int:
    """Add two numbers and return the sum."""
    return a + b

async def multiply(x: int, y: int) -> int:
    """Multiply two numbers and return the product."""
    return x * y

async def concat(s1: str, s2: str) -> str:
    """Concatenate two strings and return the result."""
    return s1 + s2

async def agent(system_prompt: str, prompt: str, temperature: float) -> str:
    """Generate text using a language model based on a system prompt and user prompt."""
    try:
        response = await litellm.acompletion(
            model="gemini/gemini-2.0-flash",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Failed to generate text: {str(e)}")
        raise typer.BadParameter(f"Failed to generate text: {str(e)}")

# Rest of the code remains largely the same, just update the tools initialization
async def generate_program(task_description: str, tools: List[Tool], model: str, max_tokens: int) -> str:
    logger.debug(f"Generating program for task: {task_description}")
    tool_docstrings = "\n\n".join([tool.to_docstring() for tool in tools])
    # ... rest of the function remains the same ...

async def generate_core(task: str, model: str, max_tokens: int) -> None:
    logger.info(f"Starting generate command for task: {task}")
    if not task.strip():
        raise typer.BadParameter("Task description cannot be empty")
    if max_tokens <= 0:
        raise typer.BadParameter("max-tokens must be a positive integer")

    # Initialize tools using create_tool
    tools = [
        create_tool(add),
        create_tool(multiply),
        create_tool(concat),
        create_tool(agent)
    ]

    # ... rest of the function remains similar, just update namespace ...
    program = await generate_program(task, tools, model, max_tokens)
    typer.echo(typer.style("Generated Python Program:", fg=typer.colors.GREEN, bold=True))
    typer.echo(program)

    # Validate and execute
    ast_tree = ast.parse(program)
    has_async_main = any(
        isinstance(node, ast.AsyncFunctionDef) and node.name == "main"
        for node in ast.walk(ast_tree)
    )
    if not has_async_main:
        typer.echo(typer.style("Warning: Generated code lacks an async main() function", fg=typer.colors.YELLOW))
        return

    namespace: Dict[str, Callable] = {
        "asyncio": asyncio,
        "add": partial(tools[0].async_execute),
        "multiply": partial(tools[1].async_execute),
        "concat": partial(tools[2].async_execute),
        "agent": partial(tools[3].async_execute),
    }

    # ... rest of execution logic remains the same ...

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