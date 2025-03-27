#!/usr/bin/env -S uv run

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "litellm",
#     "typer",
#     "loguru",
#     "aiohttp",
#     "quantalogic_pythonbox"
# ]
# ///
#!/usr/bin/env -S uv run

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "litellm",
#     "typer",
#     "loguru",
#     "aiohttp"
# ]
# ///

import ast
import asyncio
from typing import Callable, Optional
import inspect
import litellm
import typer
from loguru import logger
import aiohttp
from quantalogic_pythonbox import execute_async

# Configure logging
logger.add("action_gen.log", rotation="10 MB", level="DEBUG")
app = typer.Typer()

class ToolExecutor:
    def __init__(self):
        self.tools = {}

    def register(self, tool: Callable):
        self.tools[tool.name] = tool

    async def execute(self, tool_name: str, **kwargs) -> str:
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not found")
        
        tool = self.tools[tool_name]
        code = f"async def {tool.name}({', '.join([arg['name'] for arg in tool.arguments])}):\n    return {tool.name}({', '.join([arg['name'] for arg in tool.arguments])})"
        
        result = await execute_async(
            code=code,
            entry_point=tool.name,
            kwargs=kwargs,
            timeout=300
        )
        
        if result.error:
            raise RuntimeError(f"Tool execution failed: {result.error}")
        return result.result

# Tool creation function (unchanged)
def make_tool(func: Callable, name: str, description: str, executor: ToolExecutor, model: str = None) -> Callable:
    """Create a callable tool from a function with metadata and docstring generation."""
    from typing import get_type_hints
    type_hints = get_type_hints(func)
    type_map = {int: "int", str: "string", float: "float", bool: "boolean"}

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

    signature_parts = [f"{arg['name']}: {arg['arg_type']}" + (f" = {defaults[i]}" if i < len(defaults) and defaults[i] else "")
                      for i, arg in enumerate(arguments)]
    docstring = f"""async def {name}({", ".join(signature_parts)}) -> {return_type}
\n\n{description}\n\nArgs:\n""" + \
                "\n".join(f"    {arg['name']} ({arg['arg_type']}): {arg['description']}" for arg in arguments) + \
                f"\n\nReturns:\n    {return_type}: {description}\n"

    async def tool(**kwargs) -> str:
        logger.info(f"Starting tool execution: {name}")
        try:
            result = await executor.execute(name, **kwargs)
            logger.info(f"Finished tool execution: {name}")
            return result
        except Exception as e:
            logger.error(f"Tool {name} failed: {str(e)}")
            raise

    tool.name = name
    tool.description = description
    tool.arguments = arguments
    tool.return_type = return_type
    tool.to_docstring = lambda: docstring
    executor.register(tool)
    return tool

# Define tools (updated)
executor = ToolExecutor()

def register_tool(func):
    """Decorator to register a tool with the executor"""
    name = func.__name__
    description = func.__doc__ or ""
    return make_tool(func, name, description, executor)

@register_tool
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@register_tool
def multiply(x: int, y: int) -> int:
    """Multiply two numbers"""
    return x * y

@register_tool
def concat(s1: str, s2: str) -> str:
    """Concatenate two strings"""
    return s1 + s2

@register_tool
async def wikipedia_search(query: str, timeout: float = 10.0) -> Optional[str]:
    """
    Performs an asynchronous search query using the Wikipedia API.
    """
    try:
        timeout_config = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=timeout_config) as session:
            url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": query,
                "srlimit": 1
            }

            async with session.get(url, params=params) as response:
                if response.status != 200:
                    logger.error(f"Wikipedia API returned status {response.status}")
                    return None

                data = await response.json()
                if not data.get("query", {}).get("search"):
                    return None

                page_title = data["query"]["search"][0]["title"]
                params = {
                    "action": "query",
                    "format": "json",
                    "prop": "extracts",
                    "exintro": True,
                    "explaintext": True,
                    "titles": page_title
                }

                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        logger.error(f"Wikipedia API returned status {response.status}")
                        return None

                    data = await response.json()
                    pages = data.get("query", {}).get("pages", {})
                    if not pages:
                        return None

                    return next(iter(pages.values())).get("extract")
    except Exception as e:
        logger.exception(f"Unexpected error during Wikipedia search: {e}")
        return None

@register_tool
def agent_tool(system_prompt: str, prompt: str, temperature: float) -> str:
    """
    Use litellm to generate text using a language model.
    """
    return litellm.acompletion(
        model="gemini/gemini-2.0-flash",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
        temperature=temperature,
    )

# Updated generate_core with ReAct Loop
async def generate_core(task: str, model: str, max_tokens: int, max_steps: int = 10) -> None:


    logger.info(f"Starting generate command for task: {task}")
    if not task.strip():
        logger.error("Task description is empty")
        raise typer.BadParameter("Task description cannot be empty")
    if max_tokens <= 0:
        logger.error("max-tokens must be positive")
        raise typer.BadParameter("max-tokens must be a positive integer")
    if max_steps <= 0:
        logger.error("max-steps must be positive")
        raise typer.BadParameter("max-steps must be a positive integer")

    # Initialize tools
    tools = [
        add,
        multiply,
        concat,
        wikipedia_search,
        agent_tool
    ]

    # Generate tool descriptions for the prompt
    tool_descriptions = "\n".join([
        f"- {tool.name}({', '.join([f'{arg['name']}: {arg['arg_type']}' for arg in tool.arguments])}) -> {tool.return_type}: {tool.description}"
        for tool in tools
    ])

    # Initialize history
    history = f"Task: {task}\n"
    typer.echo(typer.style("Starting ReAct Agent Loop:", fg=typer.colors.GREEN, bold=True))

    # ReAct Loop
    for step in range(max_steps):
        prompt = f"""
You are an AI agent tasked with solving the following problem: {task}

So far, the following has happened:
{history}

Now, decide the next action to take. You can use the following tools:
{tool_descriptions}

To take an action, respond with 'Action: tool_name(arg1=value1, arg2=value2)'.
When specifying arguments, use the correct types: integers without quotes, floats with decimal points, strings enclosed in single quotes, booleans as True or False.
If you think the task is completed, respond with 'Stop: [final answer]'.
"""
        logger.debug(f"Step {step} prompt:\n{prompt}")

        try:
            response = await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.3
            )
            action_text = response.choices[0].message.content.strip()
            logger.debug(f"Model response: {action_text}")
            typer.echo(f"\nStep {step + 1}: {action_text}")

            if action_text.startswith("Stop:"):
                final_answer = action_text[len("Stop:"):].strip()
                typer.echo(typer.style("\nFinal Answer:", fg=typer.colors.GREEN, bold=True))
                typer.echo(final_answer)
                return

            elif action_text.startswith("Action:"):
                action_str = action_text[len("Action:"):].strip()
                try:
                    tool_name, args_str = action_str.split("(", 1)
                    args_str = args_str.rstrip(")")
                    args_list = [arg.strip() for arg in args_str.split(",")]
                    args_dict = {}
                    for arg in args_list:
                        if "=" in arg:
                            key, value = arg.split("=", 1)
                            args_dict[key.strip()] = ast.literal_eval(value.strip())
                        else:
                            raise ValueError(f"Invalid argument format: {arg}")

                    tool = next((t for t in tools if t.name == tool_name), None)
                    if not tool:
                        raise ValueError(f"Unknown tool: {tool_name}")

                    result = await executor.execute(tool_name, **args_dict)
                    history += f"Action: {action_str}\nResult: {result}\n"
                    typer.echo(f"Result: {result}")

                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    history += f"Action: {action_str}\n{error_msg}\n"
                    typer.echo(typer.style(error_msg, fg=typer.colors.RED))

            else:
                error_msg = f"Invalid response: {action_text}"
                history += f"{error_msg}\n"
                typer.echo(typer.style(error_msg, fg=typer.colors.YELLOW))

        except Exception as e:
            error_msg = f"Error in step {step}: {str(e)}"
            logger.error(error_msg)
            history += f"{error_msg}\n"
            typer.echo(typer.style(error_msg, fg=typer.colors.RED))

    typer.echo(typer.style("\nMaximum steps reached without completing the task.", fg=typer.colors.RED))

# Updated CLI with max_steps option
@app.command()
def generate(
    task: str = typer.Argument(..., help="The task description to solve"),
    model: str = typer.Option("gemini/gemini-2.0-flash", "--model", "-m", help="Language model to use"),
    max_tokens: int = typer.Option(4000, "--max-tokens", "-t", help="Maximum tokens for model responses"),
    max_steps: int = typer.Option(10, "--max-steps", "-s", help="Maximum steps for the ReAct loop")
) -> None:
    asyncio.run(generate_core(task, model, max_tokens, max_steps))

def main() -> None:
    logger.debug("Starting script execution")
    app()

if __name__ == "__main__":
    main()