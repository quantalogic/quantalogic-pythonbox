#!/usr/bin/env python3

import typer

app = typer.Typer()

@app.command()
def test_command(
    task: str = typer.Argument(..., help="The task description"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Quiet mode")
):
    print(f"Task: {task}")
    print(f"Quiet: {quiet}")

if __name__ == "__main__":
    app()
