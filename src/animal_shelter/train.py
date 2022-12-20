from pathlib import Path
import typer

def train(input_path: Path):
    typer.echo(f"Loading {input_path}") 

if __name__ == "__main__":
    typer.run(train)
