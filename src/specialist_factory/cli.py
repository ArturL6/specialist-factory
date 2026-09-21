from __future__ import annotations

import json

import typer

from .annotate import annotate as run_annotate
from .config import load_config
from .evaluate import evaluate as run_evaluation
from .generate import generate as run_generate
from .schemas import Sample
from .training.engine import load_student
from .training.engine import predict as run_predict
from .training.engine import train as run_train

app = typer.Typer(add_completion=False, help="Teacher-to-specialist POC")

@app.command()
def generate(config: str = typer.Option(..., "--config"), count: int = typer.Option(48, "--count")):
    cfg = load_config(config); rows = run_generate(cfg, count); typer.echo(json.dumps({"generated": len(rows), "data_path": cfg.data_path}))

@app.command()
def annotate(config: str = typer.Option(..., "--config")):
    cfg = load_config(config); signals, labels = run_annotate(cfg)
    typer.echo(json.dumps({"signals": len(signals), "aggregated": len(labels), "accepted": sum(not row.abstained for row in labels), "run_dir": cfg.run_dir}))

@app.command()
def train(config: str = typer.Option(..., "--config")):
    cfg = load_config(config); checkpoint = run_train(cfg); typer.echo(json.dumps({"checkpoint": str(checkpoint)}))

@app.command()
def evaluate(config: str = typer.Option(..., "--config"), checkpoint: str = typer.Option(..., "--checkpoint")):
    report = run_evaluation(load_config(config), checkpoint); typer.echo(json.dumps({key: value for key, value in report.items() if key != "student_predictions"}, indent=2))

@app.command()
def predict(checkpoint: str = typer.Option(..., "--checkpoint"), text: str | None = typer.Option(None, "--text"), image: str | None = typer.Option(None, "--image")):
    if not text and not image: raise typer.BadParameter("Provide --text and/or --image")
    module = load_student(checkpoint); result = run_predict(module, [Sample(id="prediction", text=text, image=image)])[0]; typer.echo(json.dumps(result, indent=2))
