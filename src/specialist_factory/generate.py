from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw

from .config import write_jsonl
from .schemas import AppConfig, Sample


def _intent_examples() -> list[tuple[str, str]]:
    return [("Where is my package?", "order_status"), ("Track order number 17", "order_status"), ("My delivery has not arrived", "order_status"), ("Please cancel my order", "cancel_order"), ("I ordered by mistake, cancel it", "cancel_order"), ("Stop shipment immediately", "cancel_order"), ("I want my money back", "refund"), ("Please issue a refund", "refund"), ("The item arrived broken; return my payment", "refund"), ("What are your opening hours?", "other"), ("Can I change my account email?", "other"), ("Do you sell gift cards?", "other")]


def generate(config: AppConfig, count: int = 48) -> list[Sample]:
    rng = random.Random(config.training.seed)
    modality = config.task.modality.value
    samples: list[Sample] = []
    data_path = Path(config.data_path); image_dir = data_path.parent / "images"; image_dir.mkdir(parents=True, exist_ok=True)
    intents = _intent_examples()
    for index in range(count):
        label = config.task.labels[index % len(config.task.labels)]
        text = image = None
        if modality in ("text", "multimodal"):
            candidates = [item[0] for item in intents if item[1] == label] or [f"Example for {label}"]
            text = rng.choice(candidates)
        if modality in ("image", "multimodal"):
            image = str(image_dir / f"sample-{index:03d}.png")
            palette = {name: ((230, 70, 70), "scratch") for name in config.task.labels}
            palette.update({"scratch": ((220, 60, 60), "scratch"), "clean": ((70, 180, 80), "clean"), "damaged": ((220, 60, 60), "damage"), "undamaged": ((70, 180, 80), "clear")})
            color, tag = palette.get(label, ((90, 120, 210), label))
            canvas = Image.new("RGB", (96, 96), "white"); draw = ImageDraw.Draw(canvas)
            draw.rectangle((12, 12, 84, 84), fill=color)
            if tag in ("scratch", "damage"): draw.line((20, 20, 76, 76), fill="black", width=7)
            draw.text((5, 5), label[:8], fill="black"); canvas.save(image)
        # Synthetic gold is explicitly marked as such; it must never be presented as a human annotation.
        labeled_fraction = float(config.generator.get("synthetic_human_label_rate", 2 / 3))
        human = label if rng.random() < labeled_fraction else None
        metadata = {"synthetic": True, "generator": "builtin-v1", "label_source": "synthetic_gold_proxy" if human else "unlabeled"}
        samples.append(Sample(id=f"sample-{index:03d}", text=text, image=image, human_label=human, metadata=metadata))
    write_jsonl(data_path, samples)
    return samples
