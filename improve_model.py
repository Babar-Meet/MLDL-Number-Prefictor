"""Train an improved model while preserving the best previous checkpoint."""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from backend.model_utils import (
    DATA_DIR,
    DEVICE,
    MODEL_PATH,
    VisualizerNet,
    evaluate_model,
    export_model_snapshot,
)


PROJECT_ROOT = Path(__file__).resolve().parent
PRIMARY_MODEL_PATH = MODEL_PATH
PREVIOUS_MODEL_PATH = MODEL_PATH.with_name("mnist_visualizer_model_previous_best.pth")
CHECKPOINT_META_PATH = PROJECT_ROOT / "backend" / "model_checkpoints.json"
DEFAULT_EPOCHS = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Improve the MNIST showcase model.")
    parser.add_argument("--epochs", type=int, help="How many epochs to train for.")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Adam learning rate.")
    return parser.parse_args()


def prompt_for_epochs(default_epochs: int = DEFAULT_EPOCHS) -> int:
    raw_value = input(f"Enter epochs for improvement training [{default_epochs}]: ").strip()
    if not raw_value:
        return default_epochs
    try:
        epochs = int(raw_value)
    except ValueError as exc:
        raise ValueError("Epochs must be a whole number.") from exc
    if epochs <= 0:
        raise ValueError("Epochs must be greater than 0.")
    return epochs


def get_requested_epochs(cli_epochs: int | None) -> int:
    if cli_epochs is not None:
        if cli_epochs <= 0:
            raise ValueError("Epochs must be greater than 0.")
        return cli_epochs
    return prompt_for_epochs()


def save_checkpoint_metadata(current_best_accuracy: float, previous_best_accuracy: float | None) -> None:
    payload = {
        "updatedAt": datetime.now(UTC).isoformat(),
        "currentBestModel": str(PRIMARY_MODEL_PATH),
        "previousBestModel": str(PREVIOUS_MODEL_PATH),
        "currentBestAccuracy": current_best_accuracy,
        "previousBestAccuracy": previous_best_accuracy,
    }
    CHECKPOINT_META_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_starting_model() -> tuple[VisualizerNet, Path | None]:
    model = VisualizerNet().to(DEVICE)
    for checkpoint_path in (PRIMARY_MODEL_PATH, PREVIOUS_MODEL_PATH):
        if not checkpoint_path.exists():
            continue
        try:
            state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
            model.load_state_dict(state_dict)
            return model, checkpoint_path
        except RuntimeError:
            print(f"Skipped incompatible checkpoint: {checkpoint_path}")
    return model, None


def rotate_best_models(current_best_accuracy: float | None, improved_model: VisualizerNet, improved_accuracy: float) -> None:
    if PRIMARY_MODEL_PATH.exists():
        shutil.copy2(PRIMARY_MODEL_PATH, PREVIOUS_MODEL_PATH)
        previous_best_accuracy = current_best_accuracy
    else:
        previous_best_accuracy = None

    torch.save(improved_model.state_dict(), PRIMARY_MODEL_PATH)
    save_checkpoint_metadata(improved_accuracy, previous_best_accuracy)


def train_improved(
    num_epochs: int,
    batch_size: int = 128,
    lr: float = 0.001,
) -> tuple[VisualizerNet, float, float, bool]:
    """Fine-tune from the best saved weights and keep a backup checkpoint."""
    augmented_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.92, 1.08)),
        transforms.ToTensor(),
    ])
    plain_transform = transforms.ToTensor()

    train_data = datasets.MNIST(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=augmented_transform,
    )
    test_data = datasets.MNIST(
        root=str(DATA_DIR),
        train=False,
        download=True,
        transform=plain_transform,
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    model, source_checkpoint = load_starting_model()
    if source_checkpoint is not None:
        print(f"Loaded starting checkpoint from {source_checkpoint}")
    else:
        print("No saved checkpoint found. Starting from scratch this one time.")

    baseline_accuracy = evaluate_model(model, test_loader)
    print(f"Current saved best accuracy: {baseline_accuracy:.2%}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    loss_fn = nn.CrossEntropyLoss()

    best_accuracy = baseline_accuracy
    improved = False

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        batch_count = 0
        for batch_index, (data, target) in enumerate(train_loader, start=1):
            data = data.to(DEVICE)
            target = target.to(DEVICE)
            logits = model(data)
            loss = loss_fn(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            batch_count += 1

            if batch_index % 100 == 0:
                avg_loss = running_loss / batch_count
                print(
                    f"  Epoch {epoch + 1}/{num_epochs} | batch {batch_index} | loss {avg_loss:.4f}"
                )

        scheduler.step()
        accuracy = evaluate_model(model, test_loader)
        print(
            f"  Epoch {epoch + 1}/{num_epochs} done | accuracy {accuracy:.2%} | lr {scheduler.get_last_lr()[0]:.6f}"
        )

        if accuracy > best_accuracy:
            prior_best_accuracy = best_accuracy
            best_accuracy = accuracy
            improved = True
            rotate_best_models(prior_best_accuracy, model, accuracy)
            print(f"  >>> New best model saved ({best_accuracy:.2%})")

    if not PRIMARY_MODEL_PATH.exists():
        rotate_best_models(None, model, best_accuracy)

    if not CHECKPOINT_META_PATH.exists():
        save_checkpoint_metadata(baseline_accuracy, None)

    best_model_state = torch.load(PRIMARY_MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(best_model_state)
    return model, baseline_accuracy, best_accuracy, improved


def main() -> None:
    args = parse_args()
    epochs = get_requested_epochs(args.epochs)

    print(f"Using device: {DEVICE}")
    print(f"Starting improved training ({epochs} epochs, data augmentation)...\n")

    model, baseline_accuracy, best_accuracy, improved = train_improved(
        num_epochs=epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
    )
    snapshot_path = export_model_snapshot()

    print()
    if improved:
        print(f"Improvement complete. Accuracy moved from {baseline_accuracy:.2%} to {best_accuracy:.2%}.")
    else:
        print(f"No better checkpoint found. Keeping saved best at {baseline_accuracy:.2%}.")
    print(model)
    print(f"Current best model: {PRIMARY_MODEL_PATH}")
    print(f"Previous best backup: {PREVIOUS_MODEL_PATH}")
    print(f"Checkpoint metadata: {CHECKPOINT_META_PATH}")
    print(f"Snapshot stored at {snapshot_path}")


if __name__ == '__main__':
    main()
