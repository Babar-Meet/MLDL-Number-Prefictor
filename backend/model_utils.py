from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data'
MODEL_PATH = PROJECT_ROOT / 'backend' / 'mnist_visualizer_model.pth'
SNAPSHOT_PATH = PROJECT_ROOT / 'backend' / 'model_snapshot.json'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
LAYER_SIZES = (784, 96, 48, 10)
LAYER_NAMES = ('input', 'hidden_1', 'hidden_2', 'output')

_MODEL_CACHE: 'VisualizerNet | None' = None


class VisualizerNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(LAYER_SIZES[0], LAYER_SIZES[1])
        self.drop1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(LAYER_SIZES[1], LAYER_SIZES[2])
        self.drop2 = nn.Dropout(0.25)
        self.fc3 = nn.Linear(LAYER_SIZES[2], LAYER_SIZES[3])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        hidden_1 = self.drop1(F.relu(self.fc1(x)))
        hidden_2 = self.drop2(F.relu(self.fc2(hidden_1)))
        return self.fc3(hidden_2)


def get_data() -> tuple[datasets.MNIST, datasets.MNIST]:
    train_transform = transforms.Compose([
        transforms.RandomRotation(12),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ToTensor(),
    ])
    test_transform = ToTensor()
    training_data = datasets.MNIST(root=str(DATA_DIR), train=True, download=True, transform=train_transform)
    test_data = datasets.MNIST(root=str(DATA_DIR), train=False, download=True, transform=test_transform)
    return training_data, test_data


def _state_dict_compatible(model: VisualizerNet, state_dict: dict[str, Any]) -> bool:
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        return False
    return True


def train_model(num_epochs: int = 6, batch_size: int = 128, learning_rate: float = 0.001) -> VisualizerNet:
    training_data, test_data = get_data()
    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    model = VisualizerNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    loss_fn = nn.CrossEntropyLoss()

    best_accuracy = 0.0
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_index, (data, target) in enumerate(train_loader, start=1):
            data = data.to(DEVICE)
            target = target.to(DEVICE)
            logits = model(data)
            loss = loss_fn(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())

            if batch_index % 150 == 0:
                average_loss = running_loss / 150
                print(f'Epoch {epoch + 1}/{num_epochs} | batch {batch_index} | loss {average_loss:.4f}')
                running_loss = 0.0

        scheduler.step()
        accuracy = evaluate_model(model, test_loader)
        print(f'Epoch {epoch + 1}/{num_epochs} | accuracy {accuracy:.2%}')

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Load best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), MODEL_PATH)
    print(f'Saved model to {MODEL_PATH}')
    print(f'Best test accuracy: {best_accuracy:.2%}')
    return model


def evaluate_model(model: VisualizerNet, test_loader: DataLoader | None = None) -> float:
    if test_loader is None:
        _, test_data = get_data()
        test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(DEVICE)
            target = target.to(DEVICE)
            prediction = model(data).argmax(dim=1)
            correct += int((prediction == target).sum().item())
            total += int(target.numel())
    return correct / max(total, 1)


def get_model(force_train: bool = False, force_reload: bool = False) -> VisualizerNet:
    global _MODEL_CACHE

    if _MODEL_CACHE is not None and not force_train and not force_reload:
        return _MODEL_CACHE

    model = VisualizerNet().to(DEVICE)

    if not force_train and MODEL_PATH.exists():
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        if _state_dict_compatible(model, state_dict):
            model.eval()
            _MODEL_CACHE = model
            return model
        print('Existing model weights are incompatible with the current architecture. Re-training.')

    _MODEL_CACHE = train_model()
    _MODEL_CACHE.eval()
    return _MODEL_CACHE


def _round_nested(values: Any, digits: int = 6) -> Any:
    if isinstance(values, list):
        return [_round_nested(value, digits) for value in values]
    if isinstance(values, float):
        return round(values, digits)
    return values


def _tensor_list(tensor: torch.Tensor, digits: int = 6) -> list[Any]:
    return _round_nested(tensor.detach().cpu().tolist(), digits)


def _stats(tensor: torch.Tensor) -> dict[str, float]:
    detached = tensor.detach().float().cpu()
    return {
        'min': round(float(detached.min().item()), 6),
        'max': round(float(detached.max().item()), 6),
        'mean': round(float(detached.mean().item()), 6),
        'std': round(float(detached.std().item()), 6),
    }


def _sparsity(tensor: torch.Tensor) -> float:
    detached = tensor.detach()
    return round(float((detached <= 1e-6).float().mean().item()), 6)


def _top_connections(matrix: torch.Tensor, limit: int = 60) -> list[dict[str, Any]]:
    values = matrix.detach().cpu().numpy()
    flat = np.abs(values).reshape(-1)
    if not flat.size:
        return []

    limit = min(limit, flat.size)
    indices = np.argpartition(flat, -limit)[-limit:]
    ordered = indices[np.argsort(flat[indices])[::-1]]
    connections: list[dict[str, Any]] = []
    columns = values.shape[1]

    for index in ordered:
        to_index = int(index // columns)
        from_index = int(index % columns)
        connections.append(
            {
                'from': from_index,
                'to': to_index,
                'weight': round(float(values[to_index, from_index]), 6),
                'strength': round(float(abs(values[to_index, from_index])), 6),
            }
        )

    return connections


def _top_active_connections(source: torch.Tensor, matrix: torch.Tensor, limit: int = 36) -> list[dict[str, Any]]:
    source_values = source.detach().cpu().numpy().reshape(-1)
    matrix_values = matrix.detach().cpu().numpy()
    if not source_values.size or not matrix_values.size:
        return []

    influence = np.abs(matrix_values * source_values[np.newaxis, :])
    flat = influence.reshape(-1)
    limit = min(limit, flat.size)
    indices = np.argpartition(flat, -limit)[-limit:]
    ordered = indices[np.argsort(flat[indices])[::-1]]
    columns = matrix_values.shape[1]
    connections: list[dict[str, Any]] = []

    for index in ordered:
        to_index = int(index // columns)
        from_index = int(index % columns)
        signal = float(source_values[from_index] * matrix_values[to_index, from_index])
        connections.append(
            {
                'from': from_index,
                'to': to_index,
                'signal': round(signal, 6),
                'magnitude': round(abs(signal), 6),
                'activation': round(float(source_values[from_index]), 6),
                'weight': round(float(matrix_values[to_index, from_index]), 6),
            }
        )

    return connections


def _layer_summary(name: str, activations: torch.Tensor, gradient: torch.Tensor | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        'name': name,
        'size': int(activations.numel()),
        'activations': _tensor_list(activations.reshape(-1)),
        'stats': _stats(activations),
        'sparsity': _sparsity(activations),
        'topNeurons': [
            {
                'index': int(index),
                'activation': round(float(activations.reshape(-1)[index].item()), 6),
            }
            for index in torch.topk(activations.reshape(-1), k=min(8, activations.numel())).indices.detach().cpu().tolist()
        ],
    }
    if gradient is not None:
        payload['gradient'] = _tensor_list(gradient.reshape(-1))
        payload['gradientStats'] = _stats(gradient.abs())
    return payload


def _prepare_adjustments(adjustments: dict[str, Any] | None) -> tuple[list[float], list[float], int | None]:
    default_scales = [1.0, 1.0, 1.0]
    default_bias_offsets = [0.0] * 10
    target_digit = None

    if not adjustments:
        return default_scales, default_bias_offsets, target_digit

    layer_scales = adjustments.get('layerScales', default_scales)
    if len(layer_scales) != 3:
        layer_scales = default_scales

    bias_offsets = adjustments.get('outputBiasOffsets', default_bias_offsets)
    if len(bias_offsets) != 10:
        bias_offsets = default_bias_offsets

    requested_target = adjustments.get('targetDigit')
    if requested_target is not None:
        target_digit = int(max(0, min(9, requested_target)))

    return [float(scale) for scale in layer_scales], [float(offset) for offset in bias_offsets], target_digit


def build_model_snapshot() -> dict[str, Any]:
    model = get_model()
    model.eval()

    weights = [
        {
            'name': 'input_to_hidden_1',
            'shape': [LAYER_SIZES[1], LAYER_SIZES[0]],
            'matrix': _tensor_list(model.fc1.weight),
            'bias': _tensor_list(model.fc1.bias),
            'stats': _stats(model.fc1.weight),
            'topConnections': _top_connections(model.fc1.weight),
            'receptiveFields': [_tensor_list(field.reshape(28, 28)) for field in model.fc1.weight[:18]],
        },
        {
            'name': 'hidden_1_to_hidden_2',
            'shape': [LAYER_SIZES[2], LAYER_SIZES[1]],
            'matrix': _tensor_list(model.fc2.weight),
            'bias': _tensor_list(model.fc2.bias),
            'stats': _stats(model.fc2.weight),
            'topConnections': _top_connections(model.fc2.weight),
        },
        {
            'name': 'hidden_2_to_output',
            'shape': [LAYER_SIZES[3], LAYER_SIZES[2]],
            'matrix': _tensor_list(model.fc3.weight),
            'bias': _tensor_list(model.fc3.bias),
            'stats': _stats(model.fc3.weight),
            'topConnections': _top_connections(model.fc3.weight),
        },
    ]

    return {
        'architecture': list(LAYER_SIZES),
        'layerNames': list(LAYER_NAMES),
        'device': str(DEVICE),
        'weights': weights,
        'outputLabels': list(range(10)),
    }


def export_model_snapshot(output_path: Path = SNAPSHOT_PATH) -> Path:
    snapshot = build_model_snapshot()
    output_path.write_text(json.dumps(snapshot), encoding='utf-8')
    return output_path


def analyze_digit(
    pixels: list[float] | list[list[float]],
    adjustments: dict[str, Any] | None = None,
    include_backprop: bool = True,
) -> dict[str, Any]:
    model = get_model()
    model.eval()

    layer_scales, bias_offsets, requested_target = _prepare_adjustments(adjustments)
    pixel_tensor = torch.tensor(pixels, dtype=torch.float32, device=DEVICE)
    if pixel_tensor.ndim == 2:
        pixel_tensor = pixel_tensor.reshape(1, -1)
    elif pixel_tensor.ndim == 1:
        pixel_tensor = pixel_tensor.unsqueeze(0)
    pixel_tensor = pixel_tensor.clamp(0.0, 1.0)
    pixel_tensor.requires_grad_(include_backprop)

    fc1_weight = model.fc1.weight.detach().clone().to(DEVICE).requires_grad_(include_backprop)
    fc1_bias = model.fc1.bias.detach().clone().to(DEVICE).requires_grad_(include_backprop)
    fc2_weight = model.fc2.weight.detach().clone().to(DEVICE).requires_grad_(include_backprop)
    fc2_bias = model.fc2.bias.detach().clone().to(DEVICE).requires_grad_(include_backprop)
    fc3_weight = model.fc3.weight.detach().clone().to(DEVICE).requires_grad_(include_backprop)
    fc3_bias = model.fc3.bias.detach().clone().to(DEVICE).requires_grad_(include_backprop)

    scaled_fc1_weight = fc1_weight * layer_scales[0]
    scaled_fc1_bias = fc1_bias * layer_scales[0]
    scaled_fc2_weight = fc2_weight * layer_scales[1]
    scaled_fc2_bias = fc2_bias * layer_scales[1]
    scaled_fc3_weight = fc3_weight * layer_scales[2]
    scaled_fc3_bias = fc3_bias + torch.tensor(bias_offsets, device=DEVICE)

    input_layer = pixel_tensor.view(1, -1)
    pre_hidden_1 = F.linear(input_layer, scaled_fc1_weight, scaled_fc1_bias)
    hidden_1 = F.relu(pre_hidden_1)
    pre_hidden_2 = F.linear(hidden_1, scaled_fc2_weight, scaled_fc2_bias)
    hidden_2 = F.relu(pre_hidden_2)
    logits = F.linear(hidden_2, scaled_fc3_weight, scaled_fc3_bias)
    probabilities = F.softmax(logits, dim=1)

    if include_backprop:
        input_layer.retain_grad()
        hidden_1.retain_grad()
        hidden_2.retain_grad()
        logits.retain_grad()

    predicted_digit = int(probabilities.argmax(dim=1).item())
    confidence = float(probabilities[0, predicted_digit].item())
    target_digit = predicted_digit if requested_target is None else requested_target

    if include_backprop:
        target = torch.tensor([target_digit], device=DEVICE)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        input_gradient = input_layer.grad.detach()
        hidden_1_gradient = hidden_1.grad.detach()
        hidden_2_gradient = hidden_2.grad.detach()
        logits_gradient = logits.grad.detach()
        backprop = {
            'loss': round(float(loss.item()), 6),
            'targetDigit': target_digit,
            'layerGradientNorms': [
                round(float(input_gradient.norm().item()), 6),
                round(float(hidden_1_gradient.norm().item()), 6),
                round(float(hidden_2_gradient.norm().item()), 6),
                round(float(logits_gradient.norm().item()), 6),
            ],
            'strongestWeightUpdates': {
                'input_to_hidden_1': _top_connections(fc1_weight.grad, limit=18),
                'hidden_1_to_hidden_2': _top_connections(fc2_weight.grad, limit=18),
                'hidden_2_to_output': _top_connections(fc3_weight.grad, limit=18),
            },
        }
    else:
        input_gradient = None
        hidden_1_gradient = None
        hidden_2_gradient = None
        logits_gradient = None
        backprop = None

    hidden_2_contribution = hidden_2[0] * scaled_fc3_weight[predicted_digit]
    top_contributor_indices = torch.topk(hidden_2_contribution.abs(), k=min(10, hidden_2_contribution.numel())).indices
    top_contributors = [
        {
            'index': int(index),
            'contribution': round(float(hidden_2_contribution[index].item()), 6),
            'activation': round(float(hidden_2[0, index].item()), 6),
            'weight': round(float(scaled_fc3_weight[predicted_digit, index].item()), 6),
        }
        for index in top_contributor_indices.detach().cpu().tolist()
    ]

    layer_payload = [
        _layer_summary('input', input_layer[0], input_gradient[0] if input_gradient is not None else None),
        _layer_summary('hidden_1', hidden_1[0], hidden_1_gradient[0] if hidden_1_gradient is not None else None),
        _layer_summary('hidden_2', hidden_2[0], hidden_2_gradient[0] if hidden_2_gradient is not None else None),
        _layer_summary('output', probabilities[0], logits_gradient[0] if logits_gradient is not None else None),
    ]

    layer_payload[0]['grid'] = _tensor_list(input_layer[0].reshape(28, 28))
    layer_payload[0]['gradientGrid'] = _tensor_list(input_gradient[0].reshape(28, 28)) if input_gradient is not None else None
    layer_payload[1]['preActivations'] = _tensor_list(pre_hidden_1[0])
    layer_payload[2]['preActivations'] = _tensor_list(pre_hidden_2[0])
    layer_payload[3]['logits'] = _tensor_list(logits[0])

    dynamic_edges = {
        'input_to_hidden_1': _top_active_connections(input_layer[0], scaled_fc1_weight),
        'hidden_1_to_hidden_2': _top_active_connections(hidden_1[0], scaled_fc2_weight),
        'hidden_2_to_output': _top_active_connections(hidden_2[0], scaled_fc3_weight),
    }

    prediction = {
        'digit': predicted_digit,
        'confidence': round(confidence, 6),
        'probabilities': _tensor_list(probabilities[0]),
    }

    return {
        'prediction': prediction,
        'layers': layer_payload,
        'dynamicEdges': dynamic_edges,
        'topContributors': top_contributors,
        'controls': {
            'layerScales': [round(scale, 4) for scale in layer_scales],
            'outputBiasOffsets': [round(offset, 4) for offset in bias_offsets],
            'targetDigit': target_digit,
        },
        'backprop': backprop,
        'weightStats': {
            'input_to_hidden_1': _stats(scaled_fc1_weight),
            'hidden_1_to_hidden_2': _stats(scaled_fc2_weight),
            'hidden_2_to_output': _stats(scaled_fc3_weight),
        },
    }


def train_on_example(pixels: list[float] | list[list[float]], label: int, learning_rate: float = 0.0005) -> dict[str, Any]:
    global _MODEL_CACHE

    model = get_model(force_reload=True)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    target = torch.tensor([int(max(0, min(9, label)))], dtype=torch.long, device=DEVICE)
    input_tensor = torch.tensor(pixels, dtype=torch.float32, device=DEVICE).reshape(1, 1, 28, 28).clamp(0.0, 1.0)
    logits = model(input_tensor)
    loss = F.cross_entropy(logits, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    torch.save(model.state_dict(), MODEL_PATH)
    _MODEL_CACHE = model.eval()

    analysis = analyze_digit(input_tensor.view(28, 28).detach().cpu().tolist(), include_backprop=True)
    analysis['trainingStep'] = {
        'label': int(target.item()),
        'loss': round(float(loss.item()), 6),
    }
    return analysis