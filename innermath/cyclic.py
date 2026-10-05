"""Train a small MLP and probe C_n actions in its hidden activations (NumPy only)."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform

import numpy as np

from innermath.structures import exact_cyclic_orbits, probe_cyclic


@dataclass(frozen=True)
class ExperimentConfig:
    modulus: int = 11
    hidden: int = 48
    steps: int = 2000
    learning_rate: float = 0.01
    train_fraction: float = 0.8
    ridge: float = 1e-6
    seed: int = 0

    def validate(self) -> None:
        if self.modulus < 5 or self.hidden < 1 or self.steps < 1:
            raise ValueError("modulus >= 5, hidden >= 1 and steps >= 1 are required.")
        if not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction must lie strictly between 0 and 1.")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")
        if not np.isfinite(self.ridge) or self.ridge <= 0:
            raise ValueError("ridge must be finite and positive.")


def modular_data(modulus: int) -> tuple[np.ndarray, np.ndarray]:
    # Row b*n+a corresponds to (a,b); shifting a generates each orbit.
    b, a = np.indices((modulus, modulus))
    eye = np.eye(modulus)
    return np.concatenate((eye[a.ravel()], eye[b.ravel()]), axis=1), ((a+b) % modulus).ravel()


def forward(x: np.ndarray, weights: dict) -> tuple[np.ndarray, np.ndarray]:
    hidden = np.tanh(x @ weights["w1"] + weights["b1"])
    return hidden, hidden @ weights["w2"] + weights["b2"]


def loss_and_gradients(x: np.ndarray, y: np.ndarray, weights: dict) -> tuple[float, dict]:
    h, logits = forward(x, weights)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    loss = float(np.mean(np.log(exp.sum(axis=1)) - logits[np.arange(len(y)), y]))
    delta = probs.copy()
    delta[np.arange(len(y)), y] -= 1
    delta /= len(y)
    dh = (delta @ weights["w2"].T) * (1 - h*h)
    return loss, {"w1": x.T @ dh, "b1": dh.sum(axis=0),
                  "w2": h.T @ delta, "b2": delta.sum(axis=0)}


def run_experiment(config: ExperimentConfig, output: str | Path) -> dict:
    config.validate()
    rng = np.random.default_rng(config.seed)
    n, d = config.modulus, config.hidden
    x, y = modular_data(n)
    permutation = rng.permutation(len(y))
    split = max(1, min(len(y)-1, int(len(y)*config.train_fraction)))
    task_train, task_test = permutation[:split], permutation[split:]
    weights = {"w1": rng.normal(0, 1/np.sqrt(2*n), (2*n, d)), "b1": np.zeros(d),
               "w2": rng.normal(0, 1/np.sqrt(d), (d, n)), "b2": np.zeros(n)}
    initial_weights = {key: value.copy() for key, value in weights.items()}
    initial, initial_logits = forward(x, weights)
    moment = {key: np.zeros_like(value) for key, value in weights.items()}
    variance = {key: np.zeros_like(value) for key, value in weights.items()}
    history = []
    for step in range(1, config.steps+1):
        loss, gradients = loss_and_gradients(x[task_train], y[task_train], weights)
        for key, gradient in gradients.items():
            moment[key] = 0.9*moment[key] + 0.1*gradient
            variance[key] = 0.999*variance[key] + 0.001*gradient**2
            weights[key] -= config.learning_rate * (moment[key]/(1-0.9**step)) / (np.sqrt(variance[key]/(1-0.999**step)) + 1e-8)
        if step == 1 or step % 100 == 0 or step == config.steps:
            history.append({"step": step, "train_loss_before_update": loss})
    hidden, logits = forward(x, weights)
    orbit_ids = rng.permutation(n)
    orbit_split = max(2, min(n-1, int(n*0.7)))
    probe_train, probe_test = orbit_ids[:orbit_split], orbit_ids[orbit_split:]
    learned = hidden.reshape(n, n, d)
    initial = initial.reshape(n, n, d)
    shuffled = learned.copy()
    for b in range(n):
        shuffled[b] = shuffled[b, rng.permutation(n)]
    positive = exact_cyclic_orbits(n, n, config.seed)
    probes, generators = {}, {}
    for name, states in {"trained": learned, "untrained": initial,
                         "shuffled": shuffled, "exact_control": positive}.items():
        probes[name], generators[name] = probe_cyclic(states[probe_train], states[probe_test], config.ridge)
    report = {
        "config": asdict(config),
        "versions": {"python": platform.python_version(), "numpy": np.__version__},
        "task": {"train_accuracy": float(np.mean(logits[task_train].argmax(axis=1) == y[task_train])),
                 "test_accuracy": float(np.mean(logits[task_test].argmax(axis=1) == y[task_test])),
                 "untrained_test_accuracy": float(np.mean(initial_logits[task_test].argmax(axis=1) == y[task_test])),
                 "chance_accuracy": 1/n},
        "splits": {"task_train": task_train.tolist(), "task_test": task_test.tolist(),
                   "probe_train_orbits": probe_train.tolist(), "probe_test_orbits": probe_test.tolist()},
        "probes": probes,
        "training_history": history,
        "interpretation": "Exploratory diagnostics, not proof of an emergent group representation. Task and probe splits are distinct. DFT energy alone is not evidence of structure.",
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    np.savez_compressed(output/"artifacts.npz", **weights,
                        **{f"initial_{k}": v for k, v in initial_weights.items()},
                        inputs=x, labels=y, hidden=learned, untrained_hidden=initial,
                        shuffled_hidden=shuffled, exact_control=positive,
                        **{f"generator_{k}": v for k, v in generators.items()})
    lines = ["# Cyclic representation probe", "", f"Configuration: `{asdict(config)}`", "",
             f"Task accuracy: train {report['task']['train_accuracy']:.3f}, held-out {report['task']['test_accuracy']:.3f}.", "",
             "| States | Held-out equivariance error | Mean power error | Cycle error on states |",
             "|---|---:|---:|---:|"]
    for name, metrics in probes.items():
        lines.append(f"| {name} | {metrics['test_equivariance_error']:.6f} | {metrics['test_mean_power_error']:.6f} | {metrics['test_cycle_error']:.6f} |")
    lines.extend(["", report["interpretation"], "", "See docs/cyclic_representations.md for definitions and limitations.", ""])
    (output/"summary.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="InnerMath: probe cyclic group structure in an MLP's hidden states.")
    for name in ("modulus", "hidden", "steps", "seed"):
        parser.add_argument(f"--{name}", type=int, default=getattr(ExperimentConfig(), name))
    for name in ("learning_rate", "train_fraction", "ridge"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=float, default=getattr(ExperimentConfig(), name))
    parser.add_argument("--output-dir", default="outputs/cyclic/default")
    args = vars(parser.parse_args(argv))
    output = args.pop("output_dir")
    try:
        report = run_experiment(ExperimentConfig(**args), output)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Saved experiment to {output}; task test accuracy = {report['task']['test_accuracy']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
