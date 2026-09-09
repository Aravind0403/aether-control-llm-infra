from pathlib import Path
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
import json
import time


class CheckpointMetadata(BaseModel):
    checkpoint_name: str
    step: int
    validation_accuracy: float
    format_score: float
    harmonic_f1_score: float
    timestamp: float
    path: str


class CheckpointManifest(BaseModel):
    checkpoints: Dict[str, CheckpointMetadata] = {}


class ParetoCheckpointer:
    """Manages Decoupled Pareto Checkpoint Classing (RFC-002).
    
    Maintains:
    - checkpoint_best_accuracy.pt (Highest held-out mathematical correctness)
    - checkpoint_best_format.pt (Highest XML <think> structural compliance)
    - checkpoint_balanced_pareto.pt (Highest harmonic mean F1 of Acc + Format)
    - checkpoint_latest.pt (Last executed step)
    """

    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.checkpoint_dir / "manifest.json"

        self.best_accuracy = 0.0
        self.best_format = 0.0
        self.best_pareto_f1 = 0.0
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> CheckpointManifest:
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r") as f:
                    data = json.load(f)
                    manifest = CheckpointManifest(**data)
                    for k, meta in manifest.checkpoints.items():
                        if k == "best_accuracy":
                            self.best_accuracy = meta.validation_accuracy
                        elif k == "best_format":
                            self.best_format = meta.format_score
                        elif k == "balanced_pareto":
                            self.best_pareto_f1 = meta.harmonic_f1_score
                    return manifest
            except Exception:
                pass
        return CheckpointManifest()

    def _save_manifest(self):
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest.model_dump(), f, indent=2)

    @staticmethod
    def compute_harmonic_f1(acc: float, fmt: float) -> float:
        """Calculates harmonic mean F1 between accuracy (0-1) and format (0-1)."""
        if acc + fmt <= 1e-8:
            return 0.0
        return float(2.0 * (acc * fmt) / (acc + fmt))

    def save_step(
        self,
        step: int,
        model_state: Any,
        validation_accuracy: float,
        format_score: float,
    ) -> List[str]:
        """Evaluates metrics and persists checkpoints into their respective Pareto classes."""
        saved_classes = []
        f1_score = self.compute_harmonic_f1(validation_accuracy, format_score)
        now = time.time()

        # 1. Checkpoint Latest
        latest_file = self.checkpoint_dir / "checkpoint_latest.pt"
        self._write_state(latest_file, model_state)
        self.manifest.checkpoints["latest"] = CheckpointMetadata(
            checkpoint_name="checkpoint_latest.pt",
            step=step,
            validation_accuracy=validation_accuracy,
            format_score=format_score,
            harmonic_f1_score=f1_score,
            timestamp=now,
            path=str(latest_file),
        )
        saved_classes.append("latest")

        # 2. Best Accuracy
        if validation_accuracy > self.best_accuracy:
            self.best_accuracy = validation_accuracy
            acc_file = self.checkpoint_dir / "checkpoint_best_accuracy.pt"
            self._write_state(acc_file, model_state)
            self.manifest.checkpoints["best_accuracy"] = CheckpointMetadata(
                checkpoint_name="checkpoint_best_accuracy.pt",
                step=step,
                validation_accuracy=validation_accuracy,
                format_score=format_score,
                harmonic_f1_score=f1_score,
                timestamp=now,
                path=str(acc_file),
            )
            saved_classes.append("best_accuracy")

        # 3. Best Format
        if format_score > self.best_format:
            self.best_format = format_score
            fmt_file = self.checkpoint_dir / "checkpoint_best_format.pt"
            self._write_state(fmt_file, model_state)
            self.manifest.checkpoints["best_format"] = CheckpointMetadata(
                checkpoint_name="checkpoint_best_format.pt",
                step=step,
                validation_accuracy=validation_accuracy,
                format_score=format_score,
                harmonic_f1_score=f1_score,
                timestamp=now,
                path=str(fmt_file),
            )
            saved_classes.append("best_format")

        # 4. Balanced Pareto
        if f1_score > self.best_pareto_f1:
            self.best_pareto_f1 = f1_score
            pareto_file = self.checkpoint_dir / "checkpoint_balanced_pareto.pt"
            self._write_state(pareto_file, model_state)
            self.manifest.checkpoints["balanced_pareto"] = CheckpointMetadata(
                checkpoint_name="checkpoint_balanced_pareto.pt",
                step=step,
                validation_accuracy=validation_accuracy,
                format_score=format_score,
                harmonic_f1_score=f1_score,
                timestamp=now,
                path=str(pareto_file),
            )
            saved_classes.append("balanced_pareto")

        self._save_manifest()
        return saved_classes

    def _write_state(self, filepath: Path, state: Any):
        """Simulates or executes saving state to file."""
        if isinstance(state, dict):
            with open(filepath.with_suffix(".json"), "w") as f:
                json.dump(state, f, indent=2)
        else:
            with open(filepath, "w") as f:
                f.write(str(state))

    def get_summary_table(self) -> List[Dict[str, Any]]:
        """Returns rows for inspection table."""
        rows = []
        for name, meta in self.manifest.checkpoints.items():
            rows.append({
                "class": name,
                "filename": meta.checkpoint_name,
                "step": meta.step,
                "accuracy": f"{meta.validation_accuracy * 100:.1f}%",
                "format": f"{meta.format_score:.2f}",
                "pareto_f1": f"{meta.harmonic_f1_score:.3f}",
            })
        return rows
