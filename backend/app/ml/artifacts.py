"""Save/load trained models with the scikit-learn version recorded.

A pickled sklearn model may fail to load, or silently behave differently, under
another sklearn version, so load_bundle refuses a mismatch and asks to retrain.
Security: joblib/pickle can run code while loading. Only load files this
project created itself; never load a model file from an untrusted source.
"""
from pathlib import Path

import joblib
import sklearn


def save_bundle(path: Path, **objects) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"sklearn_version": sklearn.__version__, **objects}, path)


def load_bundle(path: Path, train_command: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"No model at {path}. Run: {train_command}")
    bundle = joblib.load(path)
    if bundle.get("sklearn_version") != sklearn.__version__:
        raise RuntimeError(f"{path.name} was trained with scikit-learn {bundle.get('sklearn_version')} "
                           f"but {sklearn.__version__} is installed. Retrain: {train_command}")
    return bundle
