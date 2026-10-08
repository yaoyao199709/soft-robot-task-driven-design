"""Verify environment imports and load packaged models without training."""
from pathlib import Path
import importlib
import sys

ROOT = Path(__file__).resolve().parents[1]
modules = ('numpy', 'scipy', 'pandas', 'matplotlib', 'pybullet',
           'tensorflow', 'sklearn', 'joblib', 'cma', 'gym',
           'gymnasium', 'shimmy', 'stable_baselines3', 'wandb')
for name in modules:
    module = importlib.import_module(name)
    print(f'IMPORT OK: {name} {getattr(module, "__version__", "")}')
import tensorflow as tf
for model in ROOT.glob('cases/**/*.tflite'):
    interpreter = tf.lite.Interpreter(model_path=str(model))
    interpreter.allocate_tensors()
    print(f'TFLITE OK: {model.relative_to(ROOT)}')
print('PASS: dependency imports and TFLite loading')
