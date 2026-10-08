"""Read-only smoke tests for packaged robot-learning and surrogate assets.

No training, optimization, GUI windows, or output files are produced.
"""
from pathlib import Path
import numpy as np
import joblib
import pybullet as p
import tensorflow as tf
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "cases"

def test_tflite_inference():
    models = sorted(CASES.glob("**/*.tflite"))
    assert models, "No packaged TFLite models found"
    for path in models:
        interpreter = tf.lite.Interpreter(model_path=str(path))
        interpreter.allocate_tensors()
        ins = interpreter.get_input_details()
        assert len(ins) == 1, f"Expected one model input: {path}"
        desc = ins[0]
        shape = [max(1, int(n)) for n in desc["shape"]]
        assert len(shape) >= 2, f"Unexpected input rank: {path}: {shape}"
        value = np.zeros(shape, dtype=desc["dtype"])
        if np.issubdtype(value.dtype, np.integer):
            value.fill(int(desc["quantization"][1]))
        interpreter.set_tensor(desc["index"], value)
        interpreter.invoke()
        for output in interpreter.get_output_details():
            y = interpreter.get_tensor(output["index"])
            assert y.size > 0 and np.isfinite(y).all(), f"Invalid inference output: {path}"
        print(f"INFERENCE OK: {path.relative_to(ROOT)}")
    print(f"PASS: {len(models)} TFLite model inferences")

def test_scalers():
    paths = sorted(CASES.glob("**/*.joblib"))
    assert paths, "No packaged joblib objects found"
    for path in paths:
        model = joblib.load(path)
        assert hasattr(model, "transform") or hasattr(model, "predict"), f"Unexpected joblib asset: {path}"
        if hasattr(model, "n_features_in_") and hasattr(model, "transform"):
            data = np.zeros((1, int(model.n_features_in_)), dtype=np.float64)
            out = model.transform(data)
            assert np.isfinite(out).all(), f"Invalid scaler output: {path}"
        print(f"JOBLIB OK: {path.relative_to(ROOT)}")
    print(f"PASS: {len(paths)} packaged joblib objects")

def test_ppo_checkpoint():
    checkpoint = CASES / "gripper/RL/models/ywk5xiqs/final_model.zip"
    assert checkpoint.is_file(), f"Missing {checkpoint}"
    model = PPO.load(str(checkpoint), device="cpu")
    assert model.policy is not None
    observation = np.zeros(model.observation_space.shape, dtype=np.float32)
    action, _ = model.predict(observation, deterministic=True)
    assert np.isfinite(np.asarray(action)).all(), "Invalid PPO prediction"
    print("PASS: Gripper PPO checkpoint loading and deterministic policy prediction")

def test_pybullet_geometry():
    meshes = [
        CASES / "gripper/link/top.STL",
        CASES / "gripper/link/module.STL",
        CASES / "gripper/link/end.STL",
        CASES / "tendon_driven/link/link1.STL",
        CASES / "tendon_driven/link/link23.STL",
        CASES / "tendon_driven/link/link4.STL",
    ]
    connection = p.connect(p.DIRECT)
    assert connection >= 0, "Could not connect to PyBullet DIRECT"
    try:
        p.setGravity(0, 0, -9.81)
        for mesh in meshes:
            assert mesh.is_file(), f"Missing mesh: {mesh}"
            visual = p.createVisualShape(p.GEOM_MESH, fileName=str(mesh), meshScale=[0.001]*3)
            assert visual >= 0, f"Could not load visual mesh: {mesh}"
            obj = p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual)
            assert obj >= 0, f"Could not create mesh body: {mesh}"
            print(f"MESH OK: {mesh.relative_to(ROOT)}")
        for _ in range(5):
            p.stepSimulation()
        print("PASS: PyBullet DIRECT geometry loading and stepping")
    finally:
        p.disconnect(connection)

if __name__ == "__main__":
    test_tflite_inference()
    test_scalers()
    test_ppo_checkpoint()
    test_pybullet_geometry()
    print("PASS: research smoke tests")
