#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort

CONTEXT = 400
HORIZON = 20


def normalize(q):
    q = np.asarray(q, dtype=np.float32)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.maximum(n, 1e-8)


def quat_mul(a, b):
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)

    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=-1).astype(np.float32)


def quat_inv(q):
    result = np.array(q, dtype=np.float32, copy=True)
    result[..., :3] *= -1
    return result


def quat_to_rotvec(q):
    q = normalize(q)

    xyz = q[..., :3]
    w = np.clip(q[..., 3], -1.0, 1.0)

    # Match shortest-arc quaternion representation.
    sign = np.where(w < 0.0, -1.0, 1.0)
    xyz = xyz * sign[..., None]
    w = w * sign

    sin_half = np.linalg.norm(xyz, axis=-1)
    angle = 2.0 * np.arctan2(sin_half, w)

    scale = np.where(
        sin_half > 1e-8,
        angle / np.maximum(sin_half, 1e-8),
        2.0,
    )

    return (xyz * scale[..., None]).astype(np.float32)


def rotvec_to_quat(v):
    v = np.asarray(v, dtype=np.float32)
    angle = np.linalg.norm(v, axis=-1, keepdims=True)
    half = 0.5 * angle

    scale = np.where(
        angle > 1e-8,
        np.sin(half) / np.maximum(angle, 1e-8),
        0.5,
    )

    xyz = v * scale
    w = np.cos(half)

    return normalize(np.concatenate([xyz, w], axis=-1))


def load_model(model_dir: str) -> ort.InferenceSession:
    path = Path(model_dir) / "model_fp32.onnx"

    if not path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {path}")

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1

    session = ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    inputs = session.get_inputs()
    outputs = session.get_outputs()

    if len(inputs) != 1 or len(outputs) != 1:
        raise RuntimeError("Unexpected ONNX input/output count")

    return session


def predict(model, context: np.ndarray) -> np.ndarray:
    context = np.asarray(context, dtype=np.float32)

    if context.shape != (CONTEXT, 4):
        raise ValueError(
            f"Expected context shape (400, 4), got {context.shape}"
        )

    if not np.isfinite(context).all():
        raise ValueError("Context contains NaN or Inf")

    # Challenge convention: w,x,y,z.
    # Model convention: x,y,z,w.
    q = context[:, [1, 2, 3, 0]]
    q = normalize(q)

    # Consecutive relative rotations, matching training preprocessing.
    q_relative = quat_mul(
        quat_inv(q[:-1]),
        q[1:],
    )

    delta_rotvec = quat_to_rotvec(q_relative)
    model_input = delta_rotvec[None, :, :].astype(np.float32)

    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name

    future_rotvec = model.run(
        [output_name],
        {input_name: model_input},
    )[0]

    future_rotvec = np.asarray(future_rotvec, dtype=np.float32)

    if future_rotvec.shape != (1, HORIZON, 3):
        raise RuntimeError(
            f"Unexpected ONNX output shape: {future_rotvec.shape}"
        )

    if not np.isfinite(future_rotvec).all():
        raise RuntimeError("ONNX output contains NaN or Inf")

    # Decode predicted body-frame rotations relative to final context pose.
    delta_q = rotvec_to_quat(future_rotvec[0])
    future_q = quat_mul(q[-1], delta_q)
    future_q = normalize(future_q)

    # Back to challenge convention: w,x,y,z.
    result = future_q[:, [3, 0, 1, 2]].astype(np.float32)

    if result.shape != (HORIZON, 4):
        raise RuntimeError(f"Invalid prediction shape: {result.shape}")

    if not np.isfinite(result).all():
        raise RuntimeError("Prediction contains NaN or Inf")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=".")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    model = load_model(args.model_dir)

    data = np.load(args.input)

    if isinstance(data, np.lib.npyio.NpzFile):
        key = "context" if "context" in data.files else data.files[0]
        context = data[key]
        data.close()
    else:
        context = data

    result = predict(model, context)
    np.save(args.output, result)

    print("Prediction saved:", args.output)
    print("Shape:", result.shape)
    print("Dtype:", result.dtype)
    print("Quaternion norms:", np.linalg.norm(result, axis=-1))


if __name__ == "__main__":
    main()
