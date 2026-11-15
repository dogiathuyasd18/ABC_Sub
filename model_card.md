# Model Card: Ordinary ResMLP

## Model overview

A residual multilayer perceptron for head-pose forecasting.
This is the ordinary ResMLP model, not the acceleration-weighted variant.

## Architecture

- Model: ResMLP
- Hidden dimension: 64
- Residual blocks: 3
- Dropout: 0.1
- Trainable parameters: 105,916
- Context length: 400 samples
- Prediction horizon: 20 samples
- Sampling rate: 100 Hz
- Input features: 399 consecutive relative rotation vectors
- Output features: 20 future rotation vectors

## Training

- Dataset: head_pose_full_v3.h5
- Random seed: 42
- Batch size: 32
- Learning rate: 0.001
- Gradient clipping: 1.0
- Validation fraction: 0.15
- Test fraction: 0.15
- Mixed precision: disabled
- Best epoch: 26
- Best recorded validation error: 4.001457 degrees

The reported validation value is a local validation result, not an official
challenge test score.

## Quaternion conventions

The training model uses x,y,z,w quaternion order.
The inference wrapper accepts and returns w,x,y,z as required by the challenge.

## Inference pipeline

1. Normalize the input context quaternions.
2. Convert w,x,y,z to x,y,z,w.
3. Compute consecutive relative rotations.
4. Convert relative rotations to rotation vectors.
5. Run the ONNX model.
6. Decode predicted rotation vectors relative to the final context pose.
7. Normalize and convert predictions to w,x,y,z.

## Deployment

- ONNX model: model_fp32.onnx
- Runtime: ONNX Runtime CPU
- Streaming state: stateless; each call uses the supplied context.
- Quantization: none.
- Pruning: none.
- Distillation: none.

## Known limitations

Performance may degrade for motion patterns or users not represented
in the training distribution. The reported validation result does not
guarantee the official hidden-test score.
