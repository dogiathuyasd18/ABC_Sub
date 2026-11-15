# ABC Head Pose Forecasting

Model: ordinary ResMLP (not acceleration-weighted).

## Inference interface

- Input: float32 [400, 4], quaternion order w,x,y,z.
- Output: float32 [20, 4], quaternion order w,x,y,z.
- Sampling frequency: 100 Hz.
- Forecast horizon: 20 samples.
- ONNX Runtime CPU inference.

## Installation

    pip install -r requirements.txt

## Run

    python run_inference.py --model_dir . --input sample.npy --output pred.npy

The input NPY file must contain a float32 array with shape [400, 4].
No network access is needed during inference.
# ABC_Sub
