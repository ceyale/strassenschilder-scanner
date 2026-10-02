# Digit CNN

`mnist-12.onnx` is the ONNX Model Zoo MNIST convolutional digit model.

- Source: https://github.com/onnx/models/tree/main/validated/vision/classification/mnist
- Model card/license: MIT
- Input: float32 tensor `[1, 1, 28, 28]`, white digit on black background, values in `[0, 1]`
- Output: 10 class scores, one per digit `0` through `9`

The model is trained on handwritten MNIST digits, not specifically on printed traffic-sign digits.
