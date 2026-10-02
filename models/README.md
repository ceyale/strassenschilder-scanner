# Digit CNN

`mnist-12.onnx` is the ONNX Model Zoo MNIST convolutional digit model.

- Source: https://github.com/onnx/models/tree/main/validated/vision/classification/mnist
- Model card/license: MIT
- Input: float32 tensor `[1, 1, 28, 28]`, white digit on black background, values in `[0, 1]`
- Output: 10 class scores, one per digit `0` through `9`

The model is trained on handwritten MNIST digits, not specifically on printed traffic-sign digits.

`emnist-alphanumeric.onnx` is an alphanumeric EMNIST model from the Hugging Face repository [`hermitkk/alphabet-classifier`](https://huggingface.co/hermitkk/alphabet-classifier).

- File: `outputs/exports/alphanumeric_model.onnx`
- License: MIT
- Classes: digits `0`–`9`, uppercase letters `A`–`Z`, and blank
- Input: float32 tensor `[1, 1, 96, 96]`, grayscale normalized to `[-1, 1]`
- The model card reports 91.42% validation accuracy on handwritten EMNIST byclass data. It is not trained specifically on printed road signs.
