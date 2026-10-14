# Digit CNN

`mnist-12.onnx` is the ONNX Model Zoo MNIST convolutional digit model.

- Source: https://github.com/onnx/models/tree/main/validated/vision/classification/mnist
- Model card/license: MIT
- Input: float32 tensor `[1, 1, 28, 28]`, white digit on black background, values in `[0, 1]`
- Output: 10 class scores, one per digit `0` through `9`

The model is trained on handwritten MNIST digits, not specifically on printed traffic-sign digits, and is no longer used for speed-limit recognition. The traffic-sign CNN's GTSRB class directly provides the speed value.

`emnist-alphanumeric.onnx` is an alphanumeric EMNIST model from the Hugging Face repository [`hermitkk/alphabet-classifier`](https://huggingface.co/hermitkk/alphabet-classifier).

- File: `outputs/exports/alphanumeric_model.onnx`
- License: MIT
- Classes: digits `0`–`9`, uppercase letters `A`–`Z`, and blank
- Input: float32 tensor `[1, 1, 96, 96]`, grayscale normalized to `[-1, 1]`
- The model card reports 91.42% validation accuracy on handwritten EMNIST byclass data. It is not trained specifically on printed road signs.

`gtsrb-sign-cnn.onnx` is the 99,019-parameter TrafficSignNet checkpoint from [`tanmayai23/gtsrb-traffic-sign-recognition`](https://github.com/tanmayai23/gtsrb-traffic-sign-recognition), exported to ONNX. Its model has 43 outputs in the official GTSRB class order; class 14 is Stop. The repository reports 97.98% validation and 96.82% official test accuracy and is MIT-licensed.

- Input: normalized RGB tensor `[1, 3, 32, 32]`.
- Normalization uses mean `[0.41359345, 0.38223032, 0.39460091]` and standard deviation `[0.27057118, 0.26105214, 0.26897097]` after CLAHE-style luminance equalization.
- Red color/shape checks propose boxes; the CNN class and, for speed signs, matching multi-threshold digit CNN output are both required before a box is shown.

`sign-binary-32.onnx` is the 4,202-parameter depthwise-separable binary CNN from the private Kaggle CPU run. It accepts float RGB `[1, 3, 32, 32]` in `[0, 1]`; logits are ordered `not_traffic_sign`, `traffic_sign`. The best epoch was 43 with 99.50% accuracy on a balanced validation split. Negatives include red-heavy ordinary objects, GTSDB background crops, CIFAR-10 scenes, and synthetic colored blobs/noise; non-traffic warning signs are excluded. It runs after heuristic proposals and before the GTSRB classifier. The frontend switch defaults to on.

`learnable-box-head.json` contains a 149-parameter `18-6-5` score/regression head trained on GTSDB candidate boxes. HSV/color components generate proposals; the small head scores and refines them. It runs in JavaScript without ONNX Runtime. Its Kaggle validation F1 at IoU 0.5 was about 0.50, so it remains independently switchable; the switch defaults to on and the original heuristic runs alongside it.
