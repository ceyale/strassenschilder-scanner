# German traffic-sign training pipeline

This private Kaggle kernel trains a compact two-stage, sparse MoE pipeline:

1. **YOLO26n** finds traffic signs in full GTSDB road scenes. GTSDB scenes with no annotations are retained as negative images.
2. A MobileNetV3-Small gate routes each detected sign crop to one of five fine-grained experts: speed, prohibition, priority, warning, or direction. Experts classify the 43 GTSRB sign classes. The gate activates one expert per crop.

The Kaggle inputs are the GTSDB Pascal-VOC dataset and GTSRB. GPU training is enabled in `kernel-metadata.json`. Training defaults to 80 detector epochs and 12 classifier epochs; set `DETECT_EPOCHS` and `CLASS_EPOCHS` to change them. Outputs include YOLO `.pt`, gate/expert `.pt` and `.onnx` files, class mapping, and a metrics/limitations report under `/kaggle/working`.

The model is a prototype, not a safety-certified vehicle controller. GTSRB contains cropped signs, so it trains classification experts but does not provide scene context. GTSDB provides only a small set of full scenes. Traffic-light state, town-name OCR, and additional sign classes need dedicated labeled data and are not claimed by this run. Negative GTSDB scenes teach the detector when no sign is present; there is deliberately no `no_sign` object class.
