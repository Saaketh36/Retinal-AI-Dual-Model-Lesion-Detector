# Model Weights Directory

This directory stores fine-tuned PyTorch checkpoint weights for the dual-model diagnostic engine:
- `best_efficientnet_b3.pth`: EfficientNet-B3 model state dictionary for 4-class multi-label retinal disease detection.
- `best_resnet50.pth`: ResNet-50 model state dictionary.

### How to Generate / Place Weights:
1. Run the training script:
   ```bash
   python train.py --epochs 25 --batch-size 16 --data-dir path/to/dataset
   ```
   Trained checkpoints are automatically saved to this `models/` directory upon reaching a new validation Macro-F1 score high.

2. **Fallback Mode**:
   If checkpoints are not present, `app.py` runs gracefully in **Demonstration Mode** using ImageNet pre-trained backbones, displaying a clear notice in the clinical dashboard.
