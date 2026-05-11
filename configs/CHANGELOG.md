# Configuration History

## [2026-05-11] - Reverting Hyperparameters
**Reason**: The recent hyperparameter tweaks resulted in worse model performance. Reverting to the previous stable baseline configuration.

### Reverted (Bad) Configuration (Recorded before reverting):
- **Model**: `yolo11n.pt`
- **Optimizer**: `AdamW`
- **Patience**: 15
- **Weight Decay**: 0.0005
- **LRF (Final learning rate fraction)**: 0.01
- **Augmentasi - HSV_V**: 0.25
- **Augmentasi - Translate**: 0.15
- **Augmentasi - Copy-Paste**: 0.1
- **Device**: 0

### Restored (Stable) Configuration:
- **Model**: `yolo12n.pt`
- **Optimizer**: `auto` (Ultralytics default - SGD)
- **Patience**: 10
- **Weight Decay**: 0.001
- **LRF (Final learning rate fraction)**: 0.05
- **Augmentasi - HSV_V**: 0.4
- **Augmentasi - Translate**: 0.2
- **Augmentasi - Copy-Paste**: 0.0
- **Device**: auto
