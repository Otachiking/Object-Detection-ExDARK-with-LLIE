# Configuration History

## [2026-05-18] - Refined Augmentation for Per-Class Balancing
**Reason**: While the overall mAP increased from 54.91% to 55.28%, large/rigid objects (Bus, Car) and low-visibility objects (Cat) dropped in accuracy. Meanwhile, small objects (Motorbike, Bicycle) improved significantly. 
We tweaked `hsv_v`, `mixup`, and `copy_paste` to balance this out.

### Refined Configuration:
- **Augmentasi - HSV_V**: 0.3 (Increased from 0.25 to help large objects handle brightness variance better, without going back to the extreme 0.4)
- **Augmentasi - Mixup**: 0.25 (Increased from 0.2 to improve generalization for low-visibility classes like Cat and Table)
- **Augmentasi - Copy-Paste**: 0.05 (Decreased from 0.1 because pasting large objects like Bus/Car into random contexts hurts their spatial understanding)

## [2026-05-18] - Switching Back to "Bad" Hyperparameters (They were actually better!)
**Reason**: User confirmed the recent hyperparameter tweaks (previously marked as bad) actually yielded better results, returning to them, specifically ensuring `yolo11n.pt` is used.

### New Active Configuration (from previous "Bad" config):
- **Model**: `yolo11n.pt`
- **Optimizer**: `AdamW`
- **Patience**: 15
- **Weight Decay**: 0.0005
- **LRF (Final learning rate fraction)**: 0.01
- **Augmentasi - HSV_V**: 0.25
- **Augmentasi - Translate**: 0.15
- **Augmentasi - Copy-Paste**: 0.1
- **Device**: 0

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
