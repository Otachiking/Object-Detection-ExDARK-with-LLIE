# Configuration History

## [2026-05-18] - FINAL LOCK: Rolling back to the Undisputed Best Configuration (0.5528 mAP)
**Reason**: Lowering `mosaic` (to 0.75) and `mixup` (to 0.15) caused a disastrous drop in mAP (0.5416) and Recall (0.5203 -> 0.5054). 
Research Insight: ExDARK is a very small dataset (~7300 images). Lowering heavy augmentations like Mosaic caused the model to overfit and lose its generalization power. The model drastically missed objects (Recall drop), heavily penalizing classes like Cup, Chair, and Boat.
**Action**: We are locking in the configuration from the `[2026-05-17 21:27:38]` run. No more micro-tweaking augmentations downwards.

### FINAL BEST CONFIGURATION (Locked in base.yaml):
- **Model**: `yolo11n.pt`
- **Optimizer**: `AdamW`
- **Patience**: 15
- **Weight Decay**: 0.0005
- **LRF**: 0.01
- **Augmentasi - HSV_V**: 0.25 (Dark-domain optimized)
- **Augmentasi - Translate**: 0.15
- **Augmentasi - Copy-Paste**: 0.1 (Crucial for Motorbike/Boat)
- **Augmentasi - Mixup**: 0.2 (Optimal balance)
- **Augmentasi - Mosaic**: 1.0 (MANDATORY for small dataset scaling)

## [2026-05-18] - Reverting to "Best" Config with Mosaic & Mixup Adjustments
**Reason**: The previous refined configuration (mAP 54.78%) helped large objects (Bus, Car) but absolutely destroyed Motorbike (-4.6%), Boat (-3.4%), and Cat (-1.3%). We are reverting back to the settings that yielded 55.28% mAP, but altering `mosaic` and `mixup` to specifically address the large and low-visibility objects.

### Adjusted Configuration (Adopting Best + 2 Changes):
- **Augmentasi - HSV_V**: 0.25 (Reverted back to best; too much brightness variation hurts ExDARK's natural domain).
- **Augmentasi - Copy-Paste**: 0.1 (Reverted back to best; Motorbike and Boat heavily rely on this to learn).
- **Augmentasi - Mixup**: 0.15 (Decreased from 0.2/0.25! Mixup creates "ghostly" transparent objects. For dark objects like Cat and Boat, this makes them nearly invisible and ruins training. Lower is better here).
- **Augmentasi - Mosaic**: 0.75 (Decreased from 1.0! This means 25% of the time, the model sees a full, uncut image. This directly helps Bus and Car understand their spatial context without being chopped into 4 pieces).

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
