# Dataset

## Face Mask Detection Dataset
- **Author:** vijaykumar1799
- **Source:** https://www.kaggle.com/datasets/vijaykumar1799/face-mask-detection
- **License:** See Kaggle dataset page

## Structure
3 classes, 2,994 images each (perfectly balanced):
- `with_mask/`
- `without_mask/`
- `mask_weared_incorrect/`

Images are pre-cropped face photos in PNG format. No bounding box annotations.

## Download
```python
import kagglehub
path = kagglehub.dataset_download("vijaykumar1799/face-mask-detection")
```
Requires a free Kaggle account.
