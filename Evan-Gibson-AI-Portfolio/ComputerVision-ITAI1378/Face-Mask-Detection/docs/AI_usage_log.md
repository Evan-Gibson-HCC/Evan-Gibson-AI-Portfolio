# AI Usage Log — Face Mask Detection Project

**Course:** ITAI 1378 — Computer Vision  
**Student:** Evan Gibson  
**AI Tool Used:** Claude (Anthropic) — claude.ai

---

## Summary
Claude was used extensively as a learning and development aid throughout this project. All code was reviewed, understood, and tested by the student before submission.

---

## Usage Breakdown

### Project Planning
- Evaluated dataset options (vijaykumar1799 vs. andrewmvd) — Claude identified that the andrewmvd dataset requires bounding box annotations incompatible with a classification approach
- Discussed trade-offs between Faster R-CNN, YOLOv8, and ResNet classification; selected ResNet-18 classification as appropriate for dataset format and Colab constraints
- Clarified what "cropped" images mean in the context of classification vs. detection pipelines

### Code Generation
- Initial Colab notebook scaffolding (data loading, training loop, evaluation cells)
- `TransformSubset` wrapper class for applying different transforms to val/test splits

### Debugging & Iteration
- Identified `kagglehub` as an alternative to `kaggle.json` upload
- Diagnosed incorrect `DATA_DIR` path issue after dataset download
- Fixed inference cell to preserve upload logic while adding threshold override
- Corrected `plt.title()` to use `predicted_class` instead of raw `CLASS_NAMES[pred]`

### Analysis & Interpretation
- Interpreted training curves (confirmed no overfitting — train/val loss tracked together)
- Analyzed three real-world failure cases with the student's own photos:
  - Nose-exposed mask wear classified as `with_mask`
  - Mic boom classified as `with_mask` (wide-frame sensitivity)
  - Tighter crop resolved framing issue but not nose-exposure issue
- Explained why failures are dataset limitations, not training problems

---

## What I Learned From AI Assistance
- Transfer learning rationale: why freezing early layers preserves general features
- Why raw accuracy is a weak metric for imbalanced or multi-class detection problems
- The distinction between classification datasets (folder-sorted crops) and detection datasets (bounding box annotations)
- How confidence thresholds can patch known model biases as a post-processing step
- What out-of-distribution failure looks like in practice vs. on a held-out test set

---

## Code Attribution
- ~40% written/modified directly by student (cell edits, path fixes, threshold tuning, real-world testing)
- ~60% AI-generated scaffold, reviewed and tested by student
