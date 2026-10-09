# Satellite Water Segmentation with Deep Learning

> Pixel-level water mapping from multispectral satellite and geospatial data using PyTorch.

This project explores deep-learning approaches for identifying water pixels in satellite-derived data. It includes two segmentation experiments: a custom U-Net baseline and a DeepLabV3+ model with a ResNet34 encoder.

## Project highlights

- **Task:** Binary semantic segmentation of water vs. non-water pixels.
- **Framework:** PyTorch and Segmentation Models PyTorch (SMP).
- **Architectures explored:** Custom U-Net and DeepLabV3+ with a ResNet34 encoder.
- **Input features:** Selected multispectral and geospatial channels, including NIR, SWIR, elevation, and land-cover information.
- **Preprocessing:** Per-band min–max scaling based on training data.
- **Patch-based learning:** Satellite images are divided into 16 × 16 patches.
- **Training strategy:** Image-level train/validation/test splitting to keep patches from the same image in the same split.
- **Loss:** Combined Dice Loss and Binary Cross-Entropy with logits.
- **Evaluation:** F1/Dice, Intersection over Union (IoU), and confusion matrix.
- **Prediction:** Generate a probability map and threshold it into a binary water mask.

## Model experiments and reported results

The following values are taken from the outputs recorded in the supplied notebooks. They are results from those notebook runs, not a controlled head-to-head benchmark.

| Experiment | Input channels in notebook | Test F1 / Dice | Test IoU |
|---|---:|---:|---:|
| Custom U-Net | 6 | 0.9345 | 0.8770 |
| DeepLabV3+ with ResNet34 | 11 | 0.9599 | 0.9229 |

**Important comparison note:** These runs do not use the same number of input channels, and the DeepLabV3+ notebook converts the water-occurrence target to a binary mask using a threshold of 75. The scores should therefore not be interpreted as a perfectly controlled architecture-only comparison. For a fair comparison, use identical image splits, target construction, input features, preprocessing, evaluation threshold, and training protocol.

## Data and target preparation

The notebooks organize data as image-level pixel tables and reconstruct each image before dividing it into patches. The DeepLabV3+ notebook converts `Water occ prob` to a binary target using:

- Water: `Water occ prob >= 75`
- Non-water: `Water occ prob < 75`

Confirm that this threshold matches the definition and units of the source dataset before using the resulting mask for scientific or operational conclusions.


The U-Net notebook uses six input channels:

- NIR
- SWIR1
- SWIR2
- Merit DEM
- Copernicus DEM
- ESA WorldCover map

The DeepLabV3+ notebook uses 11 input channels according to its recorded `FEATURE_COLS`/model configuration. Check the notebook's final feature list before reproducing the experiment, as comments in the notebook refer to different channel counts in different places.

## Methodology

1. Reconstruct satellite images from the pixel table.
2. Split data by image into training, validation, and test sets.
3. Derive normalization ranges from the training set to reduce data leakage.
4. Divide images into 16 × 16 patches.
5. Apply synchronized geometric augmentation to images and masks during training.
6. Train the segmentation model using a combined Dice + BCE-with-logits loss.
7. Monitor validation loss, F1/Dice, and IoU; save the best checkpoint.
8. Evaluate the selected checkpoint on the held-out test set.
9. Visualize the source image/features, ground-truth mask, predicted probability map, and thresholded mask.

## Loss function

The combined objective is:

Dice Loss encourages overlap between predicted and ground-truth water regions, while BCE-with-logits provides pixel-wise supervision. The model outputs logits during training; sigmoid is applied during inference to obtain probabilities.

## Evaluation metrics

- **F1 / Dice:** overlap between predicted and ground-truth water pixels.
- **IoU:** intersection divided by union of predicted and ground-truth water regions.
- **Confusion matrix:** counts true negatives, false positives, false negatives, and true positives.

Metric values depend on the threshold used to convert probabilities into binary predictions. Choose this threshold using validation data, then keep it fixed for final test evaluation.

## Environment

The notebooks use Python with libraries including:

- PyTorch
- `segmentation-models-pytorch`
- NumPy and pandas
- scikit-learn
- Albumentations
- OpenCV
- Matplotlib

Install the main packages in your environment as needed:

```bash
pip install torch segmentation-models-pytorch numpy pandas scikit-learn albumentations opencv-python matplotlib
```

The notebooks were developed in a Kaggle-style environment. Paths, available hardware, and package versions may need adjustment for another environment.

## Project status

This is an experimental computer-vision project focused on satellite water segmentation. Reported metrics reflect the supplied notebook runs and should be independently reproduced before being treated as benchmark results.

