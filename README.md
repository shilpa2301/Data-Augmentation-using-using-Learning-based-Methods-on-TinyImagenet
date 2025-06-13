# A Comparative Analysis of Classical and Learning-Based Data Augmentation for Image Classification

## Summary

SaliencyMix is a Saliency Guided Data Augmentation Strategy for Better Regularization.  
The original code of SaliencyMix is adapted from ImageNet to Tiny ImageNet.  
The baseline Random Erase is written from scratch; the core function is adapted from the official implementation of Random Erase.  
The evaluation metrics for Random Erase are adapted as per SaliencyMix.  
Preprocessing for Random Erase is implemented from scratch using a Normalizing transform with ImageNet parameter values.  
The preprocessing for SaliencyMix includes RandomCrop, RandomHorizontalFlip, Jitter, Lighting, and Normalization (provided in the official implementation of SaliencyMix).  
Both models are run on ResNet50 for comparison and trained from scratch without initializing with pretrained ResNet weights.  
SaliencyMix uses He initialization, whereas Random Erase uses Xavier initialization.

---

## Requirements

Environment file provided. Tested on **CUDA 12.6**.

```bash
conda env create -f environment.yml
```

---

## Training

### SaliencyMix

To train ResNet50 on Tiny ImageNet with SaliencyMix and traditional data augmentation:

```bash
python SaliencyMix-ImageNet/train.py \
  --net_type resnet \
  --dataset imagenet \
  --batch_size 256 \
  --lr 0.1 \
  --depth 50 \
  --epochs 300 \
  --expname ResNet50 \
  -j 40 \
  --beta 1.0 \
  --salmix_prob 1.0 \
  --no-verbose > run_job_saliency.log 2>&1
```

### Random Erase

To train ResNet50 on Tiny ImageNet with Random Erase:

```bash
python SaliencyMix-ImageNet/baseline_random_erase/resnet.py > run_job_randomerase.log 2>&1
```

---

## Generate Comparison Plots

```bash
python compare_plots.py
```

---

# Model Performance

## Accuracy Plot

![Combined Accuracy Plot](combined_accuracy_plot.png)

## Error Plot

![Combined Error Plot](combined_error_plot.png)

## Loss Plot

![Combined Loss Plot](combined_loss_plot.png)

---

## Acknowledgments

- [**SaliencyMix**](https://github.com/afm-shahab-uddin/SaliencyMix)  
- [**Random Erasing**](https://github.com/zhunzhong07/Random-Erasing)
