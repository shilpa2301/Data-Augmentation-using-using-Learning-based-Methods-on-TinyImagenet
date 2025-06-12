# SaliencyMix
SaliencyMix is a Saliency Guided Data Augmentation Strategy for Better Regularization
The original code of SaliencyMix is adapted from ImageNet to TinyImagenet
The baseline random erase is written from scratch, the core function is adapted from official implementation of randomErase.
The evaluation metrics for Random Erase is adapted as per Saliency Mix.
The preprocessing is done from scratch using Normalizing transform with Imagenet parameter values.
Both the Models are run on Resnet50 for comparison.


### Requirements : Environment File provided, tested on CUDA 12.6  
```
conda env create -f environment.yml
```


#### SaliencyMix
-To train ResNet50 on TinyImageNet with SaliencyMix and traditional data augmentation:    
```
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

#### RandomErase
-To train ResNet50 on TinyImageNet with RandomErase:    
```
python SaliencyMix-ImageNet/baseline_random_erase/resnet.py > run_job_randomerase.log 2>&1
```

-To generate comparison plots:    
```
python compare_plots.py 
```


# Model Performance

## Accuracy Plot
![Combined Accuracy Plot](combined_accuracy_plot.png)

## Error Plot
![Combined Error Plot](combined_error_plot.png)

## Loss Plot
![Combined Loss Plot](combined_loss_plot.png)

