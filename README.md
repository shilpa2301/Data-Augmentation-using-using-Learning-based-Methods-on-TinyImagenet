# SaliencyMix
SaliencyMix: A Saliency Guided Data Augmentation Strategy for Better Regularization
The original code of SaliencyMix is adapted from ImageNet to TinyImagenet
The baseline random erase is written from scratch, the main function is adapted from official implementation of randomErase.
Both teh Models are run on Resnet50 for comparison 


### Requirements  
- Python3
- PyTorch (> 1.0)
- torchvision (> 0.2)
- NumPy
- OpenCV-contrib-python (4.2.0.32)


#### TinyImageNet
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

-To train ResNet50 on TinyImageNet with RandomErase:    
```
python SaliencyMix-ImageNet/baseline_random_erase/resnet.py > run_job_randomerase.log 2>&1
```

-To generate comparison plots:    
```
python compare_plots.py 
```


### Test Examples using ImageNet Pretrained models

- Trained models can be downloaded from [here](https://drive.google.com/drive/folders/1vnJHtgzcBInuPZVkwQxQ5A5SE_i_-EON?usp=sharing)

- ResNet-50
```
python test.py \
--net_type resnet \
--dataset imagenet \
--batch_size 64 \
--depth 50 \
--pretrained /runs/ResNet50_SaliencyMix_21.26/model_best.pth.tar
```
- ResNet-101
```
python test.py \
--net_type resnet \
--dataset imagenet \
--batch_size 64 \
--depth 101 \
--pretrained /runs/ResNet101_SaliencyMix_20.09/model_best.pth.tar
```


# Model Performance

## Accuracy Plot
![Combined Accuracy Plot](combined_accuracy_plot.png)

## Error Plot
![Combined Error Plot](combined_error_plot.png)

## Loss Plot
![Combined Loss Plot](combined_loss_plot.png)

