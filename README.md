# CIFAR-10 Image Classification

Image classification on CIFAR-10 using two CNN architectures:

- Baseline CNN
- Improved CNN (BatchNorm + Dropout + Label Smoothing)

## How to Run

Train both models:
`python train.py`

Train only baseline:
`python train.py baseline`

Train only improved:
`python train.py improved`

Evaluate:
`python evaluate.py`

## Training Loss Curves

Baseline Model

![Baseline Training Loss](outputs/plots/baseline_loss.png)

Improved Model

![Improved Training Loss](outputs/plots/improved_loss.png)

## Train / Validation Split

Random 90:10 split using PyTorch random_split().

Training: 45,000 images  
Validation: 5,000 images  

## Metrics Reported

- Accuracy
- Precision
- Recall
- F1-score
- Macro Average
- Weighted Average
- Confusion Matrix

Metric files:
outputs/metrics/baseline_metrics.txt  
outputs/metrics/improved_metrics.txt  

## Results

### Baseline CNN

Accuracy: 0.7543

Macro Avg:
Precision: 0.76  
Recall: 0.75  
F1-score: 0.75  

Weighted Avg:
Precision: 0.76  
Recall: 0.75  
F1-score: 0.75  

Class Highlights:
Class 1 F1-score: 0.87  
Class 6 F1-score: 0.80  
Class 3 F1-score: 0.58  

### Improved CNN

Accuracy: 0.8004

Macro Avg:
Precision: 0.80  
Recall: 0.80  
F1-score: 0.80  

Weighted Avg:
Precision: 0.80  
Recall: 0.80  
F1-score: 0.80  

Class Highlights:
Class 1 F1-score: 0.90  
Class 8 F1-score: 0.90  
Class 9 F1-score: 0.89  

## Best Result

Improved CNN  
Test Accuracy: 80.04%

## Additions

- Label Smoothing during training
- Batch Normalization + Dropout in improved model
- Automatic plot generation with labeled axes
- Human-readable metric output files
- Option to run both or individual models
- CPU-only execution
- Grad-CAM visualization module included
