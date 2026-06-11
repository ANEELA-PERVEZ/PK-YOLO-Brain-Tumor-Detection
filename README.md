# PK-YOLO-Brain-Tumor-Detection
PK-YOLO based brain tumor detection in multiplanar MRI slices (axial, sagittal, coronal) using SparK pretraining and Focaler-IoU loss. Includes GradCAM explainability visualizations.
# PK-YOLO: Brain Tumor Detection in Multiplanar MRI Slices

## Overview
This project implements PK-YOLO for detecting brain tumors across 
axial, sagittal, and coronal MRI planes using pretrained knowledge 
and Focaler-IoU loss optimization.

## Results
| MRI Plane | mAP@50 |
|-----------|--------|
| Axial     | 0.930  |
| Sagittal  | 0.600  |
| Coronal   | 0.631  |

## Dataset
RSNA-MICCAI Brain Tumor Challenge 2021

## GradCAM Explainability
Gradient-weighted Class Activation Maps generated for clinical interpretability.

## Author
Aneela Pervez — GIFT University, Gujranwala
