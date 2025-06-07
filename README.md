# When to Stop Federated Learning: Zero-Shot Generation of Synthetic Validation Data with Generative AI for Early Stopping

This is an official implementation of the following paper:
> Anonymous Author(s).
**When to Stop Federated Learning: Zero-Shot Generation of Synthetic Validation Data with Generative AI for Early Stopping**  
_Submitted_.

## Federated Learning Methods
This paper considers the following federated learning techniques:
- **FedAvg**: [Communication-Efficient Learning of Deep Networks from Decentralized Data](http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf)
- **FedDyn**: [Federated Learning Based on Dynamic Regularization](https://openreview.net/pdf?id=B7v4QMR6Z9w)
- **FedSAM**: [Generalized Federated Learning via Sharpness Aware Minimization](https://proceedings.mlr.press/v162/qu22a/qu22a.pdf)
- **FedGamma**: [Fedgamma: Federated learning with global sharpness-aware minimization](https://ieeexplore.ieee.org/abstract/document/10269141)
- **FedSpeed**: [FedSpeed: Larger Local Interval, Less Communication Round, and Higher Generalization Accuracy](https://openreview.net/pdf?id=bZjxxYURKT)
- **FedSMOO**: [Dynamic Regularized Sharpness Aware Minimization in Federated Learning: Approaching Global Consistency and Smooth Landscape](https://proceedings.mlr.press/v202/sun23h.html)

## Docker Image
`docker pull pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel`

Additionally, request to RoentGen authors to get the pretrained weight and please install the required packages as below
```pip install transformers datasets timm diffusers huggingface_hub medmnist```

## Dataset
- Chest X-ray Classification Dataset ([Chestx-ray8: Hospital-scale chest x-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases.](https://arxiv.org/abs/1705.02315))

## Experiments

Run `bash shell/gen.sh` to generated sufficient synthetic data

To run the 'Impact of Synthetic Validation-based Early Stopping' experiment: `bash shell/run1.sh`

To run the 'Impact of non-IID Degree: `bash shell/run2.sh`

To run the 'Ablation Study' experiment: `bash shell/run3.sh`

## References
This repository draws inspiration from:
- https://github.com/woodenchild95/FL-Simulator