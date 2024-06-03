# Installation Guide

This document provides step-by-step instructions for installing CUDA Toolkit, PyTorch, OpenFace, and openSMILE. Additionally, you will learn how to specify executor paths and configuration files within `app.py`.

## Prerequisites

Ensure your system meets the following requirements before you begin the installation process:

- NVIDIA GPU (CUDA-compatible)
- At least 8 GB of RAM (16 GB recommended)
- At least 10 GB of free disk space
- Operating System: Windows, Linux, or macOS

## Step 1: Installing CUDA Toolkit

The CUDA Toolkit enables GPU acceleration for intensive compute tasks. To install the CUDA Toolkit:

1. Visit the [CUDA Toolkit download page](https://developer.nvidia.com/cuda-downloads).
2. Select your OS, architecture, CUDA version, and the desired installer type (e.g., exe for Windows, runfile for Linux).
3. Download and run the installer.
4. Follow the on-screen instructions to complete the installation.

## Step 2: Installing PyTorch

PyTorch is a machine learning library that supports CUDA acceleration. Install PyTorch by running, make sure cuda is availabel:

```bash
# For Windows/Linux/MacOS
pip install torch torchvision torchaudio
```
## Step 3: Installing OpenFace
## Step 4: Install OpenSmile
## Step 5: specify the executor paths and configuration files in app.py Edit the file as follows:

```bash
# For Windows/Linux/MacOS
openface_path = "E:\Anaconda\envs\OpenFace\openface\FeatureExtraction.exe"
opensmile_path = "../opensmile/bin/SMILExtract" 
opensmile_config = "../opensmile/config/emobase/emobase2010.conf"
```