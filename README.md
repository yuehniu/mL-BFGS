## mL-BFGS codebase
This doc provide necessary guides for training models (CNNs, Transformers) with mL-BFGS optimizer.

Paper can be found at: https://openreview.net/pdf?id=9jnsPp8DP3

### Quick Start

---

Run `run-SLIM.sh` to quickly train a model with mL-BFGS optimizer. 
For running args, please refer to `run-SLIM.sh` for further information.

**ResNet-18 on CIFAR-100** (1 GPU needed)

```bash
bash run-SLIM.sh 1 127.0.0.1 11113 resnet18 cifar100 0.1 0.0002 0.9 150 4 0.999 0.9 50
```

**ResNet-50 on ImageNet** (4 GPUs needed)

```bash
bash run-SLIM.sh 4 127.0.0.1 11113 resnet50 imagenet 0.1 0.0002 0.9 100 "0,1,2,3" 0.999 0.9 50
```

**DeiT on CIFAR-100** (1 GPU needed)

```bash
bash run-SLIM.sh 1 127.0.0.1 11113 deit cifar100 0.1 0.0005 0.9 50 1 0.999 0.9 50
```

Running parameters are explained in `run-SLIM.sh`.

### Package Dependency

---

Below is my running environment:

python=3.8  
torch=1.9  
numpy=1.17  
tensorboard=2.6

### Further Details

---

To train a model, the code follows a standard machine learning training flow. 
If using mL-BFGS optimizer, we only need to call the `slimblock.py` to define the optimizer.
The rest of training is the same as the standard training with SGD. 

Please go to `./opt/slimblock.py` for implementation details.

While `./opt/slimblock.py` is a block-wise optimization, we also provide `./opt/slim.py`, 
a full-model Hessian estimation version. 

### Report Issues

---

If you have any issues/questions, you can either create an issue in the repo, 
or directly reach me via `yueniu@usc.edu`.