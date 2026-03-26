# Distilled MNISTNet

A neural network trained on MNIST with **defensive distillation** (teacher and student models).

## Architecture

`MNISTModel` is a CNN with:
- 4 convolutional layers (32 → 64 filters)
- 2 max-pooling layers
- 3 fully-connected layers (1024 → 200 → 200 → 10)
- Dropout (0.5)

Input: `1×28×28` grayscale image — Output: 10-class logits.

## Usage with Torch Hub

### Load the student (distilled) model

```python
import torch

model = torch.hub.load("maurapintor/distilled_mnist", "mnist_model", weights="student")
model.eval()
```

### Load the teacher model

```python
model = torch.hub.load("maurapintor/distilled_mnist", "mnist_model", weights="teacher")
model.eval()
```

### Load the untrained model

```python
model = torch.hub.load("maurapintor/distilled_mnist", "mnist_model")
model.eval()
```

### Run inference

```python
import torch

x = torch.randn(1, 1, 28, 28)  # replace with your preprocessed image
with torch.no_grad():
    logits = model(x)
pred = logits.argmax(dim=1).item()
print("Predicted class:", pred)
```

## Pretrained Weights

| Model | Checkpoint |
|-------|-----------|
| Student (distilled) | `mnist_distilled.pt` |
| Teacher | `mnist_teacher.pt` |

Weights are downloaded automatically by `torch.hub`.

