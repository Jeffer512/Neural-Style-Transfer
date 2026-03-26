# Neural Style Transfer (PyTorch & Keras)

This repository contains two implementations of Neural Style Transfer (NST) based on the paper **"A Neural Algorithm of Artistic Style"** ([Gatys et al.](https://arxiv.org/abs/1508.06576)). The project includes separate notebooks for **PyTorch** and **Keras (TensorFlow)** to demonstrate the framework-specific nuances of the algorithm.

## Technical Summary

Both implementations share the following core configurations:
* **Architecture:** VGG19 (Convolutional layers only).
* **Pooling:** Following the paper's suggestion, all `MaxPooling` layers were replaced with `AveragePooling` to achieve smoother gradient flow and superior stylization.
* **Weights:** Style and content weights were kept consistent across both frameworks (e.g., `style_weight = 1000`).
* **Optimizer:** Both implementations utilize the **Adam optimizer**, instead of the L-BFGS optimizer used in the original paper.

### Implementation Differences

| Feature | PyTorch Implementation | Keras Implementation |
| :--- | :--- | :--- |
| **Preprocessing** | Images are scaled to the `[0, 1]` range. | Uses `preprocess_input` (BGR conversion and ImageNet mean subtraction). |
| **Constraints** | Uses `clamp(0, 1)` within the optimization loop. | Unconstrained optimization (no clamping). |
| **Learning Rate** | Lower learning rate(e.g., 0.01 - 0.05). | Significantly higher learning rate(e.g., 1.0 - 10.0). |

## Observations

* **Layer Manipulation:** Swapping internal layers (e.g., MaxPool to AvgPool) is a straightforward list iteration in PyTorch's `nn.Sequential`. In contrast, the Keras Functional API requires manually rebuilding the computational graph to re-link layer outputs to their preceding inputs.
* **Activation and Loss Disparity:** Even when tested with identical configurations and `[0, 1]` scaling, the Keras implementation produced higher internal activations. This resulted in substantially higher content and style loss values compared to the PyTorch implementation.
* **Numerical Range & Learning Rate:** Because the Keras implementation operates on a much larger numerical range (due to the lack of `[0, 1]` scaling), it requires a higher learning rate to effectively minimize the loss.
* **Pooling Layers:** Replacing `MaxPooling` with `AveragePooling` consistently yielded better-defined artistic textures and more cohesive structural warping, confirming the findings in the Gatys paper.

## Dependencies
* `torch`, `torchvision`, `tensorflow`, `numpy`, `matplotlib`, `Pillow`.
  
## References
Gatys, L. A., Ecker, A. S., & Bethge, M. (2015). [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576). *arXiv preprint arXiv:1508.06576*.
