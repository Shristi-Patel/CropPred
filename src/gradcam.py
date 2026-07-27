import cv2
import numpy as np
import torch


class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.target_layer.register_forward_hook(
            self.forward_hook
        )

        self.target_layer.register_full_backward_hook(
            self.backward_hook
        )

    def forward_hook(self, module, input, output):
        self.activations = output.detach()

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):

        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()

        score = output[:, class_idx]

        score.backward()

        gradients = self.gradients[0]

        activations = self.activations[0]

        weights = gradients.mean(dim=(1, 2))

        cam = (weights[:, None, None] * activations).sum(dim=0)

        cam = torch.relu(cam)

        cam = cam.cpu().numpy()

        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam


def create_overlay(image, cam):

    heatmap = cv2.resize(
        cam,
        (image.width, image.height)
    )

    original = np.array(image)

    heatmap_uint8 = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        original,
        0.6,
        heatmap_color,
        0.4,
        0
    )

    return heatmap, overlay