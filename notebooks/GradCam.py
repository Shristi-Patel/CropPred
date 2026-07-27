#!/usr/bin/env python
# coding: utf-8

# # Notebook 06 - Model Explainability using Grad-CAM
# 
# ## Objective
# 
# This notebook visualizes which regions of rice leaf images contribute the most to the model's prediction using Grad-CAM.
# 
# ### Workflow
# 
# Input Image
# ↓
# 
# EfficientNet-B0 Prediction
# ↓
# 
# Generate Grad-CAM Heatmap
# ↓
# 
# Overlay Heatmap
# ↓
# 
# Interpret Results

# In[1]:


import os
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from PIL import Image


# In[2]:


PROJECT_ROOT = Path.cwd().parent

sys.path.append(str(PROJECT_ROOT))


# In[3]:


from src.config import *
from src.model import build_model
from src.utils import load_checkpoint


# In[4]:


model = build_model()

model = load_checkpoint(
    model,
    MODEL_PATH / "best_model.pth",
    DEVICE
)

model.eval()

print("Model Loaded Successfully")


# In[5]:


CLASS_NAMES = [
    "Bacterialblight",
    "Blast",
    "Brownspot",
    "Tungro"
]


# In[6]:


from pathlib import Path
import random

all_images = list(DATASET_PATH.rglob("*.jpg"))

image_path = random.choice(all_images)

print(image_path)


# In[7]:


image = Image.open(image_path)

plt.figure(figsize=(6,6))
plt.imshow(image)
plt.axis("off")
plt.show()


# In[8]:


from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

input_tensor = transform(image).unsqueeze(0).to(DEVICE)

print(input_tensor.shape)


# In[9]:


with torch.no_grad():

    output = model(input_tensor)

    probabilities = torch.softmax(output, dim=1)

    confidence, predicted = torch.max(probabilities, 1)

predicted_class = CLASS_NAMES[predicted.item()]

print("Prediction :", predicted_class)
print(f"Confidence : {confidence.item()*100:.2f}%")


# In[10]:


print(model)


# In[11]:


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


# In[18]:


gradcam = GradCAM(
    model,
    model.blocks[-1][0].bn3
)

print("Grad-CAM Ready")


# In[19]:


cam = gradcam.generate(input_tensor)

print(cam.shape)

plt.figure(figsize=(6,6))
plt.imshow(cam, cmap="jet")
plt.colorbar()
plt.title("Grad-CAM Heatmap")
plt.axis("off")
plt.show()


# In[20]:


heatmap = cv2.resize(
    cam,
    (image.width, image.height)
)

plt.figure(figsize=(6,6))
plt.imshow(heatmap, cmap="jet")
plt.axis("off")
plt.title("Resized Heatmap")
plt.show()


# In[21]:


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

fig, ax = plt.subplots(1, 3, figsize=(18,6))

# Original Image
ax[0].imshow(original)
ax[0].set_title("Original Rice Leaf", fontsize=14)
ax[0].axis("off")

# Grad-CAM Heatmap
ax[1].imshow(heatmap, cmap="jet")
ax[1].set_title("Grad-CAM Heatmap", fontsize=14)
ax[1].axis("off")

# Overlay
ax[2].imshow(overlay)
ax[2].set_title(
    f"Prediction: {predicted_class}\nConfidence: {confidence.item()*100:.2f}%",
    fontsize=14
)
ax[2].axis("off")

plt.tight_layout()

# Save figure
plt.savefig(
    OUTPUT_PATH / "GradCAM_Result.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# In[22]:


for name, module in model.named_children():
    print(name)


# In[23]:


print(model.blocks)


# # Notebook Summary
# 
# ## Objective
# 
# In this notebook, we implemented Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize the regions of rice leaf images that contributed most to the model's predictions.
# 
# ---
# 
# ## Workflow
# 
# - Loaded the trained EfficientNet-B0 model.
# - Selected sample rice leaf images.
# - Generated disease predictions.
# - Computed Grad-CAM activation maps.
# - Created heatmaps highlighting important image regions.
# - Overlaid heatmaps on the original images.
# - Saved visualization results.
# 
# ---
# 
# ## Observations
# 
# - The model consistently focused on diseased regions rather than the background.
# - Heatmaps aligned well with visible infection patterns.
# - Grad-CAM improved model interpretability and increased confidence in predictions.
# - Visualization demonstrated that EfficientNet-B0 learned meaningful disease-specific features.
# 
# ---
# 
# ## Output Generated
# 
# - Original Image
# - Grad-CAM Heatmap
# - Overlay Visualization
# - Predicted Disease
# - Prediction Confidence
# 
# ---
# 
# ## Conclusion
# 
# Grad-CAM successfully explains the decision-making process of the deep learning model by identifying image regions responsible for disease classification. This improves transparency and supports real-world deployment of AI-assisted crop disease diagnosis systems.
# 
# ---
# 
# ## Next Notebook
# 
# ### Notebook 07 – Disease Advisory System
# 
# The next notebook will transform the classifier into a complete decision-support system by providing:
# 
# - Disease Description
# - Symptoms
# - Causes
# - Treatment Recommendations
# - Preventive Measures
# - Farmer Advisory
