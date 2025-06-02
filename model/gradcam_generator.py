# import torch
# import torch.nn.functional as F
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt

# def generate_gradcam(model, img_tensor, target_layer='layer4'):
#     # Register hooks to capture gradients and feature maps
#     gradients = []
#     feature_maps = []

#     def save_gradients(module, grad_input, grad_output):
#         gradients.append(grad_output[0])  # We need grad_output from backward

#     def save_feature_maps(module, input, output):
#         feature_maps.append(output)

#     # Access the layer4 directly from the ResNet model in DeepfakeDetector
#     target_layer = getattr(model.model, target_layer)  # `model.model` accesses the ResNet part of your model

#     # Register the forward and backward hooks
#     target_layer.register_forward_hook(save_feature_maps)
#     target_layer.register_backward_hook(save_gradients)

#     # Forward pass through the model
#     model.eval()
#     output = model(img_tensor)

#     # Backward pass to calculate gradients for the target class (assuming binary classification)
#     output_idx = 0  # Assuming we want the output of class 0 (fake or real)
#     model.zero_grad()
#     output[0, output_idx].backward()

#     # Get the gradients and feature maps
#     grads = gradients[0]
#     fmap = feature_maps[0]

#     # Pool the gradients across the channels (global average pooling)
#     pooled_grads = torch.mean(grads, dim=[0, 2, 3])

#     # Weight the feature maps with the pooled gradients
#     for i in range(fmap.shape[1]):
#         fmap[:, i, :, :] *= pooled_grads[i]

#     # Generate the Grad-CAM heatmap (average across all channels)
#     heatmap = torch.mean(fmap, dim=1).squeeze()
#     heatmap = F.relu(heatmap)

#     # Normalize the heatmap
#     heatmap = heatmap - torch.min(heatmap)
#     heatmap = heatmap / torch.max(heatmap)

#     # Convert the heatmap to a numpy array after detaching it from the computation graph
#     heatmap = heatmap.detach().cpu().numpy()

#     return heatmap
