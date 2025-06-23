# Utility/gradcam_helper.py

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import sys

class GradCAM:
    """Implements Grad-CAM for a PyTorch model."""
    def __init__(self, model, target_layer_name):
        self.model = model.eval() # Ensure model is in evaluation mode
        self.target_layer = None
        self.activation = None
        self.gradient = None
        self.hook_handles = [] # List to store hook handles

        try:
            self.target_layer = self.model.get_submodule(target_layer_name)
            # print(f"Grad-CAM: Successfully found target layer '{target_layer_name}'.") # Debug
        except AttributeError:
            raise ValueError(f"Target layer '{target_layer_name}' not found in the model.")
        except Exception as e:
             raise RuntimeError(f"Error finding target layer '{target_layer_name}': {e}") from e


    def _save_activation(self, module, input, output):
        """Hook to save the output (activation) of the target layer."""
        self.activation = output.clone().detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to save the gradient w.r.t. the output of the target layer."""
        self.gradient = grad_output[0].clone().detach()


    def __call__(self, x, target_category=None):
        """
        Compute Grad-CAM for a single input image.
        Args:
            x (torch.Tensor): Input image tensor (shape: [1, C, H, W]). Must be on the correct device.
            target_category (int or None): The target class index for backpropagation.
                                            For binary classification with 1 output neuron,
                                            backpropagate the single logit (None).
        Returns:
            torch.Tensor: Grad-CAM map (shape: [1, H_cam, W_cam]) on the same device as input x.
                          Returns None if processing fails.
        """
        if x.ndim != 4 or x.shape[0] != 1:
            print(f"Grad-CAM: Input tensor must have shape [1, C, H, W], but got {x.shape}. Skipping.", file=sys.stderr)
            return None

        self.remove_hooks()

        try:
            self.hook_handles.append(self.target_layer.register_forward_hook(self._save_activation))
            self.hook_handles.append(self.target_layer.register_full_backward_hook(self._save_gradient))
        except Exception as e:
             print(f"Grad-CAM: Error registering hooks on target layer: {e}. Skipping Grad-CAM.", file=sys.stderr)
             self.remove_hooks()
             return None


        self.model.zero_grad()
        self.activation = None
        self.gradient = None

        try:
             if next(self.model.parameters()).device != x.device:
                  self.model.to(x.device)
             output = self.model(x)

             if isinstance(output, tuple):
                 print("Warning: Model returned a tuple output during Grad-CAM forward pass. Using only the first element.", file=sys.stderr)
                 output = output[0]

        except Exception as e:
             print(f"Grad-CAM: Error during model forward pass: {e}. Skipping Grad-CAM.", file=sys.stderr)
             self.remove_hooks()
             return None

        try:
            if output.numel() == 1:
                 target_score = output.squeeze()
            elif output.ndim == 2 and output.shape[1] > 1 and target_category is not None:
                 target_score = output[:, target_category].squeeze()
            elif output.ndim == 2 and output.shape[1] > 1 and target_category is None:
                 predicted_class = output.argmax(dim=1).item()
                 target_score = output[:, predicted_class].squeeze()
                 print(f"Warning: Multi-class output detected but target_category is None. Using predicted class {predicted_class} for Grad-CAM.", file=sys.stderr)
            else:
                print(f"Grad-CAM: Unexpected model output shape {output.shape}. Skipping Grad-CAM.", file=sys.stderr)
                self.remove_hooks()
                return None

        except Exception as e:
             print(f"Grad-CAM: Error selecting target score for backprop: {e}. Skipping Grad-CAM.", file=sys.stderr)
             self.remove_hooks()
             return None


        if self.activation is None:
             print(f"Grad-CAM: Activation was not captured. Make sure target layer '{self.target_layer}' is correctly named and part of the model's forward pass.", file=sys.stderr)
             self.remove_hooks()
             return None

        try:
             target_score.backward(gradient=torch.ones_like(target_score), retain_graph=True)
        except RuntimeError as e:
             print(f"\nGrad-CAM: RuntimeError during backward pass: {e}", file=sys.stderr)
             print("This might be due to inplace operations or issues with graph retention.", file=sys.stderr)
             self.remove_hooks()
             return None
        except Exception as e:
             print(f"Grad-CAM: Error during backward pass: {e}. Skipping Grad-CAM.", file=sys.stderr)
             self.remove_hooks()
             return None


        if self.gradient is None:
             print("Grad-CAM: Gradient was not captured by the hook.", file=sys.stderr)
             self.remove_hooks()
             return None

        try:
             if self.gradient.shape[0] != 1 or self.activation.shape[0] != 1:
                  print(f"Grad-CAM: Expected batch size 1 for gradient ({self.gradient.shape[0]}) or activation ({self.activation.shape[0]}). Skipping.", file=sys.stderr)
                  self.remove_hooks()
                  return None

             pooled_gradients = torch.mean(self.gradient, dim=[2, 3], keepdim=True)

             if pooled_gradients.shape[0] != self.activation.shape[0] or pooled_gradients.shape[1] != self.activation.shape[1]:
                  print(f"Grad-CAM: Shape mismatch between pooled gradients {pooled_gradients.shape} and activation {self.activation.shape}. Cannot compute CAM. Skipping.", file=sys.stderr)
                  self.remove_hooks()
                  return None

             weighted_activation = self.activation * pooled_gradients
             cam = torch.sum(weighted_activation, dim=1, keepdim=True)
             cam = F.relu(cam)

        except Exception as e:
             print(f"Grad-CAM: Error computing CAM map: {e}. Skipping Grad-CAM.", file=sys.stderr)
             self.remove_hooks()
             return None

        self.remove_hooks()
        return cam

    def remove_hooks(self):
        """Removes registered hooks to clean up."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []


def visualize_cam(original_img_pil, cam_map, title="Grad-CAM"):
    """
    Visualizes the Grad-CAM map overlaid on the original image.
    Args:
        original_img_pil (PIL.Image): The original image.
        cam_map (np.ndarray): The Grad-CAM map (HxW numpy array).
        title (str): Title for the plot.
    """
    if cam_map is None or cam_map.size == 0:
         print("Warning: CAM map is empty or None. Cannot visualize.", file=sys.stderr)
         return

    try:
        cam_map = np.nan_to_num(cam_map, nan=0.0, posinf=1e5, neginf=-1e5)
        cam_map = cam_map.astype(np.float32)
        cam_map_resized = cv2.resize(cam_map, (original_img_pil.width, original_img_pil.height), interpolation=cv2.INTER_LINEAR)
    except Exception as e:
        print(f"Error resizing CAM map: {e}. Skipping visualization.", file=sys.stderr)
        return

    cam_min, cam_max = cam_map_resized.min(), cam_map_resized.max()
    if cam_max - cam_min < 1e-8:
         cam_map_normalized = np.zeros_like(cam_map_resized)
    else:
        cam_map_normalized = (cam_map_resized - cam_min) / (cam_max - cam_min)
        cam_map_normalized = np.clip(cam_map_normalized, 0, 1)

    original_img_np = np.array(original_img_pil)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam_map_normalized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    alpha = 0.5
    original_img_float = original_img_np.astype(np.float32) / 255.0
    heatmap_float = heatmap.astype(np.float32) / 255.0

    if original_img_float.ndim == 2:
        original_img_float = np.stack([original_img_float]*3, axis=-1)
    elif original_img_float.ndim == 3 and original_img_float.shape[2] == 1:
         original_img_float = np.repeat(original_img_float, 3, axis=2)

    overlaid_img_float = cv2.addWeighted(original_img_float, 1 - alpha, heatmap_float, alpha, 0)
    overlaid_img_np = np.uint8(255 * overlaid_img_float)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(original_img_np)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(overlaid_img_np)
    plt.title(title)
    plt.axis('off')

    plt.tight_layout()
    plt.show()
