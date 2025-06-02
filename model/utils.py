from tkinter import Image
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    return model



def generate_confidence_graph(frame_ids, real_confidences, fake_confidences, output_path):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(6, 3))

    # Convert inputs to NumPy arrays
    frame_ids = np.array(frame_ids)
    real_confidences = np.array(real_confidences)
    fake_confidences = np.array(fake_confidences)

    # Print for debug
    print("Frame IDs:", frame_ids)
    print("Real Confidences:", real_confidences)
    print("Fake Confidences:", fake_confidences)

    # Plot bars
    width = 0.4
    frame_shift = 0.2

    plt.bar(frame_ids - frame_shift, real_confidences, width=width, color='green', label='Real Confidence')
    plt.bar(frame_ids + frame_shift, fake_confidences, width=width, color='red', label='Fake Confidence')

    # Highlight max fake confidence
    if len(fake_confidences) > 0:
        max_fake_index = np.argmax(fake_confidences)
        max_fake_frame = frame_ids[max_fake_index]
        max_fake_value = fake_confidences[max_fake_index]

        plt.plot(max_fake_frame + frame_shift, max_fake_value, 'ro', markersize=6)
        plt.text(max_fake_frame + frame_shift + 0.2, max_fake_value,
                 f'Max Fake\n{max_fake_value:.1f}%', color='red', fontsize=8)

    plt.title('Prediction Confidence per Frame')
    plt.xlabel('Frame')
    plt.ylabel('Confidence (%)')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# utils.py (or a new helper file)
# utils.py
import torch
import torch.nn.functional as F
import numpy as np
import cv2

def get_gradcam_heatmap(model, input_tensor):
    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations["value"] = output.detach()

    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0].detach()

    # Register hooks on last conv layer
    target_layer = model.model.layer4[1].conv2
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    model.eval()
    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()
    model.zero_grad()
    output[0, pred_class].backward()

    activations = activations["value"][0]
    gradients = gradients["value"][0]

    weights = gradients.mean(dim=(1, 2))
    grad_cam = torch.zeros(activations.shape[1:], dtype=torch.float32)

    for i, w in enumerate(weights):
        grad_cam += w * activations[i]

    grad_cam = F.relu(grad_cam)
    grad_cam -= grad_cam.min()
    grad_cam /= grad_cam.max()

    heatmap = grad_cam.cpu().numpy()
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)

    forward_handle.remove()
    backward_handle.remove()

    return heatmap



