# import torch
# import torch.nn.functional as F
# import pickle
# from torchvision import models, transforms
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt

# # Step 1: Load the models
# # Load the .pth.tar model
# pth_tar_path = "model_best.pth.tar"  # Replace with your .pth.tar file path
# checkpoint = torch.load(pth_tar_path)
# model_pth = checkpoint["model"]  # Assuming the model is stored under the key "model"
# model_pth.eval()

# # Load the .pkl model
# # pkl_path = "path_to_pkl_model.pkl"  # Replace with your .pkl file path
# # with open(pkl_path, "rb") as f:
# #     model_pkl = pickle.load(f)
# # model_pkl.eval()  # Ensure the model is in evaluation mode

# # Step 2: Load and preprocess the input image
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# image_path = "grad_cam_images/val_77.JPEG"  # Replace with your image path
# image = Image.open(image_path).convert("RGB")
# input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# # Step 3: Define Grad-CAM functions
# def generate_gradcam(model, input_tensor, target_layer):
#     gradients = []
#     activations = []

#     def backward_hook(module, grad_input, grad_output):
#         gradients.append(grad_output[0])

#     def forward_hook(module, input, output):
#         activations.append(output)

#     # Register hooks
#     target_layer.register_forward_hook(forward_hook)
#     target_layer.register_backward_hook(backward_hook)

#     # Forward pass
#     output = model(input_tensor)
#     target_class = output.argmax(dim=1).item()  # Predicted class
#     model.zero_grad()
#     output[:, target_class].backward()  # Backward pass for the target class

#     # Compute Grad-CAM
#     grads = gradients[0].detach().cpu().numpy()
#     acts = activations[0].detach().cpu().numpy()
#     weights = np.mean(grads, axis=(2, 3))  # Global average pooling of gradients
#     heatmap = np.sum(weights[:, :, np.newaxis, np.newaxis] * acts, axis=1)
#     heatmap = np.maximum(heatmap, 0)  # Apply ReLU
#     heatmap = heatmap[0]  # Remove batch dimension
#     heatmap /= np.max(heatmap)  # Normalize to [0, 1]
#     return heatmap

# # Step 4: Generate Grad-CAMs for both models
# # For the .pth.tar model
# target_layer_pth = model_pth.layer4[-1]  # Replace with the correct layer for your model
# gradcam_pth = generate_gradcam(model_pth, input_tensor, target_layer_pth)

# # For the .pkl model
# target_layer_pkl = model_pkl.layer4[-1]  # Replace with the correct layer for your model
# gradcam_pkl = generate_gradcam(model_pkl, input_tensor, target_layer_pkl)

# # Step 5: Overlay the Grad-CAM heatmaps on the input image
# def overlay_heatmap(image, heatmap):
#     heatmap = np.uint8(255 * heatmap)  # Scale to [0, 255]
#     heatmap = Image.fromarray(heatmap).resize(image.size, Image.BILINEAR)
#     heatmap = np.array(heatmap)
#     plt.imshow(image)
#     plt.imshow(heatmap, cmap="jet", alpha=0.5)  # Overlay heatmap with transparency
#     plt.axis("off")

# plt.figure(figsize=(12, 6))

# # Grad-CAM for .pth.tar model
# plt.subplot(1, 2, 1)
# plt.title("Grad-CAM (.pth.tar model)")
# overlay_heatmap(image, gradcam_pth)

# # Grad-CAM for .pkl model
# plt.subplot(1, 2, 2)
# plt.title("Grad-CAM (.pkl model)")
# overlay_heatmap(image, gradcam_pkl)

# plt.show()


import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision import models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load the model from .pth.tar
def load_model(pth_tar_path):
    checkpoint = torch.load(pth_tar_path)
    model = checkpoint['model']  # Adjust this line based on how the model is stored
    model.eval()  # Set the model to evaluation mode
    return model

# Preprocessing function
def preprocess_image(image_path):
    # Same preprocessing as in the original code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    return input_tensor

# Grad-CAM implementation
def generate_gradcam(model, input_tensor, target_layer):
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Register hooks
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    # Forward pass
    output = model(input_tensor)
    target_class = output.argmax(dim=1).item()  # Predicted class
    model.zero_grad()
    output[:, target_class].backward()  # Backward pass for the target class

    # Compute Grad-CAM
    grads = gradients[0].detach().cpu().numpy()
    acts = activations[0].detach().cpu().numpy()
    weights = np.mean(grads, axis=(2, 3))  # Global average pooling of gradients
    heatmap = np.sum(weights[:, :, np.newaxis, np.newaxis] * acts, axis=1)
    heatmap = np.maximum(heatmap, 0)  # Apply ReLU
    heatmap = heatmap[0]  # Remove batch dimension
    heatmap /= np.max(heatmap)  # Normalize to [0, 1]
    return heatmap

# Overlay heatmap on image
def overlay_heatmap(image, heatmap):
    heatmap = np.uint8(255 * heatmap)  # Scale to [0, 255]
    heatmap = Image.fromarray(heatmap).resize(image.size, Image.BILINEAR)
    heatmap = np.array(heatmap)
    plt.imshow(image)
    plt.imshow(heatmap, cmap="jet", alpha=0.5)  # Overlay heatmap with transparency
    plt.axis("off")

# Main function to execute Grad-CAM
def main():
    pth_tar_path = "model_best.pth.tar"  # Replace with your .pth.tar file path
    image_path = "grad_cam_images/val_77.JPEG"  # Replace with your image path

    # Step 1: Load the .pth.tar file
    net_type = 'resnet'
    dataset = 'imagenet'
    depth = 50
    bottleneck = 'bottleneck'
    alpha = 300
    num_classes = 200  # Adjust based on your dataset (e.g., Tiny ImageNet has 200 classes)

    # Load the model
    model = load_model(pth_tar_path, net_type, dataset, depth, bottleneck, alpha, num_classes)

    print("Model loaded successfully!")
    model = load_model(pth_tar_path)
    input_tensor = preprocess_image(image_path)
    image = Image.open(image_path).convert("RGB")

    # Assuming the model is ResNet and using the last conv layer
    target_layer = model.layer4[-1]  # Adjust this based on your model's architecture

    gradcam = generate_gradcam(model, input_tensor, target_layer)
    overlay_heatmap(image, gradcam)
    plt.show()

if __name__ == '__main__':
    main()

