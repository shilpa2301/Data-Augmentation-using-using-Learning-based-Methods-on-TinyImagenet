import torch
import pickle
import matplotlib.pyplot as plt

# Step 1: Load the .pth.tar file
net_type = args.net_type
dataset = args.dataset
depth = args.depth
bottleneck = args.bottleneck
alpha = args.alpha
num_classes = 200  # Adjust based on your dataset (e.g., Tiny ImageNet has 200 classes)

# Load the model
model = load_model(pth_tar_path, net_type, dataset, depth, bottleneck, alpha, num_classes)

print("Model loaded successfully!")
pth_tar_file_path = "model_best.pth.tar"
checkpoint = torch.load(pth_tar_file_path)

# Assuming the checkpoint contains accuracy histories
resnet18_train_acc = checkpoint["train_acc_history"]  # Replace with actual key if different
resnet18_val_acc = checkpoint["val_acc_history"]      # Replace with actual key if different

# Step 2: Load the .pkl file
pkl_file_path = "resnet50_random_erase_data.pkl"
with open(pkl_file_path, "rb") as f:
    resnet50_data = pickle.load(f)

# Extract accuracy histories from the .pkl file
resnet50_train_acc = resnet50_data["train_acc_history"]
resnet50_val_acc = resnet50_data["val_acc_history"]

# Step 3: Plot training accuracy
plt.figure(figsize=(10, 5))
plt.plot(resnet18_train_acc, label="ResNet-18 Training Accuracy", linestyle="--", marker="o")
plt.plot(resnet50_train_acc, label="ResNet-50 Training Accuracy", linestyle="--", marker="x")
plt.title("Training Accuracy Comparison")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()

# Step 4: Plot validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(resnet18_val_acc, label="ResNet-18 Validation Accuracy", linestyle="-", marker="o")
plt.plot(resnet50_val_acc, label="ResNet-50 Validation Accuracy", linestyle="-", marker="x")
plt.title("Validation Accuracy Comparison")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()
