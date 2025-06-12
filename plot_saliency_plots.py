import re
import matplotlib.pyplot as plt

# Function to parse the log file and extract data
def parse_log(file_path):
    # Regex pattern for extracting epoch, train loss, top-1 error, and top-5 error
    pattern = r"\* Epoch: \[(\d+)/300\]\t Top 1-err (\d+\.\d+)  Top 5-err (\d+\.\d+)\t Train Loss (\d+\.\d+)"
    
    # Initialize lists to store data
    epochs, train_loss, top1_err, top5_err = [], [], [], []
    
    # Read the file and extract matches
    with open(file_path, "r") as file:
        log_content = file.read()
        matches = re.findall(pattern, log_content)
    
    # Populate the lists with extracted data
    for match in matches:
        epoch, top1, top5, train = match
        epochs.append(int(epoch))
        top1_err.append(float(top1))
        top5_err.append(float(top5))
        train_loss.append(float(train))
    
    return epochs, train_loss, top1_err, top5_err

# Function to convert errors to accuracies
def convert_error_to_accuracy(top1_err, top5_err):
    top1_accuracy = [100 - err for err in top1_err]
    top5_accuracy = [100 - err for err in top5_err]
    return top1_accuracy, top5_accuracy

# Function to plot loss
def plot_loss(epochs, train_loss, output_path):
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_loss, label="Train Loss", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train Loss")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Loss plot saved to {output_path}")

# Function to plot error
def plot_error(epochs, top1_err, top5_err, output_path):
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, top1_err, label="Top-1 Error", color="red")
    plt.plot(epochs, top5_err, label="Top-5 Error", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Error (%)")
    plt.title("Top-1 and Top-5 Error")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Error plot saved to {output_path}")

# Function to plot accuracy
def plot_accuracy(epochs, top1_accuracy, top5_accuracy, output_path):
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, top1_accuracy, label="Top-1 Accuracy", color="purple")
    plt.plot(epochs, top5_accuracy, label="Top-5 Accuracy", color="cyan")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Top-1 and Top-5 Accuracy")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Accuracy plot saved to {output_path}")

# Main script
if __name__ == "__main__":
    log1_file_path = "run_job_saliency.log"  # Path to the first log file

    # Parse the logs and extract data
    epochs, train_loss, top1_err, top5_err = parse_log(log1_file_path)

    # Convert errors to accuracies
    top1_accuracy, top5_accuracy = convert_error_to_accuracy(top1_err, top5_err)

    # Plot and save the images
    plot_loss(epochs, train_loss, "loss_plot.png")
    plot_error(epochs, top1_err, top5_err, "error_plot.png")
    plot_accuracy(epochs, top1_accuracy, top5_accuracy, "accuracy_plot.png")
