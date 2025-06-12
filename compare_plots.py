import re
import matplotlib.pyplot as plt

# Function to parse log files and extract data
def parse_log(file_path, pattern):
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
def plot_loss(epochs1, loss1, epochs2, loss2, output_path):
    plt.figure(figsize=(8, 6))
    plt.plot(epochs1, loss1, label="Run Job Saliency - Loss", color="blue")
    plt.plot(epochs2, loss2, label="Run Job Random Erase - Loss", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train Loss Comparison")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Loss plot saved to {output_path}")

# Function to plot errors
def plot_error(epochs1, top1_err1, top5_err1, epochs2, top1_err2, top5_err2, output_path):
    plt.figure(figsize=(8, 6))
    plt.plot(epochs1, top1_err1, label="Run Job Saliency - Top-1 Error", color="red")
    plt.plot(epochs1, top5_err1, label="Run Job Saliency - Top-5 Error", color="green")
    plt.plot(epochs2, top1_err2, label="Run Job Random Erase - Top-1 Error", color="orange")
    plt.plot(epochs2, top5_err2, label="Run Job Random Erase - Top-5 Error", color="purple")
    plt.xlabel("Epoch")
    plt.ylabel("Error (%)")
    plt.title("Top-1 and Top-5 Errors Comparison")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Error plot saved to {output_path}")

# Function to plot accuracies
def plot_accuracy(epochs1, top1_acc1, top5_acc1, epochs2, top1_acc2, top5_acc2, output_path):
    plt.figure(figsize=(8, 6))
    plt.plot(epochs1, top1_acc1, label="Run Job Saliency - Top-1 Accuracy", color="blue")
    plt.plot(epochs1, top5_acc1, label="Run Job Saliency - Top-5 Accuracy", color="cyan")
    plt.plot(epochs2, top1_acc2, label="Run Job Random Erase - Top-1 Accuracy", color="orange")
    plt.plot(epochs2, top5_acc2, label="Run Job Random Erase - Top-5 Accuracy", color="purple")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Top-1 and Top-5 Accuracies Comparison")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Accuracy plot saved to {output_path}")

# Main script
if __name__ == "__main__":
    # Paths to log files
    log1_file_path = "run_job_saliency.log"
    log2_file_path = "run_job_randomerase.log"

    # Regex patterns for each log file
    pattern1 = r"\* Epoch: \[(\d+)/300\]\t Top 1-err (\d+\.\d+)  Top 5-err (\d+\.\d+)\t Train Loss (\d+\.\d+)"
    pattern2 = r"\* Epoch: \[(\d+)/300\]\s+Top 1-err (\d+\.\d+)\s+Top 5-err (\d+\.\d+)\s+Test Loss (\d+\.\d+)"

    # Parse the logs and extract data
    epochs1, train_loss1, top1_err1, top5_err1 = parse_log(log1_file_path, pattern1)
    epochs2, train_loss2, top1_err2, top5_err2 = parse_log(log2_file_path, pattern2)

    # Convert errors to accuracies
    top1_acc1, top5_acc1 = convert_error_to_accuracy(top1_err1, top5_err1)
    top1_acc2, top5_acc2 = convert_error_to_accuracy(top1_err2, top5_err2)

    # Plot and save the images
    plot_loss(epochs1, train_loss1, epochs2, train_loss2, "combined_loss_plot.png")
    plot_error(epochs1, top1_err1, top5_err1, epochs2, top1_err2, top5_err2, "combined_error_plot.png")
    plot_accuracy(epochs1, top1_acc1, top5_acc1, epochs2, top1_acc2, top5_acc2, "combined_accuracy_plot.png")
