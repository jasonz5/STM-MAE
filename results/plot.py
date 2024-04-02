import os
import re
import matplotlib.pyplot as plt

# Directory path
file_ = 'PEMS04_SimpleTrans'
log_dir = "./{0}".format(file_)
title = "Test MAE vs Epoch for {0}".format(file_)

def extract_data(log_file):
    epochs = []
    maes = []

    with open(log_file, 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            if "Result <test>" in line:
                # Find the test MAE
                mae_match = re.search(r'test_MAE: ([\d.]+)', line)
                # Find the epoch number from a previous line
                for j in range(i, -1, -1):
                    if "Epoch" in lines[j]:
                        epoch_match = re.search(r'Epoch (\d+) /', lines[j])
                        break

                if epoch_match and mae_match:
                    epoch = int(epoch_match.group(1))
                    mae = float(mae_match.group(1))
                    epochs.append(epoch)
                    maes.append(mae)

    return epochs, maes

# Extracting log files from the specified directory
log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]

# Extracting data from each log file using the extract_data function
data = {os.path.splitext(f)[0]: extract_data(os.path.join(log_dir, f)) for f in log_files}

# Plotting the data
plt.figure(figsize=(10, 6))
for label, (epochs, maes) in data.items():
    plt.plot(epochs, maes, label=label)

plt.xlabel('Epoch')
plt.ylabel('Test MAE')
plt.title(title)
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('{0}.png'.format(file_), format='png', dpi=300, bbox_inches='tight')
