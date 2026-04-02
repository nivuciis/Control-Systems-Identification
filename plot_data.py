import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Get all txt files from data directory
data_dir = Path('data')
txt_files = sorted(data_dir.glob('*.txt'))

if not txt_files:
    print("No .txt files found in data/ directory")
    exit(1)

# Create figure with subplots (or single plot)
fig, ax = plt.subplots(figsize=(10, 6))

for file_path in txt_files:
    # Load data
    data = np.loadtxt(file_path, delimiter=',')

    # Extract columns: first column is value, second is time
    values = data[:, 0]
    time = data[:, 1]

    # Plot with label from filename
    label = file_path.stem
    ax.plot(time, values, label=label, marker='o', markersize=3, linestyle='-')

ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.set_title('Data from all txt files')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig('data_plot.png', dpi=300)
plt.show()

print(f"Plotted {len(txt_files)} files")
print(f"Saved to data_plot.png")
