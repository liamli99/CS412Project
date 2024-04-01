import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Assuming 'data' is a DataFrame with the same structure as the one implied by the image
data = {
    'Feature': ['age', 'gender',  \
                'neoadjuvant\n_treatment', \
                'vital_status'],
    'pvalue': [6.78002814e-05, 5.22150047e-02, 1.55544336e-16, 1.06786393e-09]
}
features = data['Feature']
T = data['pvalue']
pvalues = data['pvalue']

# The number of features
num_features = len(features)

# Create a 2D array of shape (num_features, 1) because we only have one 'T' value per feature
T_array = np.array(T).reshape(num_features, 1)

# Normalize T values to map them to the colormap
norm = Normalize(vmin=T_array.min(), vmax=T_array.max())

# Create the heatmap
plt.figure(figsize=(3, 5))  # Adjust the size to fit your data
heatmap = plt.imshow(T_array, cmap='Blues', norm=norm, aspect='auto')

# Add color bar to the right
cbar = plt.colorbar(heatmap)
# cbar.set_label('T value')

# Add feature names as y-axis labels
plt.yticks(np.arange(num_features), features)

# Remove x-axis labels
plt.xticks([])
plt.xlabel('p-value')

# Annotate each cell with the corresponding p-value
for i in range(num_features):
    plt.text(0, i, f'{pvalues[i]:.2f}', ha='center', va='center', color='black')

# Optionally, add title and adjust layout
plt.title('Clinical Data')
plt.tight_layout()

# Show the plot
# plt.show()


# Show the plot
plt.savefig('t_statistics.png')
