import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load the results from the JSON file
output_file = Path('grid_search_round_2_grid_search_tutorial.json')  # Filename of your JSON file

with output_file.open("r", encoding="utf-8") as file:
    results = json.load(file)

# Extracting the relevant data (assuming the format of results is a list of dictionaries)
# We are focusing on PICNIC_BASKET1
long_thresh_values = []
short_thresh_values = []
pnls = []

for result in results:
    if 'PICNIC_BASKET1' in result:
        long_thresh = result['PICNIC_BASKET1']['long_thresh']
        short_thresh = result['PICNIC_BASKET1']['short_thresh']
        pnl = result['total_PICNIC_BASKET1_pnl']
        
        long_thresh_values.append(long_thresh)
        short_thresh_values.append(short_thresh)
        pnls.append(pnl)

# Now, let's organize the data for the heatmap
# Create unique long_thresh and short_thresh values
long_thresh_unique = np.unique(long_thresh_values)
short_thresh_unique = np.unique(short_thresh_values)

# Initialize a 2D array to store the pnl values for the heatmap
heatmap_data = np.zeros((len(long_thresh_unique), len(short_thresh_unique)))

# Populate the heatmap data array with pnl values
for i, long_thresh in enumerate(long_thresh_unique):
    for j, short_thresh in enumerate(short_thresh_unique):
        # Find the corresponding pnl value for this combination of long_thresh and short_thresh
        matching_pnls = [pnl for k, pnl in zip(range(len(long_thresh_values)), pnls) 
                         if long_thresh_values[k] == long_thresh and short_thresh_values[k] == short_thresh]
        
        if matching_pnls:
            heatmap_data[i, j] = matching_pnls[0]

# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", xticklabels=long_thresh_unique, yticklabels=short_thresh_unique)
plt.title('Heatmap of long_thresh vs short_thresh')
plt.xlabel('long_thresh')
plt.ylabel('short_thresh')
plt.show()
