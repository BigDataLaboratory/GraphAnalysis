import json
import matplotlib.pyplot as plt
import os
 
os.chdir('/home/pasquini/GraphAnalysis/resources')
# Load data from JSON file


with open('Cluster1_nodes.json', 'r') as f:

    data = json.load(f)

 

# Create an empty dictionary to store the frequency of each node number

frequency_distribution = {}

 

# Iterate through each cluster in the data

for cluster_id, node_number in data.items():

    # Convert node_number to int

    node_number = int(node_number)

    # Increment the frequency count for the node number

    frequency_distribution[node_number] = frequency_distribution.get(node_number, 0) + 1

 

# Print the frequency distribution

print("Frequency Distribution:")

for node_number, frequency in sorted(frequency_distribution.items()):

    print(f"Node number: {node_number}, Frequency: {frequency}")


# Plot the frequency distribution

plt.figure(figsize=(12, 6))

plt.bar(frequency_distribution.keys(), frequency_distribution.values(), color='skyblue')

plt.xlabel('Node Number', fontsize=14)

plt.ylabel('Frequency', fontsize=14)

plt.title('Frequency Distribution of Node Numbers', fontsize=16)

plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()









