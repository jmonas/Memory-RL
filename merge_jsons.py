import json

# Define the paths to the files
file1 = 'results/debate_results_2_3_2024-05-06_06-15-05_0.4.json'
file2 = 'debate_results_2_3_2024-05-06_06-15-05_0.4.json'

# Function to read JSON data from a file
def read_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Function to write JSON data to a file
def write_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

# Read the data from both files
data1 = read_json(file1)
data2 = read_json(file2)

# Combine the lists (assuming the top-level JSON structure is a list)
combined_data = data1 + data2

# Define the output file path
output_file = 'combined_2_agents_3_rounds_results.json'

# Write the combined data to the output file
write_json(combined_data, output_file)
