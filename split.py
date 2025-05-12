# After detect drones you can look a graph for model accuracy and when detect drone

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import os

# Correct the path to the CSV file
detection_csv_path = 'detection_results.csv'
line_graph_path = 'detection_confidence_line_graph.png'
scatter_plot_path = 'detection_confidence_scatter_plot.png'

# Check if the CSV file exists
if not os.path.exists(detection_csv_path):
    print(f"CSV file {detection_csv_path} does not exist.")
    exit()

try:
    # Load the CSV file into a DataFrame
    df_detection = pd.read_csv(detection_csv_path, skiprows=1, header=None, names=['Timestamp', 'Label', 'Confidence'])

    # Check if the DataFrame is empty
    print("CSV Preview: No data found in the CSV file.", df_detection.head())

    # Convert the 'Timestamp' column to datetime format
    df_detection['Timestamp'] = pd.to_datetime(df_detection['Timestamp'], format='%Y-%m-%d %H:%M:%S')

    # Sort valued (optional but good for Plots)
    df_detection.sort_values('Timestamp', inplace=True)

except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit()

# Plotting the confidence scores over time
# line graph
plt.figure(figsize=(14, 6))
plt.plot(df_detection['Timestamp'], df_detection['Confidence'], marker='o', linestyle='-', color='b')
plt.xlabel('Time', fontsize=12)
plt.ylabel('Detection Confidence', fontsize=12)
plt.title('Drone Detection Confidence Over Time', fontsize=14)
plt.xticks(rotation=45)
plt.ylim(0, 1)  # Set y-axis limits to [0, 1]
plt.grid(True)
plt.tight_layout()
plt.savefig(line_graph_path)
plt.show()

# Scatter plot
plt.figure(figsize=(14, 6))
plt.scatter(df_detection['Timestamp'], df_detection['Confidence'], marker='0', linestyle='-', color='dodgerblue', alpha=0.5)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Detection Confidence', fontsize=12)
plt.title('Drone Detection Confidence Over Time', fontsize=14)
plt.xticks(rotation=45)
plt.ylim(0, 1)  # Set y-axis limits to [0, 1]
plt.grid(True)
plt.tight_layout()
plt.savefig(line_graph_path)
plt.show()

# Scatter plot
plt.figure(figsize=(14, 6))
plt.scatter(df_detection['Timestamp'], df_detection['Confidence'], color='crimson', alpha=0.5)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Detection Confidence', fontsize=12)
plt.title('Drone Detection Confidence Scatter Plot', fontsize=14)
plt.xticks(rotation=45)
plt.ylim(0, 1)  # Set y-axis limits to [0, 1]
plt.grid(True)
plt.tight_layout()
plt.savefig(scatter_plot_path)
plt.show()

# Print the path to the saved plots
print(f"Line graph saved to: {line_graph_path}")
print(f"Scatter plot saved to: {scatter_plot_path}")
