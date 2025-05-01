# 📦 Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 📥 Step 1: Extract - Load the car evaluation data
# File should be in the same directory as your notebook
data = pd.read_csv('car_evaluation.csv', header=None)

# Manually assign column names based on known dataset structure
data.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

# Show the first few rows of the dataset
print("Initial Extracted Data:")
print(data.head())

# 🔄 Step 2: Transform - Prepare the data for visualization
# We'll analyze the distribution of car evaluation classes (e.g., unacc, acc, good, vgood)

# Count how many records fall into each class
class_counts = data['class'].value_counts().reset_index()
class_counts.columns = ['class', 'count']

print("\nTransformed Data (Class Distribution):")
print(class_counts)

# 📊 Step 3: Load (Visualize) - Show the distribution as a bar chart
sns.set(style="whitegrid")
plt.figure(figsize=(8, 5))

# Bar chart: x = class, y = number of cars
sns.barplot(x='class', y='count', data=class_counts, palette='coolwarm')

# Chart labels
plt.title('Distribution of Car Evaluation Classes', fontsize=14)
plt.xlabel('Car Evaluation Class')
plt.ylabel('Number of Cars')

# Show the chart
plt.show()
