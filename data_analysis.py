# data_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
try:
    from sklearn.datasets import load_iris
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
except Exception as e:
    print("Error loading dataset:", e)
    exit()

# Explore dataset
print("First 5 rows:\n", df.head())
print("\nDataset info:\n")
print(df.info())
print("\nMissing values:\n", df.isnull().sum())

# Basic cleaning (if needed)
df = df.dropna()  # or df.fillna(value)



# Basic statistics
print("\nBasic statistics:\n", df.describe())

# Grouping by species
grouped = df.groupby('species').mean()
print("\nMean values by species:\n", grouped)


# Set seaborn style
sns.set(style="whitegrid")

# 1️⃣ Line chart (trend over index)
plt.figure(figsize=(8, 5))
plt.plot(df.index, df['sepal length (cm)'], label='Sepal Length')
plt.title("Sepal Length Trend")
plt.xlabel("Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.show()

# 2️⃣ Bar chart (average petal length per species)
plt.figure(figsize=(8, 5))
sns.barplot(x=grouped.index, y=grouped['petal length (cm)'])
plt.title("Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.show()

# 3️⃣ Histogram (distribution of sepal width)
plt.figure(figsize=(8, 5))
plt.hist(df['sepal width (cm)'], bins=10, color='skyblue', edgecolor='black')
plt.title("Sepal Width Distribution")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()

# 4️⃣ Scatter plot (sepal length vs petal length)
plt.figure(figsize=(8, 5))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df)
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.show()
