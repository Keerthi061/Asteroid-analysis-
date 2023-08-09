import pandas as pd

# Load the asteroid dataset into a Pandas DataFrame
df = pd.read_csv('C:\\Users\\KEERTHI\\OneDrive\\Documents\\Preprocessing\\nasa11.csv')

# Calculate the kinetic energy for each asteroid using estimated diameters and relative velocities during close approach
import numpy as np
import matplotlib.pyplot as plt

# Calculate kinetic energy
df['kinetic_energy'] = 0.5 * 1000 * (df['est_diameter_max']/2)**2 * df['relative_velocity']**2 / 10000000000000
#1000000000

# Create size bins
size_bins = [0,2,4,6,8,10, np.inf]
size_labels = ['(0.0, 2.0]', '2.0, 4.0]', '(4.0, 6.0]', 
               '(6.0, 8.0]', '(8.0, 10.0]','(10, inf]']

df['size_range'] = pd.cut(df['est_diameter_max'], bins=size_bins, labels=size_labels)

# Group by size range and calculate average kinetic energy
grouped_df = df.groupby('size_range')['kinetic_energy'].mean()
grouped_df = grouped_df.fillna(0)

# Create bar plot
plt.bar(size_labels, grouped_df)
plt.xlabel('Asteroid Size Range (m)')
plt.ylabel('Average Kinetic Energy (GJ)')
plt.title('Relationship Between Asteroid Size and Kinetic Energy')
plt.show()
