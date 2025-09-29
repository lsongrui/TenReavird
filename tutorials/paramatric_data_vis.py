#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mohamed Elrefaie, mohamed.elrefaie@mit.edu, mohamed.elrefaie@tum.de

This module is part of the research presented in the paper
"DrivAerNet: A Parametric Car Dataset for Data-driven Aerodynamic Design and Graph-Based Drag Prediction".

It extends the work by providing an interactive visualization of the 3D car models in the DrivAerNet dataset.
This tutorial aims to facilitate an intuitive understanding of the dataset's structure
and the aerodynamic features of the car models it contains.

This tutorial will guide users through the process of loading, visualizing, and interacting with the 3D data
of car models from the DrivAerNet dataset. Users will learn how to navigate the dataset's file and folder structure,
visualize individual car models, and apply basic mesh operations to gain insights into the aerodynamic properties
of the models.

"""

# Data Visualization
"""

File: AeroCoefficients_DrivAerNet_FilteredCorrected.csv

This snippet demonstrates data visualization using Seaborn by creating histograms, scatter plots, 
and box plots of aerodynamic coefficients from the DrivAerNet dataset.
"""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
file_path = '../ParametricModels/DrivAerNet_ParametricData.csv'
data = pd.read_csv(file_path)

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Create a figure to hold the subplots
plt.figure(figsize=(20, 10))

# Histogram of Average Cd
plt.subplot(2, 2, 1)
sns.histplot(data['Average Cd'], kde=True)
plt.title('Histogram of Average Drag Coefficient (Cd)')

# Histogram of Average Cl
plt.subplot(2, 2, 2)
sns.histplot(data['Average Cl'], kde=True)
plt.title('Histogram of Average Lift Coefficient (Cl)')

# Scatter plot of Average Cd vs. Average Cl
plt.subplot(2, 2, 3)
sns.scatterplot(x='Average Cd', y='Average Cl', data=data)
plt.title('Average Drag Coefficient (Cd) vs. Average Lift Coefficient (Cl)')

# Box plot of all aerodynamic coefficients
plt.subplot(2, 2, 4)
melted_data = data.melt(value_vars=['Average Cd', 'Average Cl', 'Average Cl_f', 'Average Cl_r'], var_name='Coefficient',
                        value_name='Value')
sns.boxplot(x='Coefficient', y='Value', data=melted_data)
plt.title('Box Plot of Aerodynamic Coefficients')

plt.tight_layout()
plt.show()

