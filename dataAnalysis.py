# Import necessary libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Load data from CSV files
muscle_mass_data = pd.read_csv('muscle_mass_data.csv')
cancer_outcome_data = pd.read_csv('cancer_outcome_data.csv')

# Merge data on a common key, assume 'patient_id'
merged_data = pd.merge(muscle_mass_data, cancer_outcome_data, on='patient_id')

# Define the columns
independent_vars = ['age', 'sex', 'cancer_type', 'treatment_modality', 'country']
dependent_var = 'muscle_mass_index'

# Prepare the data for regression
X = merged_data[independent_vars]
y = merged_data[dependent_var]

# Encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Add a constant to the model (intercept)
X = sm.add_constant(X)

# Fit the regression model
model = sm.OLS(y, X).fit()

# Print the model summary
print(model.summary())

# Extract the residuals and fitted values
residuals = model.resid
fitted = model.fittedvalues

# Plot residuals vs fitted values
plt.scatter(fitted, residuals)
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
plt.axhline(y=0, color='r', linestyle='-')
plt.show()

# Calculate mean muscle mass indices by country
mean_muscle_mass_by_country = merged_data.groupby('country')['muscle_mass_index'].mean().reset_index()
mean_muscle_mass_by_country.columns = ['Country', 'Mean Muscle Mass Index']

# Calculate stomach cancer incidence by country
cancer_incidence_by_country = merged_data.groupby('country')['stomach_cancer_incidence'].mean().reset_index()
cancer_incidence_by_country.columns = ['Country', 'Stomach Cancer Incidence']

# Generate tables for the manuscript
table_2 = cancer_incidence_by_country
table_3 = mean_muscle_mass_by_country

# Save the tables as CSV files
table_2.to_csv('table_2_stomach_cancer_incidence_by_country.csv', index=False)
table_3.to_csv('table_3_mean_muscle_mass_indices_by_country.csv', index=False)

# Print the tables
print("Table 2: Stomach Cancer Incidence by Country")
print(table_2)
print("\nTable 3: Mean Muscle Mass Indices by Country")
print(table_3)

# Supplementary Data
# Incidence rates by country and stomach region
incidence_rates_by_region = merged_data.groupby(['country', 'stomach_region'])['stomach_cancer_incidence'].mean().reset_index()
incidence_rates_by_region.to_csv('supplementary_data_1_incidence_rates_by_region.csv', index=False)

# Detailed statistical analyses
detailed_analysis = model.summary2().tables[1]
detailed_analysis.to_csv('supplementary_data_2_detailed_statistical_analyses.csv')

# Save the results
merged_data.to_csv('merged_data_results.csv', index=False)
