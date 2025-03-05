import pandas as pd
import numpy as np
# Step 1: Prepare the data
data = {
    'Age': ['<=30', '<=30', '31-40', '31-40', '>40', '>40', '<=30', '>40', '31-40', '<=30'],
    'Income': ['High', 'Low', 'High', 'Low', 'High', 'Low', 'High', 'Low', 'High', 'Low'],
    'Buys Product': ['No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'No', 'No', 'Yes']
}
df = pd.DataFrame(data)

# Step 2: Convert categorical data into numeric values
df['Age'] = df['Age'].map({'<=30': 0, '31-40': 1, '>40': 2})
df['Income'] = df['Income'].map({'Low': 0, 'High': 1})
df['Buys Product'] = df['Buys Product'].map({'No': 0, 'Yes': 1})
# Display the dataset
print("Dataset:")
print(df)

# Step 3: Calculate the prior probabilities (P(Buys Product = Yes) and P(Buys Product = No))
prior_yes = df['Buys Product'].sum() / len(df)
prior_no = 1 - prior_yes

# Step 4: Calculate the conditional probabilities (P(Feature|Class))
# For 'Age' and 'Income', we will calculate conditional probabilities for both classes
# For Age given Buys Product = Yes
prob_age_given_yes = df[df['Buys Product'] == 1]['Age'].value_counts(normalize=True).to_dict()
# For Age given Buys Product = No
prob_age_given_no = df[df['Buys Product'] == 0]['Age'].value_counts(normalize=True).to_dict()
# For Income given Buys Product = Yes
prob_income_given_yes = df[df['Buys Product'] == 1]['Income'].value_counts(normalize=True).to_dict()
# For Income given Buys Product = No
prob_income_given_no = df[df['Buys Product'] == 0]['Income'].value_counts(normalize=True).to_dict()

# Step 5: Define the Naive Bayes Classifier function
def naive_bayes_predict(age, income):
    # P(Buys Product = Yes)
    prob_yes_given_features = prior_yes
    if age in prob_age_given_yes:
        prob_yes_given_features *= prob_age_given_yes[age]
    if income in prob_income_given_yes:
        prob_yes_given_features *= prob_income_given_yes[income]
      # P(Buys Product = No)
    prob_no_given_features = prior_no
    if age in prob_age_given_no:
        prob_no_given_features *= prob_age_given_no[age]
    if income in prob_income_given_no:
        prob_no_given_features *= prob_income_given_no[income]
       # Normalize and compare
    total_prob = prob_yes_given_features + prob_no_given_features
    prob_yes_given_features /= total_prob
    prob_no_given_features /= total_prob
    return "Yes" if prob_yes_given_features > prob_no_given_features else "No"

# Step 6: Test the model with a new sample
test_sample = {'Age': 1, 'Income': 1}  # Age = 31-40, Income = High
prediction = naive_bayes_predict(test_sample['Age'], test_sample['Income'])
print("\nPredicted class for the test sample (Age = 31-40, Income = High):", prediction)
