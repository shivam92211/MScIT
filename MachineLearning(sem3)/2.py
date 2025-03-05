import pandas as pd
# Step 1: Create a simple dataset and save it to a CSV file
data = {
    "Sky": ["Sunny", "Sunny", "Rainy", "Sunny", "Sunny"],
    "AirTemp": ["Warm", "Warm", "Cold", "Warm", "Warm"],
    "Humidity": ["Normal", "High", "High", "High", "Normal"],
    "Wind": ["Strong", "Strong", "Strong", "Strong", "Strong"],
    "Water": ["Warm", "Warm", "Warm", "Cool", "Warm"],
    "Forecast": ["Same", "Same", "Change", "Same", "Same"],
    "EnjoySport": ["Yes", "Yes", "No", "Yes", "Yes"]
}

# Save dataset to CSV for demonstration purposes
df = pd.DataFrame(data)
df.to_csv("training_data.csv", index=False)

# Step 2: Load the training data from CSV
training_data = pd.read_csv("training_data.csv")

# Step 3: Implement FIND-S algorithm
def find_s_algorithm(data):
    # Extract features and target
    features = data.iloc[:, :-1].values  # All columns except the last
    target = data.iloc[:, -1].values     # The last column (target)
    
    # Initialize the most specific hypothesis
    hypothesis = ["ϕ"] * features.shape[1]  # Start with the most specific hypothesis
    
    # Update hypothesis for positive examples
    for i, row in enumerate(features):
        if target[i] == "Yes":  # Consider only positive examples
            for j in range(len(hypothesis)):
                if hypothesis[j] == "ϕ":  # Update if still specific
                    hypothesis[j] = row[j]
                elif hypothesis[j] != row[j]:  # Generalize if mismatch
                    hypothesis[j] = "?"
    
    return hypothesis

# Step 4: Run the FIND-S algorithm
final_hypothesis = find_s_algorithm(training_data)

# Step 5: Display the results
print("Final Specific Hypothesis:", final_hypothesis)
