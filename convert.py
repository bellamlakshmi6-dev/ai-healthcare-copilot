import pandas as pd

# Load CSV file
df = pd.read_csv("brain_tumor_dataset.csv")

# Print column names (important to verify)
print(df.columns)

# Convert to text format
with open("sample.txt", "w") as f:
    for _, row in df.iterrows():
        text = f"Patient is a {row['Patient Age']}-year-old {row['Gender']} diagnosed with {row['Tumor Type']} in the {row['Location']}. Tumor size is {row['Size (cm)']} cm and grade is {row['Grade']}."
        f.write(text + "\n\n")

print("✅ Conversion complete! Check sample.txt")