import pandas as pd

# Read data from Excel file
file_path = 'credit_card_customers.xlsx'
df = pd.read_excel(file_path)

# Define function to extract relevant data
def attrition_customer_data(df):
    attrition_customers = df[df['Attrition_Flag'] == 'Attrited Customer']
    distributions = {
        'Gender': attrition_customers['Gender'].value_counts(normalize=True),
        'Marital_Status': attrition_customers['Marital_Status'].value_counts(normalize=True),
        'Income_Category': attrition_customers['Income_Category'].value_counts(normalize=True),
        'Months_on_book': attrition_customers['Months_on_book'].value_counts(normalize=True),
        'Education_Level': attrition_customers['Education_Level'].value_counts(normalize=True)
    }
    return distributions

# Get the distributions data
distributions = attrition_customer_data(df)

