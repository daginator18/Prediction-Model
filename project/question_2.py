import pandas as pd
from sklearn.ensemble import RandomForestClassifier

file = 'credit_card_customers.xlsx'

#Clean data 
def remove_unknown(df: pd.DataFrame):
    df.replace('Unknown', pd.NA, inplace=True)
    not_scammed = df.drop(["Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
               "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2"],
               axis=1)
    return not_scammed.dropna()

#Selecting targetted data
def importance(df: pd.DataFrame):
    X = df.drop('Attrition_Flag', axis=1)
    X = pd.get_dummies(X) 
    
    y = df['Attrition_Flag']
    
    random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
    random_forest.fit(X, y)

    feature_df = pd.DataFrame({'Feature': X.columns, 'Importance': random_forest.feature_importances_})

    feature_df = feature_df.sort_values(by="Importance", ascending=False)

    return feature_df


