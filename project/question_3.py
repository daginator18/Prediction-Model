from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd
from question_2 import remove_unknown, importance

#Function to create our prediction model
def prediction_module():
    
    file = 'credit_card_customers.xlsx'
    df = pd.read_excel(file)
    df = remove_unknown(df)

    X = df.drop('Attrition_Flag', axis=1)
    importance_features = importance(df)
    selected_features = importance_features['Feature'].iloc[:10]
    X = X[selected_features]
    X = pd.get_dummies(X) 

    selected_features = importance_features.head(10)
    
        
    y = df['Attrition_Flag']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    param_grid = {
    "n_estimators" : [100],
    "max_features" : [3,4,5],
    "min_samples_split" : [4, 5, 6],
    "max_depth" : [4,6,7]
    }

    search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5,
                        verbose=1)

    search.fit(X_train, y_train)
    # print("Best CV score: {} using {}".format(search.best_score_, search.best_params_))
    # gave this:
    # Best CV score: 0.9503887549549492 using {'max_depth': 7, 'max_features': 5, 'min_samples_split': 6, 'n_estimators': 100}
    params = search.best_params_
    clf = RandomForestClassifier(
        max_depth=params['max_depth'],       
        max_features=params['max_features'], 
        min_samples_split=params['min_samples_split'], n_estimators=params['n_estimators'], 
        random_state=42
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)


    return (clf, X_test, y_test, params, selected_features)



