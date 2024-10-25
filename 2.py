import pandas as pd
from sklearn.model_selection import KFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Reading data from the csv file
data = pd.read_csv('heart.csv')

#Separating data and labels from the file and specifying x and y
X = data.drop(columns=["output"])
y = data["output"]


k_values = list(range(1, 21))

# Data Scaling
scaler =StandardScaler()
X = scaler.fit_transform(X)


results = {}

#10-fold cross-validation
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    cv = KFold(n_splits=10, shuffle=True, random_state=42)

    scoring = {'accuracy': 'accuracy',
               'precision': 'precision',
               'recall': 'recall'}
    
    scores = cross_validate(knn, X, y, cv=cv, scoring=scoring)
    
    # Store the means
    results[k] = {
        'mean_accuracy': round(scores['test_accuracy'].mean() * 100, 3),
        'mean_precision': round(scores['test_precision'].mean() * 100, 3),
        'mean_recall': round(scores['test_recall'].mean() * 100, 3)
    }

# show result
results_df = pd.DataFrame.from_dict(results, orient='index').reset_index()
results_df.columns = ['k', 'mean_accuracy', 'mean_precision', 'mean_recall']
print(results_df)

#best k based on accuracy
best_k = results_df.loc[results_df['mean_accuracy'].idxmax()]
print()
print(f"Best k based on accuracy: {best_k['k']},\033[91m Accuracy: {best_k['mean_accuracy']}\033[0m, Precision: {best_k['mean_precision']}, Recall: {best_k['mean_recall']}")
print()
#best k based on precision
best_k = results_df.loc[results_df['mean_precision'].idxmax()]
print(f"Best k based on precision: {best_k['k']}, Accuracy: {best_k['mean_accuracy']}, \033[91m Precision: {best_k['mean_precision']}\033[0m, Recall: {best_k['mean_recall']}")
print()
#best k based on recall
best_k = results_df.loc[results_df['mean_recall'].idxmax()]
print(f"Best k based on recall: {best_k['k']}, Accuracy: {best_k['mean_accuracy']}, Precision: {best_k['mean_precision']}, \033[91mRecall: {best_k['mean_recall']}\033[0m")
print()