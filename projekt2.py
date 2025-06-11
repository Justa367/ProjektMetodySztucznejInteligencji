import sklearn.datasets as skdatasets 
from sklearn.metrics import precision_score 
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
import matplotlib.pyplot as plt 
import seaborn as sns # type: ignore
import pandas as pd
from sklearn.model_selection import RepeatedKFold 
import sklearn as sk  
import numpy as np 
from sklearn.metrics import f1_score 
from sklearn.metrics import recall_score 
from sklearn.metrics import balanced_accuracy_score 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Dataset: load_breast_cancer
# Typ problemu: klasyfikacja binarna
# Liczba klas: 2 (malignant, benign)
# Liczba cech: 30 (numeryczne)
# Niezbalansowanie: umiarkowane – benign: 357 (62.7%), malignant: 212 (37.3%)

# Dataset: fetch_covtype
# Typ problemu: klasyfikacja wieloklasowa
# Liczba klas: 7 (typ pokrycia terenu)
# Liczba cech: 54 (10 numerycznych + 44 binarne)
# Niezbalansowanie: silne – klasa 2: ~49%, inne klasy znacząco mniejsze (<10%)

# Dataset: fetch_kddcup99
# Typ problemu: klasyfikacja wieloklasowa (anomalie w ruchu sieciowym)
# Liczba klas: >20 (normal + różne typy ataków)
# Liczba cech: 41 (numeryczne i kategoryczne)
# Niezbalansowanie: bardzo silne – np. neptune + normal dominują (~80%)

# Dataset: fetch_openml(name='credit-g')
# Typ problemu: klasyfikacja binarna (zdolność kredytowa)
# Liczba klas: 2 (good, bad)
# Liczba cech: 20 (numeryczne i kategoryczne)
# Niezbalansowanie: silne – good: 700 (70%), bad: 300 (30%)

# Dataset: fetch_openml(name='PhishingWebsites')
# Typ problemu: klasyfikacja binarna (phishing vs. legalne strony)
# Liczba klas: 2 (1 = phishing, -1 = legal)
# Liczba cech: 30 (wszystkie binarne: -1, 0, 1)
# Niezbalansowanie: umiarkowane – phishing lekko dominuje (~55–60%)

# Funkcje do ładowania dodatkowych danych
def load_creditcard_data():
    """Load credit card fraud dataset"""
    df = pd.read_csv('creditcard.csv')
    X = df.drop('Class', axis=1).values
    y = df['Class'].values
    return X, y

def load_telco_churn_data():
    """Load and preprocess Telco Customer Churn dataset from CSV"""
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    df.drop(['customerID'], axis=1, inplace=True)
    
    binary_cols = {
        'gender': {'Female': 0, 'Male': 1},
        'Partner': {'Yes': 1, 'No': 0},
        'Dependents': {'Yes': 1, 'No': 0},
        'PhoneService': {'Yes': 1, 'No': 0},
        'PaperlessBilling': {'Yes': 1, 'No': 0},
        'Churn': {'Yes': 1, 'No': 0}
    }
    
    for col, mapping in binary_cols.items():
        df[col] = df[col].map(mapping)
    
    df['SeniorCitizen'] = df['SeniorCitizen'].astype(int)
    
    one_hot_cols = [
        'MultipleLines',
        'InternetService',
        'OnlineSecurity',
        'OnlineBackup',
        'DeviceProtection',
        'TechSupport',
        'StreamingTV',
        'StreamingMovies',
        'Contract',
        'PaymentMethod'
    ]
    
    df = pd.get_dummies(df, columns=one_hot_cols, drop_first=True)
    
    X = df.drop('Churn', axis=1).values
    y = df['Churn'].values
    
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return X, y

class ResamplingClassifier:
    def __init__(self, base_classifier=None, base_preprocessing=None):
        self.base_classifier = base_classifier
        self.base_preprocessing = base_preprocessing

    def fit(self, X, y):
        if self.base_preprocessing:
            X, y = self.base_preprocessing.fit_resample(X, y)
        self.base_classifier.fit(X, y)

    def predict(self, X):
        return self.base_classifier.predict(X)

# Załaduj wszystkie zestawy danych
classifications = {
    "9:1": skdatasets.make_classification(weights=[0.9, 0.1], n_samples=10000),
    "99:1": skdatasets.make_classification(weights=[0.99, 0.01], n_samples=10000),
    "999:1": skdatasets.make_classification(weights=[0.999, 0.001], n_samples=10000),
    "CreditCard": load_creditcard_data(),
    "TelcoChurn": load_telco_churn_data()
}

# Konfiguracja walidacji krzyżowej
rkf = RepeatedKFold(n_splits=2, n_repeats=5, random_state=42)

# Słownik metod
samplers = {
    "Naive bayes No Oversampling": ResamplingClassifier(base_classifier=sk.naive_bayes.GaussianNB(), base_preprocessing=None),
    "Naive bayes Random Oversampling": ResamplingClassifier(base_classifier=sk.naive_bayes.GaussianNB(), base_preprocessing=RandomOverSampler(random_state=42)),
    "Naive bayes SMOTE": ResamplingClassifier(base_classifier=sk.naive_bayes.GaussianNB(), base_preprocessing=SMOTE(random_state=42)),
    "Naive bayes ADASYN": ResamplingClassifier(base_classifier=sk.naive_bayes.GaussianNB(), base_preprocessing=ADASYN(random_state=42)),
    "KNeighbors No Oversampling": ResamplingClassifier(base_classifier=KNeighborsClassifier(n_neighbors=5), base_preprocessing=None),
    "KNeighbors Random Oversampling": ResamplingClassifier(base_classifier=KNeighborsClassifier(n_neighbors=5), base_preprocessing=RandomOverSampler(random_state=42)),
    "KNeighbors SMOTE": ResamplingClassifier(base_classifier=KNeighborsClassifier(n_neighbors=5), base_preprocessing=SMOTE(random_state=42)),
    "KNeighbors ADASYN": ResamplingClassifier(base_classifier=KNeighborsClassifier(n_neighbors=5), base_preprocessing=ADASYN(random_state=42))
}

# Inicjalizacja wyników
n_datasets = len(classifications)
n_methods = len(samplers)
n_splits = rkf.get_n_splits()

f1_scores = np.zeros((n_methods, n_datasets, n_splits))
precision_scores = np.zeros((n_methods, n_datasets, n_splits))
recall_scores = np.zeros((n_methods, n_datasets, n_splits))
balanced_accuracy_scores = np.zeros((n_methods, n_datasets, n_splits))

# Eksperyment
for i, (method_name, clf) in enumerate(samplers.items()):
    for j, (dataset_name, (X, y)) in enumerate(classifications.items()):
        print(f"Processing {method_name} on {dataset_name}...")
        
        for k, (train_index, test_index) in enumerate(rkf.split(X, y)):
            try:
                # Handle missing values
                imputer = SimpleImputer(strategy='mean')
                X_train = imputer.fit_transform(X[train_index])
                X_test = imputer.transform(X[test_index])
                
                clf.fit(X_train, y[train_index])
                y_pred = clf.predict(X_test)
                
                f1_scores[i, j, k] = f1_score(y[test_index], y_pred, zero_division=0)
                precision_scores[i, j, k] = precision_score(y[test_index], y_pred, zero_division=0)
                recall_scores[i, j, k] = recall_score(y[test_index], y_pred, zero_division=0)
                balanced_accuracy_scores[i, j, k] = balanced_accuracy_score(y[test_index], y_pred)
            except Exception as e:
                print(f"Error with {method_name} on {dataset_name} (fold {k}): {str(e)}")
                f1_scores[i, j, k] = np.nan
                precision_scores[i, j, k] = np.nan
                recall_scores[i, j, k] = np.nan
                balanced_accuracy_scores[i, j, k] = np.nan

# Przygotowanie wyników
results = []            
for i, (method_name, _) in enumerate(samplers.items()):
    for j, (dataset_name, _) in enumerate(classifications.items()):
        # Oblicz statystyki ignorując NaN
        valid_scores = ~np.isnan(f1_scores[i, j, :])
        n_valid = np.sum(valid_scores)
        
        if n_valid > 0:
            results.append({
                "Metoda": method_name,
                "Klasyfikacja": dataset_name,
                "F1 mean": np.nanmean(f1_scores[i, j, :]),
                "F1 std": np.nanstd(f1_scores[i, j, :]),
                "Precision mean": np.nanmean(precision_scores[i, j, :]),
                "Precision std": np.nanstd(precision_scores[i, j, :]),
                "Recall mean": np.nanmean(recall_scores[i, j, :]),
                "Recall std": np.nanstd(recall_scores[i, j, :]),
                "Balanced accuracy mean": np.nanmean(balanced_accuracy_scores[i, j, :]),
                "Balanced accuracy std": np.nanstd(balanced_accuracy_scores[i, j, :]),
                "Valid folds": n_valid
            })
        else:
            print(f"Warning: No valid results for {method_name} on {dataset_name}")

# Zapis wyników
df = pd.DataFrame(results)
df.to_csv("results.csv", index=False)
np.save("results.npy", np.array(results, dtype=object))
print("\nResults summary:")
print(df)

# Wizualizacja
def plot_results(metric, title, filename):
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="Metoda", y=metric, hue="Klasyfikacja")
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Wykresy dla średnich
metrics = [
    ("F1 mean", "F1 Score (mean)", "f1_mean.png"),
    ("Precision mean", "Precision (mean)", "precision_mean.png"),
    ("Recall mean", "Recall (mean)", "recall_mean.png"),
    ("Balanced accuracy mean", "Balanced Accuracy (mean)", "balanced_accuracy_mean.png")
]

for metric, title, filename in metrics:
    plot_results(metric, title, filename)

# Wykresy dla odchyleń standardowych
std_metrics = [
    ("F1 std", "F1 Score (std)", "f1_std.png"),
    ("Precision std", "Precision (std)", "precision_std.png"),
    ("Recall std", "Recall (std)", "recall_std.png"),
    ("Balanced accuracy std", "Balanced Accuracy (std)", "balanced_accuracy_std.png")
]

for metric, title, filename in std_metrics:
    plot_results(metric, title, filename)

# Wykresy porównawcze dla każdego zbioru danych
for dataset in df['Klasyfikacja'].unique():
    dataset_df = df[df['Klasyfikacja'] == dataset]
    
    plt.figure(figsize=(10, 6))
    metrics_to_plot = ['F1 mean', 'Precision mean', 'Recall mean', 'Balanced accuracy mean']
    for metric in metrics_to_plot:
        plt.plot(dataset_df['Metoda'], dataset_df[metric], label=metric)
    
    plt.title(f"Metrics comparison for {dataset}")
    plt.xlabel("Method")
    plt.ylabel("Score")
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"comparison_{dataset}.png")
    plt.close()

print("\nVisualizations saved to PNG files.")