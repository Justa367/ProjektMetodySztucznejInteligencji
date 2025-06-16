
import sklearn.datasets as skdatasets # type: ignore
from sklearn.metrics import precision_score # type: ignore
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import pandas as pd # type: ignore
from sklearn.model_selection import RepeatedKFold # type: ignore
import sklearn as sk  # type: ignore
import numpy as np 
from sklearn.metrics import f1_score # type: ignore
from sklearn.metrics import recall_score # type: ignore
from sklearn.metrics import balanced_accuracy_score # type: ignore
from sklearn.neighbors import KNeighborsClassifier # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.impute import SimpleImputer # type: ignore
from sklearn.neighbors import NearestNeighbors



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


class SimpleADASYN:
    def __init__(self, sampling_strategy=1.0, n_neighbors=5, random_state=None):
        self.sampling_strategy = sampling_strategy
        self.n_neighbors = n_neighbors
        self.random_state = np.random.RandomState(random_state)

    def fit_resample(self, X, y):
        # Start: Zbiór treningowy
        X = np.asarray(X)
        y = np.asarray(y)

        # Wybierz próbki klasy mniejszościowej
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) != 2:
            raise ValueError("Obsługiwane są tylko dwa typy klas.")
        maj_class = classes[np.argmax(counts)]
        min_class = classes[np.argmin(counts)]

        X_min = X[y == min_class]
        X_maj = X[y == maj_class]

        n_min = len(X_min)
        n_maj = len(X_maj)
        n_generate = int((n_maj - n_min) * self.sampling_strategy)
        if n_generate <= 0:
            return X, y

        # Znajdź k-najbliższych sąsiadów dla każdej próbki klasy mniejszościowej
        nn = NearestNeighbors(n_neighbors=self.n_neighbors + 1)
        nn.fit(X)
        neighbors = nn.kneighbors(X_min, return_distance=False)[:, 1:]

        # Oblicz stopień trudności klasyfikacji
        difficulties = []
        for idxs in neighbors:
            neighbor_labels = y[idxs]
            difficulty = np.sum(neighbor_labels == maj_class) / self.n_neighbors
            difficulties.append(difficulty)
        difficulties = np.array(difficulties)

        # Normalizuj trudność → rozkład r′
        r_prime = difficulties / difficulties.sum()

        # Wyznacz liczbę próbek do wygenerowania g_i
        g_i = np.round(r_prime * n_generate).astype(int)

        # Interpoluj próbki (generuj nowe punkty)
        synthetic_samples = []
        for i, num in enumerate(g_i):
            for _ in range(num):
                neighbor_idx = self.random_state.choice(neighbors[i])
                diff = X[neighbor_idx] - X_min[i]
                gap = self.random_state.rand()
                new_sample = X_min[i] + gap * diff
                synthetic_samples.append(new_sample)

        # Połącz dane oryginalne i wygenerowane
        if synthetic_samples:
            X_syn = np.vstack(synthetic_samples)
            y_syn = np.full(X_syn.shape[0], min_class)
            X_resampled = np.vstack((X, X_syn))
            y_resampled = np.hstack((y, y_syn))
        else:
            X_resampled, y_resampled = X, y

        return X_resampled, y_resampled

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
    "KNeighbors ADASYN": ResamplingClassifier(base_classifier=KNeighborsClassifier(n_neighbors=5), base_preprocessing=ADASYN(random_state=42)),
    "Naive bayes SimpleADASYN": ResamplingClassifier(base_classifier=sk.naive_bayes.GaussianNB(), base_preprocessing=SimpleADASYN(random_state=42)),
    "KNeighbors SimpleADASYN": ResamplingClassifier(base_classifier=KNeighborsClassifier(n_neighbors=5), base_preprocessing=SimpleADASYN(random_state=42))
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

# Wykresy porównawcze dla każdego zbioru danych (SLUPKOWE)
for dataset in df['Klasyfikacja'].unique():
    dataset_df = df[df['Klasyfikacja'] == dataset]
    metrics_to_plot = ['F1 mean', 'Precision mean', 'Recall mean', 'Balanced accuracy mean']
    
    melted_df = dataset_df.melt(id_vars='Metoda', value_vars=metrics_to_plot,
                                var_name='Metryka', value_name='Wartość')
    
    plt.figure(figsize=(14, 6))
    sns.barplot(data=melted_df, x='Metoda', y='Wartość', hue='Metryka', errorbar=None)
    
    plt.title(f"Porównanie metryk dla zbioru: {dataset}")
    plt.xlabel("Metoda")
    plt.ylabel("Średnia wartość metryki")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Metryka', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"barplot_comparison_{dataset}.png")
    plt.close()



print("\nVisualizations saved to PNG files.")
