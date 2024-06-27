import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Memuat Dataset
dataset = pd.read_csv("dataset.csv")

# Hapus attribute yang tidak digunakan
dataset = dataset.drop(columns=['KECAMATAN'])

# Mengganti nilai 'Tidak dilakukan' dan 'Tidak diketahui' dengan NA
dataset['FOTO TORAKS'] = dataset['FOTO TORAKS'].replace('Tidak dilakukan', pd.NA)
dataset['STATUS HIV'] = dataset['STATUS HIV'].replace('Tidak diketahui', pd.NA)
dataset['RIWAYAT DIABETES'] = dataset['RIWAYAT DIABETES'].replace('Tidak diketahui', pd.NA)
dataset['HASIL TCM'] = dataset['HASIL TCM'].replace('Tidak dilakukan', pd.NA)

# Check persentase missing value tiap attribute
print(dataset.isnull().mean() * 100)

# Mengganti missing value dengan modus dari data
for column in dataset.columns:
    mode = dataset[column].mode()[0]
    dataset[column] = dataset[column].fillna(mode)

# Normalisasi Data
min_umur = dataset['UMUR'].min()
max_umur = dataset['UMUR'].max()
dataset['UMUR'] = (dataset['UMUR'] - min_umur) / (max_umur - min_umur)

# Mengganti nilai kategori dengan angka menggunakan apply untuk menghindari future warnings
dataset['JENIS KELAMIN'] = dataset['JENIS KELAMIN'].apply(lambda x: 0 if x == 'P' else 1)
dataset['FOTO TORAKS'] = dataset['FOTO TORAKS'].apply(lambda x: 1 if x == 'Positif' else 0)
dataset['STATUS HIV'] = dataset['STATUS HIV'].apply(lambda x: 1 if x == 'Positif' else (0 if x == 'Negatif' else 0.5))
dataset['RIWAYAT DIABETES'] = dataset['RIWAYAT DIABETES'].apply(lambda x: 1 if x == 'Ya' else (0 if x == 'Tidak' else 0.5))
dataset['HASIL TCM'] = dataset['HASIL TCM'].apply(lambda x: 2 if x == 'Rif resisten' else (1 if x == 'Rif Sensitif' else (0 if x == 'Negatif' else 0.5)))
dataset['LOKASI ANATOMI (target/output)'] = dataset['LOKASI ANATOMI (target/output)'].apply(lambda x: 1 if x == 'Paru' else 0)

# Memisahkan fitur dan target
X = dataset.drop(columns=['LOKASI ANATOMI (target/output)'])
y = dataset['LOKASI ANATOMI (target/output)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10, stratify=y)

# Inisialisasi KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Evaluasi model pada data train
y_train_pred = knn.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)

# Evaluasi model pada data test
y_test_pred = knn.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Evaluasi model
conf_matrix = confusion_matrix(y_test, y_test_pred)
class_report = classification_report(y_test, y_test_pred, output_dict=True)

# Dump model dan metrik evaluasi
model_data = {
    'model': knn,
    'min_umur': min_umur,
    'max_umur': max_umur,
    'train_accuracy': train_accuracy,
    'test_accuracy': test_accuracy,
    'conf_matrix': conf_matrix,
    'class_report': class_report
}

# pickle.dump(model_data, open("model7.pkl", "wb"))

print(f"Akurasi pada data train: {train_accuracy:.4f}")
print(f"Akurasi pada data test: {test_accuracy:.4f}")
print("Model dan metrik evaluasi telah disimpan ke model.pkl")
