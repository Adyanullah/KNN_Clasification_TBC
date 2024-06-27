from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load model
models = pickle.load(open("model7.pkl", "rb"))
models5 = pickle.load(open("model5.pkl", "rb"))
models3 = pickle.load(open("model3.pkl", "rb"))
model = models['model']
min_umur = models['min_umur']
max_umur = models['max_umur']
accuracy = int(models['train_accuracy'] * 100)
accuracy5 = int(models5['train_accuracy'] * 100)
accuracy3 = int(models3['train_accuracy'] * 100)
conf_matrix = models['conf_matrix']
class_report = models['class_report']

# Load datasets
dataset = pd.read_csv('data/dataset.csv')
missing_data = pd.read_csv('data/missing.csv')
transform_data = pd.read_csv('data/transform.csv')
outliers_data = pd.read_csv('data/outliers.csv')
cleaned_data = pd.read_csv('data/cleaned_data.csv')
test_data = pd.read_csv('data/test_data.csv')
train_data = pd.read_csv('data/train_data.csv')

# Calculate train and test accuracy
X_train = train_data.drop('LOKASI ANATOMI', axis=1)
y_train = train_data['LOKASI ANATOMI']
X_test = test_data.drop('LOKASI ANATOMI', axis=1)
y_test = test_data['LOKASI ANATOMI']

train_accuracy = int(model.score(X_train, y_train) * 100)
test_accuracy = int(model.score(X_test, y_test) * 100)

@app.route("/")
def data_understanding():
    data = dataset.to_dict(orient='records')
    missing = missing_data.to_dict(orient='records')
    transform = transform_data.to_dict(orient='records')
    return render_template("data_understanding.html", data=data, missing=missing, transform=transform, title="Data Understanding")

@app.route("/preprocessing")
def preprocessing():
    data = dataset.to_dict(orient='records')
    missing = missing_data.to_dict(orient='records')
    transform = transform_data.to_dict(orient='records')
    outliers = outliers_data.to_dict(orient='records')
    cleaned = cleaned_data.to_dict(orient='records')
    return render_template("preprocessing.html", data=data, missing=missing, transform=transform, outliers=outliers, cleaned=cleaned, title="Preprocessing")

@app.route("/modeling")
def modeling():
    data = dataset.to_dict(orient='records')
    test = test_data.to_dict(orient='records')
    train = train_data.to_dict(orient='records')
    return render_template("modeling.html", data=data, test=test, train=train, accuracy=accuracy, accuracy3=accuracy, accuracy5=accuracy, class_report=class_report, conf_matrix=conf_matrix, train_accuracy=train_accuracy, test_accuracy=test_accuracy, title="Modeling")

@app.route("/metode")
def metode():
    return render_template("metode.html", title="Metode Klasifikasi")

@app.route('/anggota')
def anggota():
    return render_template('anggota.html')

@app.route("/classification", methods=["POST"])
def classification():
    # Extract form values
    req = request.form.values()
    new_data = []
    
    # Normalize and process form values
    for index, x in enumerate(req):
        if index == 0:
            age = float(x)
            age_normalized = (age - min_umur) / (max_umur - min_umur)
            new_data.append(age_normalized)
        else:
            new_data.append(float(x))
    
    # Predict using KNN model
    x = [np.array(new_data)]
    y = model.predict(x)
    result = "Paru" if y[0] == 1 else "Ekstra Paru"
    
    # Render result template with classification result
    return render_template("result.html", result_text=result, accuracy=accuracy, conf_matrix=conf_matrix, class_report=class_report, title="Result")

if __name__ == "__main__":
    app.run(debug=True)
