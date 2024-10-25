import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def load_data_from_folder(folder):
    emails = []
    labels = []
    
    ham_folder = os.path.join(folder, 'ham')
    for filename in os.listdir(ham_folder):
        with open(os.path.join(ham_folder, filename), 'r', encoding='utf-8', errors='ignore') as file:
            emails.append(file.read())
            labels.append(0)

    spam_folder = os.path.join(folder, 'spam')
    for filename in os.listdir(spam_folder):
        with open(os.path.join(spam_folder, filename), 'r', encoding='utf-8', errors='ignore') as file:
            emails.append(file.read())
            labels.append(1)

    return pd.DataFrame({'message': emails, 'label': labels})

train_data = load_data_from_folder('train')
test_data = load_data_from_folder('test')

X_train = train_data['message']
y_train = train_data['label']
X_test = test_data['message']
y_test = test_data['label']

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)        

def train_and_evaluate(model):
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)       
    return accuracy_score(y_test, y_pred)

models = {
    "Naive Bayes": MultinomialNB(),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

model_scores = {name: train_and_evaluate(model) for name, model in models.items()}

for model_name, score in model_scores.items():
    print(f"{model_name}: {score:.4f}")

final_model = MultinomialNB()
final_model.fit(X_train_vec, y_train) 

def predict_spam(text):
    text_vec = vectorizer.transform([text]) 
    prediction = final_model.predict(text_vec)
    return 'Spam' if prediction[0] == 1 else 'Ham'
