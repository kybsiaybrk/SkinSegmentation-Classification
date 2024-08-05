import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'Skin_NonSkin.txt'
df = pd.read_csv(file_path, delimiter='\t', header=None)

df.columns = ['RED', 'GREEN', 'BLUE', 'Skin']
scaler = MinMaxScaler()
df[['RED', 'GREEN', 'BLUE']] = scaler.fit_transform(df[['RED', 'GREEN', 'BLUE']])

x = df[['RED', 'GREEN', 'BLUE']]
y = df['Skin']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
y_pred_knn = knn.predict(x_test)
knn_accuracy = accuracy_score(y_test, y_pred_knn)
knn_classification_report = classification_report(y_test, y_pred_knn)

# Naive Bayes
gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred_gnb = gnb.predict(x_test)
gnb_accuracy = accuracy_score(y_test, y_pred_gnb)
gnb_classification_report = classification_report(y_test, y_pred_gnb)

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(x_train, y_train)
y_pred_dt = dt.predict(x_test)
dt_accuracy = accuracy_score(y_test, y_pred_dt)
dt_classification_report = classification_report(y_test, y_pred_dt)

def plot_confusion_matrix(y_test, y_pred, title):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()

plot_confusion_matrix(y_test, y_pred_knn, 'Confusion Matrix for KNN')
plot_confusion_matrix(y_test, y_pred_gnb, 'Confusion Matrix for Gaussian Naive Bayes')
plot_confusion_matrix(y_test, y_pred_dt, 'Confusion Matrix for Decision Tree')

print("KNN Classification Report:")
print(knn_classification_report)
print(f"KNN Accuracy: {knn_accuracy}\n")

print("Gaussian Naive Bayes Classification Report:")
print(gnb_classification_report)
print(f"Gaussian Naive Bayes Accuracy: {gnb_accuracy}\n")

print("Decision Tree Classification Report:")
print(dt_classification_report)
print(f"Decision Tree Accuracy: {dt_accuracy}\n")
