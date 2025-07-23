import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import joblib

# 讀取資料
df = pd.read_csv("../data/sample_data.csv")

# 特徵與標籤
X = df["text"]
y = df["label"]

# 切分訓練/測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF 向量化
vectorizer = TfidfVectorizer(max_features=3000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 訓練 SVM 分類器
model = LinearSVC()
model.fit(X_train_vec, y_train)

# 預測與報告
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# 儲存模型與向量器
joblib.dump(model, "../models/svm_model.pkl")
joblib.dump(vectorizer, "../models/tfidf_vectorizer.pkl")
print("✅ 模型與向量器已儲存")
