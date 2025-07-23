import sys
import joblib

# 載入模型與向量器
model = joblib.load("../models/svm_model.pkl")
vectorizer = joblib.load("../models/tfidf_vectorizer.pkl")

def predict(text):
    vec = vectorizer.transform([text])
    result = model.predict(vec)
    return result[0]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("請輸入要預測的文字")
    else:
        text = sys.argv[1]
        prediction = predict(text)
        print("分類結果：", prediction)
