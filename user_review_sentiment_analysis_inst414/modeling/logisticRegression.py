import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

def trainLogisticRegression():
    df = pd.read_csv("data/processed/combined_cleaned.csv")
    
    print(f"Total reviews: {len(df)}")
    
    df = df.dropna(subset = ["cleanedText"])
    df = df[df["cleanedText"] != ""]
    
    df = df.sample(n = 500000, random_state = 50)
    
    print(f"Using sample: {len(df)}")
    
    X = df["cleanedText"]
    y = df["rating"]
    
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size = 0.2, random_state = 50)
    
    vectorizer = TfidfVectorizer(max_features = 10000, ngram_range = (1, 2))
    XTrainVec = vectorizer.fit_transform(XTrain)
    XTestVec = vectorizer.transform(XTest)
    
    print("Training Logistic Regression: ")
    lr = LogisticRegression( max_iter = 1000, random_state = 50, class_weight = "balanced")
    lr.fit(XTrainVec, yTrain)
    
    yPred = lr.predict(XTestVec)
    
    accuracy = accuracy_score(yTest, yPred)
    print(f"\nLogistic Regression Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(yTest, yPred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(yTest, yPred))
    
    with open( "models/logistic_regression_model.pkl", "wb") as f:
        pickle.dump(lr, f)
    
    with open("models/tfidf_vectorizer_lr.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    
    print("\nModel saved to models/logistic_regression_model.pkl")

if __name__ == "__main__":
    trainLogisticRegression()