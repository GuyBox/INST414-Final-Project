import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

def trainRandomForest():
    df = pd.read_csv("data/processed/combined_cleaned.csv")
    
    print(f"Total reviews: {len( df )}")
    print(f"NaN in cleanedText: {df['cleanedText'].isna().sum()}")
    print(f"Empty strings: {(df['cleanedText'] == '').sum()}")
    
    df = df.dropna(subset = ["cleanedText"])
    df = df[df["cleanedText"] != ""]
    
    print(f"After cleaning: {len( df )}")
    
    X = df["cleanedText"]
    y = df["rating"]
    
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size = 0.2, random_state = 50)
    
    #weights words by their importance
    vectorizer = TfidfVectorizer(max_features = 10000, ngram_range = (1, 2))
    XTrainVec = vectorizer.fit_transform(XTrain)
    XTestVec = vectorizer.transform(XTest)
    
    print("Training Random Forest: ")
    rf = RandomForestClassifier( n_estimators = 50, random_state = 50, class_weight = "balanced" )
    rf.fit(XTrainVec, yTrain)
    
    yPred = rf.predict(XTestVec)
    
    accuracy = accuracy_score(yTest, yPred)
    print(f"\nRandom Forest Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report: ")
    print(classification_report( yTest, yPred))
    
    print("\nConfusion Matrix: ")
    print(confusion_matrix(yTest, yPred))
    

    with open( "models/random_forest_model.pkl", "wb") as f:
        pickle.dump(rf, f )
    
    with open("models/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    
    print("\nModel saved to models/random_forest_model.pkl")

if __name__ == "__main__":
    trainRandomForest()