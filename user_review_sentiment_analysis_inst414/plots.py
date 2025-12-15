import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.model_selection import train_test_split

with open("models/logistic_regression_model.pkl", "rb") as f:
    trainedModel = pickle.load(f)

with open("models/tfidf_vectorizer_lr.pkl", "rb") as f:
    savedVectorizer = pickle.load(f)

allReviews = pd.read_csv("data/processed/combined_cleaned.csv")
allReviews = allReviews.dropna(subset = ["cleanedText"])
allReviews = allReviews[allReviews["cleanedText"] != ""]
allReviews = allReviews.sample(n = 500000, random_state = 50)

reviewText = allReviews["cleanedText"]
actualRatings = allReviews["rating"]

trainingText, testText, trainingRatings, testRatings = train_test_split(reviewText, actualRatings, test_size = 0.2, random_state = 50)

testTextAsNumbers = savedVectorizer.transform(testText)

predictedRatings = trainedModel.predict(testTextAsNumbers)

modelCoefficients = trainedModel.coef_[0]
allFeatureNames = savedVectorizer.get_feature_names_out()
top15Positions = np.argsort(np.abs(modelCoefficients))[-15:]

plt.figure(figsize = (10, 6))
plt.barh(range(len(top15Positions)), modelCoefficients[top15Positions])
plt.yticks(range(len(top15Positions)), allFeatureNames[top15Positions])
plt.xlabel("Coefficient Value")
plt.title("Top 15 Most Important Features")
plt.tight_layout()
plt.savefig(r"C:\Users\Mrhue\OneDrive\Desktop\INST414\Sprint3Plots\feature_importance.png")
plt.close()

confusionMatrix = confusion_matrix(testRatings, predictedRatings)
plt.figure(figsize = (8, 6))
sns.heatmap(confusionMatrix, annot = True, fmt = "d", cmap = "Blues")
plt.xlabel("Predicted Rating")
plt.ylabel("Actual Rating")
plt.title("Confusion Matrix - Weighted Logistic Regression")
plt.savefig(r"C:\Users\Mrhue\OneDrive\Desktop\INST414\Sprint3Plots\confusion_matrix.png")
plt.close()

modelList = ["Baseline", "LR Unweighted", "LR Weighted", "RF Unweighted"]
modelAccuracies = [0.65, 0.7682, 0.7090, 0.8054]

plt.figure(figsize = (10, 6))
plt.bar(modelList, modelAccuracies)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.ylim(0, 1)
plt.savefig(r"C:\Users\Mrhue\OneDrive\Desktop\INST414\Sprint3Plots\model_comparison.png")
plt.close()

print("All plots saved!")