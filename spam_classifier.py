import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

data = {
    "message": [
        "Win a free lottery ticket now",
        "Congratulations you won a prize",
        "Claim your free reward",
        "Free gift card available now",
        "Urgent you have won cash",
        "Meeting tomorrow at 10 AM",
        "Project deadline next week",
        "Let's have lunch tomorrow",
        "Team meeting today",
        "Please review the project document"
    ],
    "label": [
        "spam","spam","spam","spam","spam",
        "ham","ham","ham","ham","ham"
    ]
}

df = pd.DataFrame(data)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["message"])

y = df["label"]

model = MultinomialNB()
model.fit(X, y)

print("Spam Email Classifier Ready!")
print("-----------------------------")

email = input("Enter a message: ")

email_vector = vectorizer.transform([email])

prediction = model.predict(email_vector)

print("\nPrediction:", prediction[0])