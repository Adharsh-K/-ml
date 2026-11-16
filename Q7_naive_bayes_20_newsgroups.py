from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, f1_score

categories = ["comp.graphics", "rec.sport.baseball", "sci.space", "talk.politics.misc"]

train = fetch_20newsgroups(subset="train", categories=categories)
test = fetch_20newsgroups(subset="test", categories=categories)

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train.data)
X_test = vectorizer.transform(test.data)

y_train = train.target
y_test = test.target

model1 = MultinomialNB()
model1.fit(X_train, y_train)
pred1 = model1.predict(X_test)

model2 = BernoulliNB()
model2.fit(X_train, y_train)
pred2 = model2.predict(X_test)

print("Multinomial Naive Bayes")
print("Accuracy:", accuracy_score(y_test, pred1))
print("F1 Score:", f1_score(y_test, pred1, average="macro"))

print("Bernoulli Naive Bayes")
print("Accuracy:", accuracy_score(y_test, pred2))
print("F1 Score:", f1_score(y_test, pred2, average="macro"))
