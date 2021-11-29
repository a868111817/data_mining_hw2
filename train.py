import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def LR(X_train, X_test, y_train, y_test):
    clf_logreg = LogisticRegression(random_state=0)
    clf_logreg.fit(X_train, y_train)
    y_pred = clf_logreg.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print('Accuracy :', score)


def DT(X_train, X_test, y_train, y_test):
    clf_dt = DecisionTreeClassifier(random_state=0)
    clf_dt.fit(X_train, y_train)
    y_pred = clf_dt.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print('Accuracy :', score)


if __name__ == '__main__':

    df = pd.read_csv('train.csv')
    y = df['isReiner']
    x = df.drop(['isReiner'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # Decision Tree Classifier
    DT(X_train, X_test, y_train, y_test)

    # Logistic Regression
    LR(X_train, X_test, y_train, y_test)
