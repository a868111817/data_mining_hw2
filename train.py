import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def Logistic_Regression(X_train, X_test, y_train, y_test):  # Logistic Regression
    clf_logreg = LogisticRegression(random_state=0)
    clf_logreg.fit(X_train, y_train)
    y_pred = clf_logreg.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print('Logistic Regression Accuracy :', score)


def Decision_Tree(X_train, X_test, y_train, y_test):  # Decision Tree Classifier
    clf_dt = DecisionTreeClassifier(random_state=0)
    clf_dt.fit(X_train, y_train)
    y_pred = clf_dt.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print('Decision Tree Accuracy :', score)


def Random_Forest(X_train, X_test, y_train, y_test):  # Random Forest Classifier
    clf_rf = RandomForestClassifier(
        n_estimators=100, max_depth=4, random_state=0)
    clf_rf.fit(X_train, y_train)
    y_pred = clf_rf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print('Random Forest Accuracy :', score)


def Support_Vector_Machine(X_train, X_test, y_train, y_test):
    clf_svc = svm.SVC(kernel='linear')
    clf_svc.fit(X_train, y_train)
    y_pred = clf_svc.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print('Support Vector Machine Accuracy :', score)


if __name__ == '__main__':

    df = pd.read_csv('train.csv')
    y = df['isReiner']
    x = df.drop(['isReiner'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # Logistic Regression
    Logistic_Regression(X_train, X_test, y_train, y_test)

    # Decision Tree Classifier
    Decision_Tree(X_train, X_test, y_train, y_test)

    # Random Forest Classifier
    Random_Forest(X_train, X_test, y_train, y_test)

    # Support Vector Machine
    Support_Vector_Machine(X_train, X_test, y_train, y_test)
