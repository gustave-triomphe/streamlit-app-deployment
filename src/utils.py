import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from palmerpenguins import load_penguins


def clean_data(df: pd.DataFrame):
    df = df.dropna()
    return df

def select_data(df: pd.DataFrame):
    X = df[["bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g"]]
    y = df["sex"]
    return X, y


def train_rf(X: pd.DataFrame, y: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test,y_pred,labels=clf.classes_)
    return clf, report, cm, clf.classes_


def plot_confusion_matrix(cm, classes):
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig