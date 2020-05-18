import streamlit as st
import matplotlib
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score


def main():
    st.title("Binary Classification Web App")
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Are your mushrooms edible or poisonous? :mushroom:")
    st.sidebar.markdown("Are your mushrooms edible or poisonous? :mushroom:")

    @st.cache(persist=True)
    def load_data():
        data = pd.read_csv('mushrooms.csv')
        label = LabelEncoder()
        # data = pd.get_dummies(data, drop_first = True)
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data

    @st.cache(persist=True)
    def split(df):
        y = df.type
        x = df.drop(columns=["type"])
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test

    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model, x_test, y_test,
                                  display_labels=class_names)
            st.pyplot()

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            plot_roc_curve(model, x_test, y_test)
            st.pyplot()
        if 'Precision-Recall' in metrics_list:
            st.subheader("Precision Recall")
            plot_precision_recall_curve(model, x_test, y_test)
            st.pyplot()

    df = load_data()
    x_train, x_test, y_train, y_test = split(df)
    class_names = ["Edible", "Poisionous"]

    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox(
        "classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))

    if classifier == 'Support Vector Machine (SVM)':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input(
            "C (Regularization Parameter)", 0.01, 10.0, step=0.01, key="C_SVM")
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key="kernel")
        gamma = st.sidebar.radio(
            "Gamma (Kernel Coefficients)", ("scale", "auto"), key="gamma")

        metrics = st.sidebar.multiselect(
            "Metrics", ("Confusion Matrix", "ROC Curve", "Precision-Recall"), key="metrics")

        if st.sidebar.button("Classify", key='classify'):
            st.subheader(classifier + " Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(
                y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(
                y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)

    if classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input(
            "C (Regularization Parameter)", 0.01, 10.0, step=0.01, key="C_LR")
        max_iter = st.sidebar.slider(
            "Maximum Iterations", 100, 500, key="max_iter")

        metrics = st.sidebar.multiselect(
            "Metrics", ("Confusion Matrix", "ROC Curve", "Precision-Recall"), key="metrics")

        if st.sidebar.button("Classify", key='classify'):
            st.subheader(classifier + " Results")
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(
                y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(
                y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)

    if classifier == 'Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input(
            "Number of Trees", 100, 5000, step=10, key="n_estimators")
        max_depth = st.sidebar.number_input(
            "The Maximum Depth of the Tree", 5, 20, step=1, key="max_depth")
        bootstrap = st.sidebar.radio(
            "Bootstrap Samples when building Trees", ("True", "False"), key="bootstrap")

        metrics = st.sidebar.multiselect(
            "Metrics", ("Confusion Matrix", "ROC Curve", "Precision-Recall"), key="metrics")

        if st.sidebar.button("Classify", key='classify'):
            st.subheader(classifier + " Results")
            model = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(
                y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(
                y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)

    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom Dataset (Classification)")
        st.write(df)

    st.write("- By Pallav Bhardwaj")


if __name__ == '__main__':
    main()
