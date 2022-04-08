import re
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, roc_curve, confusion_matrix

# SYNTHESIS FUNCTION
def synthetise(model, X_train, X_test, y_train, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_train_score = model.predict_proba(X_train)
    y_test_score = model.predict_proba(X_test)
    model_name = type(model).__name__
    print(re.sub(r"(\w)([A-Z])", r"\1 \2", model_name).upper())
    print('====================')
    print('TRAIN dataset')
    print(f'Accuracy score: {accuracy_score(y_train, y_train_pred):.1%}')
    print(f'Recall score (macro): {recall_score(y_train, y_train_pred, average="macro"):.1%}')
    print(f'Precision score (macro): {precision_score(y_train, y_train_pred, average="macro"):.1%}')
    print(f'ROC_AUC score (macro): {roc_auc_score(y_train, y_train_score, average="macro", multi_class="ovo"):.1%}')
    print(f'Confusion matrix')
    print(confusion_matrix(y_train, y_train_pred))
    print('====================')
    print('TEST dataset')
    print(f'Accuracy score: {accuracy_score(y_test, y_test_pred):.1%}')
    print(f'Recall score (macro): {recall_score(y_test, y_test_pred, average="macro"):.1%}')
    print(f'Precision score (macro): {precision_score(y_test, y_test_pred, average="macro"):.1%}')
    print(f'ROC_AUC score (macro): {roc_auc_score(y_test, y_test_score, average="macro", multi_class="ovo"):.1%}')
    print(f'Confusion matrix')
    print(confusion_matrix(y_test, y_test_pred))