import ls

data = ls.datasets.Tox21()
train_data, test_data = ls.learning_to_split(
    data, model={"name": "mlp"}, metric="roc_auc"
)
