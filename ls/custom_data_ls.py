import ls

from torch.utils.data import Dataset
from ultralytics.data.dataset import YOLODataset


data = YOLODataset(data="aquarium.yaml")


train_data, test_data, train_indices, test_indices, splitter = ls.learning_to_split(
    data,
    model={"name": "mlp"},
    return_order=[
        "train_data",
        "test_data",
        "train_indices",
        "test_indices",
        "splitter",
    ],
)
# splitter:                    The learned splitter (torch.nn.Module)
# train_data, test_data:       The training and testing dataset (torch.utils.data.Dataset).
# train_indices, test_indices: The indices of the training/testing examples in the original dataset (list[int])
