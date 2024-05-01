from .EuroSAT import EuroSAT
from .AID import AID
from .RESISC45 import RESISC45


dataset_list = {
    "EuroSAT": EuroSAT,
    "AID": AID,
    "RESISC45": RESISC45,
}


def build_dataset(dataset, root_path, shots):
    return dataset_list[dataset](root_path, shots)
