import os

from src.utils import Datum, DatasetBase, read_json, write_json, build_data_loader

template = ["a centered satellite photo of {}."]


def read_split(filepath, path_prefix):
    def _convert(items):
        out = []
        for impath, label, classname in items:
            impath = os.path.join(path_prefix, impath)
            item = Datum(impath=impath, label=int(label), classname=classname)
            out.append(item)
        return out

    print(f"Reading split from {filepath}")
    split = read_json(filepath)
    train = _convert(split["train"])
    val = _convert(split["val"])
    test = _convert(split["test"])

    return train, val, test


class AID(DatasetBase):

    dataset_dir = "AID"

    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "aid")
        self.split_path = os.path.join(self.dataset_dir, "split_aid.json")

        self.template = template

        train, val, test = read_split(self.split_path, self.image_dir)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=val, test=test)

    def update_classname(self, dataset_old):
        dataset_new = []
        for item_old in dataset_old:
            cname_old = item_old.classname
            cname_new = cname_old
            item_new = Datum(
                impath=item_old.impath, label=item_old.label, classname=cname_new
            )
            dataset_new.append(item_new)
        return dataset_new
