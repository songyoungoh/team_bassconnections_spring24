from .oxford_pets import OxfordPets
from .eurosat import EuroSAT
from .aid import AID
from .resisc45 import RESISC45
from .ucf101 import UCF101
from .sun397 import SUN397
from .caltech101 import Caltech101
from .dtd import DescribableTextures
from .fgvc import FGVCAircraft
from .food101 import Food101
from .oxford_flowers import OxfordFlowers
from .stanford_cars import StanfordCars


dataset_list = {
                "eurosat": EuroSAT,
                "aid": AID,
                "resisc45": RESISC45,
                }


def build_dataset(dataset, root_path, shots):
    return dataset_list[dataset](root_path, shots)