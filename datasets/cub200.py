

import os.path as osp
import pickle

from dassl.data.datasets import DATASET_REGISTRY, DatasetBase
from dassl.utils import mkdir_if_missing
from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class CUB200(DatasetBase):

    dataset_dir = "cub/CUB_200_2011"

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.image_dir = osp.join(self.dataset_dir, "images")


        self.split_path = osp.join(root, "cub", "split_zhou_CUB200.json")
        self.split_fewshot_dir = osp.join(root, "cub", "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if not osp.isfile(self.split_path):
            raise FileNotFoundError(f"Required fixed split not found: {self.split_path}")

        train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)


        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = osp.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")

            if osp.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as f:
                    data = pickle.load(f)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)

                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


        print(f"DEBUG CHECK: First class name is '{train[0].classname}'")
        print(f"DEBUG CHECK: Total train images: {len(train)}")
        print(f"DEBUG CHECK: Unique classes: {len(set([d.classname for d in train]))}")


        assert len(set([d.classname for d in train])) == 200, "Number of unique classes mismatch!"

        super().__init__(train_x=train, val=val, test=test)

