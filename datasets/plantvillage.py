import os.path as osp
import pickle
import json

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

@DATASET_REGISTRY.register()
class PlantVillage(DatasetBase):
    dataset_dir = "plantvillage"

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))


        self.dataset_dir = osp.join(root, "plantvillage/huggingface_version")

        self.image_dir = self.dataset_dir

        print(f"  ROOT: {root}")
        print(f"  dataset_dir: {self.dataset_dir}")
        print(f"  image_dir: {self.image_dir}")

        if not osp.exists(self.image_dir):
            raise FileNotFoundError(
                f" Image directory not found: {self.image_dir}\n"
                f"Please check if the data is at the correct location."
            )

        print(f" Found image directory!\n")

        self.split_path = osp.join(self.dataset_dir, "split_plantvillage_official_color.json")
        self.split_fewshot_dir = osp.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if osp.exists(self.split_path):
            train, val, test = self.read_split(self.split_path, self.image_dir)
        else:
            raise FileNotFoundError(
                f"Official PlantVillage split not found: {self.split_path}"
            )

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

        num_classes = len(set([d.classname for d in train]))
        print(f"\n{'='*60}")
        print(f"PlantVillage Dataset Summary:")
        print(f"  #train={len(train)}, #val={len(val)}, #test={len(test)}")
        print(f"  #classes={num_classes}")
        print(f"  split_path={self.split_path}")
        print(f"{'='*60}\n")

        assert num_classes == 38, f"Expected 38 classes, but found {num_classes} classes!"

        super().__init__(train_x=train, val=val, test=test)

    def read_split(self, split_path, image_dir):
        print(f"\nReading PlantVillage split from {split_path}...")

        if not osp.exists(split_path):
            raise FileNotFoundError(f"Split file not found: {split_path}")

        with open(split_path, "r") as f:
            split_dict = json.load(f)


        if "val" not in split_dict:
            raise ValueError(
                f"Split file is in old format (missing 'val' key). "
                f"Please delete {split_path} and re-run to regenerate."
            )

        train, val, test = [], [], []

        for rel_path, label, classname in split_dict["train"]:
            item = Datum(impath=osp.join(image_dir, rel_path), label=label, classname=classname)
            train.append(item)

        for rel_path, label, classname in split_dict["val"]:
            item = Datum(impath=osp.join(image_dir, rel_path), label=label, classname=classname)
            val.append(item)

        for rel_path, label, classname in split_dict["test"]:
            item = Datum(impath=osp.join(image_dir, rel_path), label=label, classname=classname)
            test.append(item)

        print(f"  Train: {len(train):6d} samples")
        print(f"  Val:   {len(val):6d} samples")
        print(f"  Test:  {len(test):6d} samples")

        return train, val, test
