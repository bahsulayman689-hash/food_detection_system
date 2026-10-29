"""
Pulls labeled bounding-box data for ~49 food classes from Google's Open
Images V7 dataset and exports it straight into the YOLO folder structure
train.py expects. This is how you go from COCO's 10 food classes to a
real, broad food detector without labeling thousands of images by hand.

Install (separate from requirements.txt — fiftyone is only needed for
this one-time download step):
    pip install fiftyone

Usage:
    python download_openimages_food.py --samples-per-class 150

This downloads up to N images per class (train + a smaller val split),
exports to ./dataset/ in YOLO format, and writes a matching data.yaml
with normalized class names that line up with calorie_db.json.

First run downloads several GB — expect it to take a while depending on
your connection. Lower --samples-per-class to start smaller and iterate
faster (even 40-60/class is enough to sanity-check the pipeline).
"""
import argparse
import shutil
from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz
import yaml

# Open Images V7 class names -> normalized key used in calorie_db.json.
# Normalized = lowercase, spaces -> underscores, some renamed to match
# common usage / to line up with COCO's pretrained class names so the
# same calorie_db.json works for both the pretrained quick-test model
# and your custom-trained one.
CLASS_MAP = {
    "Apple": "apple",
    "Banana": "banana",
    "Orange": "orange",
    "Mango": "mango",
    "Watermelon": "watermelon",
    "Grape": "grape",
    "Pineapple": "pineapple",
    "Strawberry": "strawberry",
    "Lemon": "lemon",
    "Pear": "pear",
    "Bread": "bread",
    "Bagel": "bagel",
    "Croissant": "croissant",
    "Pancake": "pancake",
    "Waffle": "waffle",
    "Muffin": "muffin",
    "Doughnut": "donut",
    "Cookie": "cookie",
    "Cake": "cake",
    "Pastry": "pastry",
    "Pretzel": "pretzel",
    "Pizza": "pizza",
    "Hamburger": "burger",
    "Hot dog": "hot_dog",
    "Sandwich": "sandwich",
    "Submarine sandwich": "submarine_sandwich",
    "Taco": "taco",
    "Burrito": "burrito",
    "Sushi": "sushi",
    "French fries": "fries",
    "Egg (Food)": "egg",
    "Cheese": "cheese",
    "Popcorn": "popcorn",
    "Guacamole": "guacamole",
    "Salad": "salad",
    "Pasta": "pasta",
    "Tomato": "tomato",
    "Potato": "potato",
    "Carrot": "carrot",
    "Broccoli": "broccoli",
    "Cucumber": "cucumber",
    "Mushroom": "mushroom",
    "Bell pepper": "bell_pepper",
    "Cabbage": "cabbage",
    "Shrimp": "shrimp",
    "Coffee": "coffee",
    "Juice": "juice",
    "Ice cream": "ice_cream",
    "Milk": "milk",
}


def export_split(split: str, samples_per_class: int, out_dir: Path):
    print(f"\nDownloading Open Images V7 '{split}' split for {len(CLASS_MAP)} classes...")
    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split=split,
        label_types=["detections"],
        classes=list(CLASS_MAP.keys()),
        max_samples=samples_per_class * len(CLASS_MAP),
        dataset_name=f"food-openimages-{split}",
        shuffle=True,
    )

    export_dir = out_dir / "images" / ("train" if split == "train" else "val")
    labels_dir = out_dir / "labels" / ("train" if split == "train" else "val")
    export_dir.parent.mkdir(parents=True, exist_ok=True)
    labels_dir.parent.mkdir(parents=True, exist_ok=True)

    dataset.export(
        export_dir=str(out_dir),
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="detections",
        classes=list(CLASS_MAP.keys()),
        split="train" if split == "train" else "val",
    )
    fo.delete_dataset(dataset.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-class", type=int, default=150,
                         help="Max images per class for train split (val uses 20%% of this)")
    parser.add_argument("--out", default="dataset")
    args = parser.parse_args()

    out_dir = Path(args.out)
    if out_dir.exists():
        print(f"'{out_dir}' already exists — remove it first if you want a clean re-download.")
        return

    export_split("train", args.samples_per_class, out_dir)
    export_split("validation", max(1, args.samples_per_class // 5), out_dir)

    # Normalize class names in the exported dataset.yaml to match calorie_db.json keys
    dataset_yaml_path = out_dir / "dataset.yaml"
    if dataset_yaml_path.exists():
        with open(dataset_yaml_path) as f:
            cfg = yaml.safe_load(f)
        raw_names = cfg.get("names", {})
        normalized = {
            idx: CLASS_MAP.get(name, name.lower().replace(" ", "_"))
            for idx, name in (raw_names.items() if isinstance(raw_names, dict) else enumerate(raw_names))
        }
        cfg["names"] = normalized
        cfg["path"] = "."
        cfg["train"] = "images/train"
        cfg["val"] = "images/val"
        with open(out_dir / "data.yaml", "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        dataset_yaml_path.unlink()

    print(f"\nDone. Dataset at ./{out_dir}/, config at ./{out_dir}/data.yaml")
    print("Next: python train.py --data dataset/data.yaml --model yolo26s.pt --epochs 100")


if __name__ == "__main__":
    main()
