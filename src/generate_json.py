import os
import json

def generate_json(dataset_dir, split):
    data = []

    categories = sorted(os.listdir(dataset_dir))
    for i, category in enumerate(categories):
        category_dir = os.path.join(dataset_dir, category)
        image_files = sorted(os.listdir(category_dir))
        for image_file in image_files:
            data.append([os.path.join(category, image_file), i, category])

    return data

def main():
    dataset_dir = 'aid'
    splits = ['train', 'val', 'test']
    dataset = {}

    for split in splits:
        dataset[split] = generate_json(dataset_dir, split)

    with open('split_aid.json', 'w') as json_file:
        json.dump(dataset, json_file, indent=4)

if __name__ == "__main__":
    main()