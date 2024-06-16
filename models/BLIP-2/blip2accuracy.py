
import random
import pickle
import matplotlib.pyplot as plt
import torch
from PIL import Image
import os
import glob
from lavis.models import load_model_and_preprocess


random.seed(12)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# #get random images 
# def get_random_images(dataset_folder, sample_size=1):
#     all_images = []
#     categories = os.listdir(dataset_folder)
#     for category in categories:
#         category_path = os.path.join(dataset_folder, category)
#         images = [os.path.join(category_path, img) for img in os.listdir(category_path) if img.endswith(('.jpg', '.png'))]
#         all_images.extend(images)    
#     return random.sample(all_images, min(sample_size, len(all_images)))


def get_random_images(dataset_folder, sample_size=10):
    result_images = []
    categories = os.listdir(dataset_folder)
    category_size = max(1, sample_size // len(categories))

    for category in categories:
        category_path = os.path.join(dataset_folder, category)
        image_paths = glob.glob(os.path.join(category_path, '*.jpg')) + glob.glob(os.path.join(category_path, '*.png'))
        sampled_images = random.sample(image_paths, min(category_size, len(image_paths)))
        result_images.extend(sampled_images)

    return result_images

def get_image_description(image_path):
    raw_image = Image.open(image_path).convert('RGB')
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True, device=device)
    # ask a random question.
    # question = "What scene is this aerial image? Return one word."
    # question = "This is an aerial image of which of the following scenes: Annual Crop, Forest, Herbaceous Vegetation, Highway, Industrial, Pasture, Permanent Crop, Residential, River, or Sea Lake?"
    question = "This aerial image belongs to which one of the following classification: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake?"
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    res = model.generate({"image": image, "prompt": f"Question: {question} Answer:"})
    # ['singapore']
    return res[0]


def visualize_accuracy(labels, predicted_classes, categories):
    # Calculate accuracy for each category
    category_accuracies = {category: 0 for category in categories}
    for label, predicted_class in zip(labels, predicted_classes):
        if label == predicted_class:
            category_accuracies[label] += 1

    # Normalize accuracy to get percentages
    total_images_per_category = len(labels) // len(categories)
    category_accuracies = {category: accuracy / total_images_per_category * 100 for category, accuracy in category_accuracies.items()}

    # Create a bar plot
    plt.bar(category_accuracies.keys(), category_accuracies.values(), color='blue')
    plt.xlabel('Categories')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy for Each Category')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    plt.savefig('accuracy.png')

    return category_accuracies

def main():
    dataset_folder = '/data/scratch/public/eurosat/team_bassconnections_eurosat/Train_Test_Splits/test' 
    
    sample_size = 100 # minmum is 10 to get 1 image in each category

    # Get random images
    random_images = get_random_images(dataset_folder, sample_size)
    # Get labels 
    labels = [img.split('/')[-2] for img in random_images]
    
    # Get predicted classes from GPT-4
    predicted_classes = [get_image_description(img).replace(' ', '') for img in random_images]
    print("Predicted:", predicted_classes)
    print("Acutal:", labels)

    # Save in dictionary
    response_dict = dict(zip(random_images, predicted_classes))
    with open('results/response_dict100.pkl', 'wb') as file:
        pickle.dump(response_dict, file)

    correct_count = sum(labels == predicted_classes for labels, predicted_classes in zip(labels, predicted_classes))
    accuracy = correct_count/sample_size
    print("Sample Size:", sample_size)
    print("Accuracy:", accuracy)

    # Visualize accuracy for each category
    categories = set(labels)
    category_accuracy = visualize_accuracy(labels, predicted_classes, categories)
    with open('results/category_acc100.pkl', 'wb') as file:
        pickle.dump(category_accuracy, file)

if __name__ == "__main__":
    main()