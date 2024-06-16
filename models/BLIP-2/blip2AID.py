
import random
import pickle
import matplotlib.pyplot as plt
import torch
from PIL import Image
import os
import glob
from lavis.models import load_model_and_preprocess

from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix


random.seed(12345)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, vis_processors, _ = load_model_and_preprocess(name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True, device=device)

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
    # model, vis_processors, _ = load_model_and_preprocess(name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True, device=device)
    # ask a random question.
    # question = "What scene is this aerial image? Return one word."
    # question = "This is an aerial image of which of the following scenes: Annual Crop, Forest, Herbaceous Vegetation, Highway, Industrial, Pasture, Permanent Crop, Residential, River, or Sea Lake?"
    question = "This image belongs to which one of the following classification: airport, bare land, baseball field, beach, bridge, center, church, commercial, dense residential, desert, farmland, forest, industrial, meadow, medium residential, mountain, park, parking, playground, pond, port, railway station, resort, river, school, sparse residential, square, stadium, storage tanks, viaduct?"
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    res = model.generate({"image": image, "prompt": f"Question: {question} Answer:"})
    # ['singapore']
    return res[0].split(",")[0]


def visualize_accuracy(labels, predicted_classes, categories, i):
    # Calculate accuracy for each category
    category_accuracies = {category: 0 for category in categories}
    for label, predicted_class in zip(labels, predicted_classes):
        if label == predicted_class:
            category_accuracies[label] += 1
    # Normalize accuracy to get percentages
    total_images_per_category = len(labels) // len(categories)
    category_accuracies = {category: accuracy / total_images_per_category * 100 for category, accuracy in category_accuracies.items()}

    # Create a bar plot
    plt.figure(figsize=(10, 6))  # Adjust figure size as needed
    plt.bar(category_accuracies.keys(), category_accuracies.values(), color='blue')
    plt.xlabel('Categories')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy for Each Category')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels and align them to the right
    plt.tight_layout()
    plt.savefig(f'/home/xm56/BLIP/fig/AID/accuracy{i}.png')

    return category_accuracies

def plot_confusion_matrix(actual_classes, predicted_classes, i):
    # Ensure unique labels are sorted or in a consistent order if necessary
    labels = sorted(set(predicted_classes + actual_classes))
    labels = [label[:20] for label in labels]  # Truncate label to no more than 20 characters

    num_classes = len(labels)
    figsize = (min(20, num_classes * 0.5), min(20, num_classes * 0.5))  # Adjust based on number of classes

    accuracy = accuracy_score(actual_classes, predicted_classes)

    # plt.figure(figsize=figsize)  # Set the figure size
    cmp = ConfusionMatrixDisplay.from_predictions(actual_classes, predicted_classes, display_labels=labels, xticks_rotation='vertical')


    fig, ax = plt.subplots(figsize=figsize)
    cmp.plot(ax=ax)
    plt.setp(ax.get_xticklabels(), rotation=90)
    ax.set_title(f"Confusion Matrix {i} - Accuracy: {accuracy:0,.2f}")
    fig.savefig(f'/home/xm56/BLIP/fig/AID/confusionmatrix{i}.png')
    # fig.show()
    return accuracy

def main():
    sample_size = 2000 # minmum is 10 to get 1 image in each category
    seed = 123
    random.seed(seed)
    dataset_folder = '/scratch/public/aid/test' 
    results_folder = "/home/xm56/BLIP/results/AID"

    # Get random images
    random_images = get_random_images(dataset_folder, sample_size)
    print(len(random_images))
    # Get labels
    labels = [img.split('/')[-2].lower().replace('_', '') for img in random_images]

    # Get predicted classes from GPT-4
    predicted_classes = [get_image_description(img).replace(' ', '') for img in random_images]
    print("Predicted:", predicted_classes)
    print("Acutal:", labels)

    correct_count = sum(labels == predicted_classes for labels, predicted_classes in zip(labels, predicted_classes))
    accuracy = correct_count/sample_size
    print("Sample Size:", sample_size)
    print("Accuracy:", accuracy)

     # Save in dictionary
    response_dict = dict(zip(random_images, predicted_classes))
    with open(results_folder + f"/response_dict{seed}.pkl", 'wb') as file:
        pickle.dump(response_dict, file)

    

    # Visualize accuracy for each category
    categories = set(labels)
    category_accuracy = visualize_accuracy(labels, predicted_classes, categories, seed)

    accuracy_conf = plot_confusion_matrix(labels, predicted_classes, seed)
    print("accuracy_conf:", accuracy_conf)

    with open(results_folder + f"/cat_acc{seed}.pkl", 'wb') as file:
        pickle.dump(category_accuracy, file)
   

    # dataset_folder = '/scratch/public/aid/test' 
    
    # sample_size = 100 # minmum is 10 to get 1 image in each category

    # # Get random images
    # random_images = get_random_images(dataset_folder, sample_size)
    # # Get labels 
    # labels = [img.split('/')[-2] for img in random_images]
    
    # # Get predicted classes from GPT-4
    # predicted_classes = [get_image_description(img).replace(' ', '') for img in random_images]
    # print("Predicted:", predicted_classes)
    # print("Acutal:", labels)

    # # Save in dictionary
    # response_dict = dict(zip(random_images, predicted_classes))
    # with open('results/response_dict_aid.pkl', 'wb') as file:
    #     pickle.dump(response_dict, file)

    # correct_count = sum(labels == predicted_classes for labels, predicted_classes in zip(labels, predicted_classes))
    # accuracy = correct_count/sample_size
    # print("Sample Size:", sample_size)
    # print("Accuracy:", accuracy)

    # # Visualize accuracy for each category
    # categories = set(labels)
    # category_accuracy = visualize_accuracy(labels, predicted_classes, categories)
    # # with open('results/category_acc.pkl', 'wb') as file:
    # #     pickle.dump(category_accuracy, file)

if __name__ == "__main__":
    main()