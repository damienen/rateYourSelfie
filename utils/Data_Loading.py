import numpy as np
from matplotlib import image
from matplotlib import pyplot
import re


def load_data(train_percent=0.94, relative_path='../Images/', photo_format='.jpg', size=750):

    male_train_data = []
    female_train_data = []
    male_test_data = []
    female_test_data = []
    male_labels = []
    female_labels = []
    train_size = size * train_percent

    for i in range(size):
        image_male = image.imread(relative_path + 'CM' + str(i + 1) + photo_format)
        image_female = image.imread(relative_path + 'CF' + str(i + 1) + photo_format)
        if i + 1 < train_size:
            male_train_data.append(image_male)
            female_train_data.append(image_female)
        else:
            male_test_data.append(image_male)
            female_test_data.append(image_female)

    label_file = open(relative_path + 'All_labels.txt', 'r')
    labels = label_file.read()
    label_file.close()
    labels = re.split(' |\n', labels)
    labels.pop(size * 4)

    for i in range(len(labels) - 2, -1, -2):
        labels.pop(i)

    for i in range(0, len(labels)):
        if i < len(labels) / 2:
            female_labels.append(labels[i])
        else:
            male_labels.append(labels[i])

    return male_train_data, female_train_data, male_test_data, female_test_data, male_labels, female_labels


# male_train_data, female_train_data, male_test_data, female_test_data, male_labels, female_labels = load_data()
# print(np.shape(male_train_data))
