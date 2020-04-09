import numpy as np
from matplotlib import image
from matplotlib import pyplot
import re


def load_data(train_percent=0.94, relative_path='Images/', photo_format='.jpg', size=750):
    male_train_data = []
    female_train_data = []
    male_test_data = []
    female_test_data = []
    male_labels = []
    female_labels = []
    # male_train_labels = []
    # female_train_labels = []
    # male_test_labels = []
    # female_test_labels = []
    train_size = size * train_percent

    for i in range(size):
        image_male = image.imread(relative_path + 'CM' + str(i + 1) + photo_format)
        image_female = image.imread(relative_path + 'CF' + str(i + 1) + photo_format)
        # if i + 1 < train_size:
        male_train_data.append(image_male)
        female_train_data.append(image_female)
        # else:
        #     male_test_data.append(image_male)
        #     female_test_data.append(image_female)

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

    # for i in range(size):
    #     if i + 1 < train_size:
    #         male_train_labels.append(male_labels[i])
    #         female_train_labels.append(female_labels[i])
    #     else:
    #         male_test_labels.append(male_labels[i])
    #         female_test_labels.append(female_labels[i])
    #
    male_train_data = np.asarray(male_train_data)
    female_train_data = np.asarray(female_train_data)
    # male_test_data = np.asarray(male_test_data)
    # female_test_data = np.asarray(female_test_data)
    #
    male_labels = np.asarray(male_labels, dtype=np.float64)
    female_labels = np.asarray(female_labels, dtype=np.float64)
    # male_test_labels = np.asarray(male_test_labels, dtype=np.float64)
    # female_test_labels = np.asarray(female_test_labels, dtype=np.float64)

    return male_train_data, female_train_data, male_labels, female_labels
    #return male_train_data, female_train_data, male_test_data, female_test_data, male_train_labels, female_train_labels, male_test_labels, female_test_labels

# male_train_data, female_train_data, male_test_data, female_test_data, male_labels, female_labels = load_data()
# print(np.shape(male_train_data))
