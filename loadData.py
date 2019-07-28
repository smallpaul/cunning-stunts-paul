import csv
import numpy as np
import random as rand
import pathlib

train_location = '/home/paul/PycharmProjects/recursion-cellular-image-classification/train.csv'
data_location = '/home/paul/PycharmProjects/recursion-cellular-image-classification/train/'
data = []
train_data = []
test_data = []
current_index = 0


def load_data():
    global data
    with open(train_location) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        first_line = True
        train_mat = []
        for row in csv_reader:
            if first_line:
                first_line = False
            else:
                train_mat = train_mat + [row]
    data = train_mat


def configure_unsequenced():
    global train_data, test_data
    train_denominator = 20 # 1/20 is 0.05 so 5% of the data is going to be used for validation
    for index, row in enumerate(data):
        if index % train_denominator is not 0: # use mode to get an even spread of test and training data
            train_data.append(row)
        else:
            test_data.append(row)

    rand.shuffle(train_data)
    rand.shuffle(test_data)


def get_unsequenced_data():
    if len(data) == 0:
        load_data()
    if len(train_data) == 0:
        configure_unsequenced()
    x = []
    y = []
    for row in train_data:
        y.append(int(row[4]))
        y.append(int(row[4]))
        week_num_index = row[0].index('_')
        image_name = row[3] + '_s1_w' + row[0][week_num_index+1:week_num_index+2] + '.png'
        path = data_location + row[1] + '/Plate' + row[2] + '/' + image_name
        x.append(path)
        image_name = row[3] + '_s2_w' + row[0][week_num_index+1:week_num_index+2] + '.png'
        path = data_location + row[1] + '/Plate' + row[2] + '/' + image_name
        x.append(path)

    return x, y


def get_unsequenced_test():
    if len(data) == 0:
        load_data()
    if len(test_data) == 0:
        configure_unsequenced()
    x = []
    y = []
    for row in test_data:
        y.append(int(row[4]))
        week_num_index = row[0].index('_')
        image_name = row[3] + '_s1_w' + row[0][week_num_index+1:week_num_index+2] + '.png'
        path = data_location + row[1] + '/Plate' + row[2] + '/' + image_name
        x.append(path)

    return x, y


def get_unsequenced_batch(size):
    global current_index
    if len(data) == 0:
        load_data()
    if len(train_data) < (current_index + size):
        configure_unsequenced()
    x = []
    y = []
    for index in range(size):
        x.append(train_data[index + current_index][0])
        y.append(train_data[index + current_index][4])
    current_index = current_index + size

    return x, y


def get_unsequenced_test_batch(size):
    if len(data) == 0:
        load_data()
    if len(test_data) < (current_index + size):
        configure_unsequenced()
    x = []
    y = []
    for index in range(size):
        x.append(train_data[index][0])
        y.append(train_data[index][4])

    return x, y


def get_randy_batch(batch_size):
    all_names = []
    all_targets = []
    for _ in range(batch_size):
        image_names, targets = get_random_sequence()
        all_names.append(image_names)
        all_targets.append(targets)
    all_names = np.array(all_names)
    all_targets = np.array(all_targets)

    return all_names, all_targets


def get_random_sequence():
    global data
    tests = get_tests()
    plates = get_classes(2)
    samples = get_classes(3)
    test = tests[rand.randint(0, len(tests)-1)]
    plate = plates[rand.randint(0, len(plates)-1)]
    sample = samples[rand.randint(0, len(samples)-1)]

    return get_sequence(test, plate, sample)


def get_sequence(test, plate, sample):
    global data
    image_names = []
    targets = []
    for row in data:
        if test in row[1] and plate in row[2] and sample in row[3]:
            image_names.append(row[0])
            targets.append(row[4])

    return image_names, targets


def get_tests():
    global data
    tests = []
    for row in data:
        dash_index = row[1].index('-')
        test_name = row[1][:dash_index]
        if test_name not in tests:
            tests.append(test_name)

    return tests


def get_classes(col):
    global data
    classes = []
    for row in data:
        if row[col] not in classes:
            classes.append(row[col])

    return classes


def reshape_by_category(mat, column):
    col_number = -1
    col_name_list = []
    new_mat = []
    for row in mat:
        if row[column] not in col_name_list:
            col_number = col_number + 1
            col_name_list.append(row[column])
            row.remove(row[column])
            new_mat.append([row])
        else:
            col_index = col_name_list.index(row[column])
            row.remove(row[column])
            new_mat[col_index].append(row)
    return new_mat


if __name__ == '__main__':
    image_data, target = get_unsequenced_data()
    print(image_data)
    print(target)