import os
import random


def spilt_dataset(annotations_path, train_list_path, test_list_path, train_ratio=0.7):
    example_name_list = os.listdir(annotations_path)
    example_num = len(example_name_list)
    train_num = int(example_num * train_ratio)
    train_list = random.sample(range(example_num), train_num)

    f_train = open(train_list_path, 'w')
    f_test = open(test_list_path, 'w')

    for i in range(example_num):
        example_name = example_name_list[i][:-4] + '\n'
        if i in train_list:
            f_train.write(example_name)
        else:
            f_test.write(example_name)

    f_train.close()
    f_test.close()




