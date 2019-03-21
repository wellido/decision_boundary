from keras.models import load_model
import numpy as np
import random
import argparse


def count_boundary_number(model, select_first, select_second, coefficient, iterations):
    """

    :param model:
    :param select_first:
    :param select_second:
    :param coefficient:
    :param iterations:
    :return:
    """
    y_first = model.predict(select_first).argmax(axis=-1)[0]
    y_second = model.predict(select_second).argmax(axis=-1)[0]
    y_previous = y_second
    bias = 1 / iterations
    return_count = 0
    for i in range(iterations):
        coefficient = coefficient + bias
        # print("coefficient: ", coefficient)
        synthetic_data = np.clip(coefficient * select_first + (1 - coefficient) * select_second, 0.0, 1.0)
        y_synthetic = model.predict(synthetic_data).argmax(axis=-1)[0]
        # print("prediction: ", y_synthetic)
        if y_synthetic == y_first and y_previous == y_second:
            return_count += 1
            y_previous = y_synthetic
        else:
            y_previous = y_synthetic
    print("boundary number: ", return_count)
    return return_count


def random_select_boundary(model_path, data_dir_path, label_num, data_num, coefficient, iterations):
    """

    :param model_path:
    :param data_dir_path:
    :param label_num:
    :param data_num:
    :param coefficient:
    :param iterations:
    :return:
    """
    model = load_model(model_path)
    total_boundary = 0
    for i in range(label_num - 1):
        for j in range(i + 1, label_num):
            first_data_path = data_dir_path + "class_" + str(i) + ".npz"
            second_data_path = data_dir_path + "class_" + str(j) + ".npz"
            data_first_list = np.load(first_data_path)["x_train"]
            data_second_list = np.load(second_data_path)["x_train"]
            for k in range(data_num):
                select_first = data_first_list[random.randint(0, len(data_first_list))]
                select_second = data_second_list[random.randint(0, len(data_second_list))]
                # print(select_first)
                select_first = select_first / 255
                select_second = select_second / 255
                select_first = select_first.reshape(1, 28, 28, 1)
                select_second = select_second.reshape(1, 28, 28, 1)
                total_boundary += count_boundary_number(model, select_first, select_second, coefficient, iterations)
    return total_boundary


def search_in_order(model, select_first, select_second, coefficient, iterations):
    """
    search data coefficient one by one
    :param model:
    :param select_first:
    :param select_second:
    :param coefficient:
    :param iterations:
    :return:
    """
    y_first = model.predict(select_first).argmax(axis=-1)[0]
    y_second = model.predict(select_second).argmax(axis=-1)[0]
    save_coefficient = coefficient
    y_previous = y_second
    bias = 1 / iterations
    success_flag = 0
    for i in range(iterations):
        coefficient = coefficient + bias
        print("coefficient: ", coefficient)
        synthetic_data = np.clip(coefficient * select_first + (1 - coefficient) * select_second, 0.0, 1.0)
        y_synthetic = model.predict(synthetic_data).argmax(axis=-1)[0]
        print("prediction: ", y_synthetic)
        if y_synthetic == y_first and y_previous == y_second:
            save_coefficient = coefficient
            success_flag = 1
            break
        else:
            y_previous = y_synthetic
            continue
    if success_flag == 1:
        print("save coefficient: ", save_coefficient)
        back_coefficient = coefficient - bias
        front_coefficient = coefficient + bias
        back_data = np.clip(back_coefficient * select_first + (1 - back_coefficient) * select_second, 0.0, 1.0)
        front_data = np.clip(front_coefficient * select_first + (1 - front_coefficient) * select_second, 0.0, 1.0)
        print("front prediction: ", model.predict(front_data).argmax(axis=-1)[0])
        print("back prediction: ", model.predict(back_data).argmax(axis=-1)[0])
        return front_data, back_data
    else:
        print("can't find data.")
        return False


def search_by_dichotomize(model, select_first, select_second, coefficient, coefficient_plus, final_plus, iterations):
    """
    search data using dichotomize
    :param model:
    :param select_first:
    :param select_second:
    :param coefficient:
    :param coefficient_plus:
    :param final_plus:
    :param iterations:
    :return:
    """

    final_plus = final_plus
    y_first = model.predict(select_first).argmax(axis=-1)[0]
    y_second = model.predict(select_second).argmax(axis=-1)[0]
    stop_flag = 0
    previous_prediction = y_first
    success_flag = 1
    for i in range(iterations):
        synthetic_data = np.clip(coefficient * select_first + (1 - coefficient) * select_second, 0.0, 1.0)
        prediction = model.predict(synthetic_data).argmax(axis=-1)[0]
        print("times: ", i)
        print("coefficient: ", coefficient)
        print("prediction: ", prediction)
        coefficient_plus = coefficient_plus / 2
        if prediction == y_first:
            if previous_prediction == y_second:
                stop_flag = 0
            else:
                stop_flag += 1
            coefficient = coefficient - coefficient_plus
            previous_prediction = y_first
        elif prediction == y_second:
            if previous_prediction == y_first:
                stop_flag = 0
            else:
                stop_flag += 1
            coefficient = coefficient + coefficient_plus
            previous_prediction = y_second
        else:
            success_flag = 0
            break
        if stop_flag > 5:
            break
    if success_flag == 1:
        back_coefficient = coefficient - final_plus
        front_coefficient = coefficient + final_plus
        back_data = np.clip(back_coefficient * select_first + (1 - back_coefficient) * select_second, 0.0, 1.0)
        front_data = np.clip(front_coefficient * select_first + (1 - front_coefficient) * select_second, 0.0, 1.0)
        print("front prediction: ", model.predict(front_data).argmax(axis=-1)[0])
        print("back prediction: ", model.predict(back_data).argmax(axis=-1)[0])
        print(back_data)
        return front_data, back_data
    else:
        print("can't find data.")
        return False


def search_all_data(model_path, first_label_path, second_label_path, coefficient, generate_num, save_path):
    """
    main runner
    :param model_path:
    :param first_label_path:
    :param second_label_path:
    :param coefficient:
    :param generate_num:
    :param save_path:
    :return:
    """

    model = load_model(model_path)
    data_first_list = np.load(first_label_path)["x_train"]
    data_second_list = np.load(second_label_path)["x_train"]

    # save data
    front_side_save_list = []
    back_side_save_list = []
    count = 0
    while count < generate_num:
        print("generate data number: ", count)
        select_first = data_first_list[random.randint(0, len(data_first_list) - 1)]
        select_second = data_second_list[random.randint(0, len(data_second_list) - 1)]
        select_first = select_first / 255
        select_second = select_second / 255
        select_first = select_first.reshape(1, 28, 28, 1)
        select_second = select_second.reshape(1, 28, 28, 1)
        synthetic_data = search_in_order(model, select_first, select_second, coefficient, 2000)
        if synthetic_data:
            front_data = synthetic_data[0] * 255
            front_data = front_data.astype(int)
            back_data = synthetic_data[1] * 255
            back_data = back_data.astype(int)
            front_side_save_list.append(front_data)
            back_side_save_list.append(back_data)
            count += 1
    np.savez(save_path, x_train=[front_side_save_list, back_side_save_list])


def get_arg_and_run():
    """
    get args from console
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        help="dnn model path")
    parser.add_argument("--first_label_path", type=str,
                        help="one side boundary label data path")
    parser.add_argument("--second_label_path", type=str,
                        help="the other side boundary label data path")
    parser.add_argument("--coefficient", type=float, default=0.00,
                        help="coefficient * data1 + (1 - coefficient) * data2")
    parser.add_argument("--generate_num", type=int,
                        help="generate data number")
    parser.add_argument("--save_path", type=str,
                        help="data save path")
    args = parser.parse_args()
    model_path = args.model_path
    first_label_path = args.first_label_path
    second_label_path = args.second_label_path
    coefficient = args.coefficient
    generate_num = args.generate_num
    save_path = args.save_path
    search_all_data(model_path, first_label_path, second_label_path, coefficient, generate_num, save_path)


if __name__ == '__main__':
    coefficient = 0.00
    coefficient_plus = 0.50
    model_path = "../model/lenet-5.h5"
    data_dir_path = "../data/original_data/"
    # model = load_model("../model/lenet-5.h5")
    # data_first_list = np.load("../data/original_data/class_0.npz")["x_train"]
    # data_second_list = np.load("../data/original_data/class_1.npz")["x_train"]
    # select_first = data_first_list[random.randint(0, len(data_first_list))]
    # select_second = data_second_list[random.randint(0, len(data_second_list))]
    # # print(select_first)
    # select_first = select_first / 255
    # select_second = select_second / 255
    # select_first = select_first.reshape(1, 28, 28, 1)
    # select_second = select_second.reshape(1, 28, 28, 1)
    # synthetic_data = search_in_order(model, select_first, select_second, coefficient, 5000)
    count = random_select_boundary(model_path, data_dir_path, 10, 20, coefficient, 5000)
    print(count)

    # search_all_data("../model/lenet-5.h5", "../data/original_data/class_0.npz", "../data/original_data/class_1.npz",
    #                 0.00, 1000, "../data/boundary_data/data_0&1.npz")

    # get_arg_and_run()

    # python find_boundary_data.py --model_path ../model/lenet-5.h5 --first_label_path ../data/original_data/class_0.npz --second_label_path ../data/original_data/class_1.npz --coefficient 0.00 --generate_num 10 --save_path ../data/boundary_data/data_0&1.npz






