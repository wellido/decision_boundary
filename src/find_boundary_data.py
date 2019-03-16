from keras.models import load_model
import numpy as np
import random


def search_in_order(model, select_first, select_second, coefficient, iterations):
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
    save_coefficient = coefficient
    y_previous = y_second
    bias = 1 / iterations
    for i in range(iterations):
        coefficient = coefficient + bias
        print("coefficient: ", coefficient)
        synthetic_data = np.clip(coefficient * select_first + (1 - coefficient) * select_second, 0.0, 1.0)
        y_synthetic = model.predict(synthetic_data).argmax(axis=-1)[0]
        print("prediction: ", y_synthetic)
        if y_synthetic == y_first and y_previous == y_second:
            save_coefficient = coefficient
            break
        else:
            y_previous = y_synthetic
            continue
    print("save coefficient: ", save_coefficient)




def search_by_dichotomize(model, select_first, select_second, coefficient, coefficient_plus, final_plus, iterations):
    """

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
            break
        if stop_flag > 5:
            break

    back_coefficient = coefficient - final_plus
    front_coefficient = coefficient + final_plus
    back_data = np.clip(back_coefficient * select_first + (1 - back_coefficient) * select_second, 0.0, 1.0)
    front_data = np.clip(front_coefficient * select_first + (1 - front_coefficient) * select_second, 0.0, 1.0)
    print("front prediction: ", model.predict(front_data).argmax(axis=-1)[0])
    print("back prediction: ", model.predict(back_data).argmax(axis=-1)[0])
    print(back_data)


if __name__ == '__main__':
    coefficient = 0.00
    coefficient_plus = 0.50
    model = load_model("../model/lenet-5.h5")
    data_first_list = np.load("../data/class_0.npz")["x_train"]
    data_second_list = np.load("../data/class_1.npz")["x_train"]
    select_first = data_first_list[random.randint(0, len(data_first_list))]
    select_second = data_second_list[random.randint(0, len(data_second_list))]
    select_first = select_first / 255
    select_second = select_second / 255
    select_first = select_first.reshape(1, 28, 28, 1)
    select_second = select_second.reshape(1, 28, 28, 1)
    search_in_order(model, select_first, select_second, coefficient, 2000)






