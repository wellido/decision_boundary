import foolbox
from keras.models import load_model
import numpy as np
# import matplotlib.pyplot as plt
import argparse


def FGSM_attack(model, data, label):
    """

    :param model:
    :param data:
    :param label:
    :return:
    """
    fmodel = foolbox.models.KerasModel(model, bounds=(0, 1), channel_axis=1)
    attack = foolbox.attacks.FGSM(fmodel)
    adversarial = attack(data, label)
    return adversarial


def all_attcak(model_path, label, number, save_path):
    """

    :param model_path:
    :param label:
    :param number:
    :param save_path:
    :return:
    """
    model = load_model(model_path)
    data = np.load("../data/original_data/class_" + str(label) + ".npz")
    x_train = data["x_train"]
    index_select = np.random.choice(len(x_train), 2 * number, replace=False)
    count = 0
    save_list = []
    for index in index_select:
        x_attack = x_train[index].reshape(28, 28, 1) / 255
        print("original label: ", label)
        adversarial = FGSM_attack(model, x_attack, label)
        if adversarial is not None:
            adv_result = model.predict(adversarial.reshape(1, 28, 28, 1)).argmax(axis=-1)[0]
            if adv_result != label:
                print("adversary label: ", adv_result)
                count += 1
                adversarial = adversarial * 255
                adversarial = adversarial.astype(int)
                save_list.append(adversarial)
            else:
                continue
        else:
            continue
        if count == number:
            break
    save_np = np.asarray(save_list)
    np.savez(save_path, x_train=save_np)
    print("save completed.")


def get_arg_and_run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        help="model path")
    parser.add_argument("--label", type=int,
                        help="data original label")
    parser.add_argument("--number", type=int,
                        help="generate number")
    parser.add_argument("--save_path", type=str,
                        help="data save path")
    args = parser.parse_args()
    model_path = args.model_path
    label = args.label
    number = args.number
    save_path = args.save_path
    all_attcak(model_path, label, number, save_path)


if __name__ == '__main__':
    get_arg_and_run()

#     python adversary_attack.py --model_path ../model/lenet-5.h5 --label 0 --number 5 --save_path ../data/adversary_data/class_0.npz

# model = load_model("../model/lenet-5.h5")
# fmodel = foolbox.models.KerasModel(model, bounds=(0, 1), channel_axis=1)
# # criterion = foolbox.criteria.TargetClass(9)
# attack = foolbox.attacks.FGSM(fmodel)
# data = np.load("../data/original_data/class_0.npz")
# x_train = data["x_train"]
# print(x_train[0])
# x_attack = x_train[1].reshape(28, 28, 1) / 255
# adversarial = attack(x_attack, 0)
# # print(adversarial)
# print(model.predict(adversarial.reshape(1, 28, 28, 1)).argmax(axis=-1))
# adversarial = adversarial * 255
# adversarial = adversarial.astype(int)
#
# plt.figure()
#
# plt.subplot(1, 3, 1)
# plt.title('Original')
# plt.imshow(adversarial.reshape(28, 28))  # division by 255 to convert [0, 255] to [0, 1]
# plt.axis('off')
# plt.show()

