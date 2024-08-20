import os
import random
import shutil
import numpy as np
import torch
import torch.nn.functional as F


def remove_case_folders(path: str, res:int):

    data_path = path
    output_folder = "/home/blin/PycharmProjects/Thesis/src/Uncertainty_interpolated_2/res_{}".format(res)


    os.makedirs(output_folder)
    os.makedirs(os.path.join(output_folder,"train"))
    os.makedirs(os.path.join(output_folder,"validation"))

    for split_folder in ["train", "validation"]:
        for idx,folder in enumerate(os.listdir(os.path.join(data_path,split_folder))):
            for file in os.listdir(os.path.join(data_path,split_folder,folder)):
                src_path = os.path.join(data_path,split_folder, folder, file)
                dest_path = "{}/{}/{}".format(output_folder,split_folder,file)
                shutil.copy(src_path, dest_path)
            print(idx)




def interpolate_and_save(path: str, interpolate:bool, res:int):

    data_path = path
    output_folder = "/home/blin/PycharmProjects/Thesis/src/Uncertainty_interpolated/res_{}".format(res)


    os.makedirs(output_folder)
    os.makedirs(os.path.join(output_folder,"train"))
    os.makedirs(os.path.join(output_folder,"validation"))

    #for split_folder in ["train", "validation"]:
    for split_folder in [ "validation"]:
        for idx,folder in enumerate(os.listdir(os.path.join(data_path,split_folder))):
            os.makedirs(os.path.join(output_folder,split_folder,folder))
            for file in os.listdir(os.path.join(data_path,split_folder,folder)):

                data = np.load(os.path.join(data_path,split_folder, folder, file))
                arrays = data["a"]
                if interpolate:
                    arrays = torch.tensor(arrays, dtype=torch.float32)
                    arrays = torch.unsqueeze(arrays, dim=0)
                    arrays = F.interpolate(arrays, size=(res,res), mode='bilinear', align_corners=False)
                    arrays = torch.squeeze(arrays)
                    arrays = arrays.numpy()

                mask = arrays[2] != 0

                arrays[2][mask] = 1

                for i in [0, 1, 3, 4, 5]:
                    arrays[i][mask] = 0

                output_path = "{}/{}/{}/{}".format(output_folder,split_folder,folder,file)
                save_path = os.path.join(output_path)
                np.savez(save_path, a=arrays)
            print(idx)





def split_folders():

    dir_path = "/home/blin/PycharmProjects/Thesis/src/Uncertainty_data"
    output_dir = "/home/blin/PycharmProjects/Thesis/src/Uncertainty_interpolated/res_128"
    train_dir = os.path.join(output_dir,"train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir)
    os.makedirs(val_dir)

    cases = os.listdir(dir_path)
    random.shuffle(cases)

    num_train = int(len(cases)*0.8)

    train_cases = cases[:num_train]
    val_cases = cases[num_train:]



    for folder in train_cases:
        shutil.move(os.path.join(dir_path, folder), os.path.join(train_dir, folder))


    for folder in val_cases:
        shutil.move(os.path.join(dir_path, folder), os.path.join(val_dir, folder))




if __name__ == '__main__':

    remove_case_folders("/home/blin/PycharmProjects/Thesis/src/Uncertainty_interpolated/res_128", 128)
    #interpolate_and_save( "/home/blin/PycharmProjects/Thesis/src/Uncertainty_interpolated/res_128", True, 64)
