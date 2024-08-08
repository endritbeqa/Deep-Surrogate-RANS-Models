import os
import random
import numpy as np
from PIL import Image
from matplotlib import cm
import shutil
from ml_collections import config_dict


def makeDirs(directoryList):
    for directory in directoryList:
        if not os.path.exists(directory):
            os.makedirs(directory)

def saveAsImage(res, filename, field_param):
    field = np.copy(field_param)
    field = np.flipud(field.transpose())

    min_value = np.min(field)
    max_value = np.max(field)
    field -= min_value
    max_value -= min_value
    field /= max_value

    im = Image.fromarray(cm.magma(field, bytes=True))
    im = im.resize((res, res))
    im.save(filename)

def generate_uniform_random_parameters(num_samples, config):
    seed = 12345678
    np.random.seed(seed)
    files = os.listdir(config.airfoil_database)
    if len(files)==0:
        print("error - no airfoils found in {}".format(config.airfoil_database))
        exit(1)
    results = []
    for i in range(num_samples):
        name = os.path.splitext(os.path.basename(files[np.random.randint(0, len(files))]))[0]
        name = name + ".dat"
        results.append([i,name,np.random.uniform(config.min_velocity, config.max_velocity),np.random.uniform(config.min_AoA, config.max_AoA)])
    return results



def write_control_dict(file_path: str, config: config_dict):

    insert_endTime_position = "/*---Insert endTime here---*/"
    insert_writeInterval_position = "/*---Insert writeInterval here---*/"
    insert_purgeWrite_position = "/*---Insert purgeWrite here---*/"

    new_contents = []

    with open(file_path, 'r') as file:
        contents = file.readlines()

    for line in contents:
        if insert_endTime_position in line:
            new_contents.append("endTime         "+str(config.end_time)+";")
            continue
        elif insert_writeInterval_position in line:
            new_contents.append("writeInterval   "+str(config.write_interval)+";")
            continue
        elif insert_purgeWrite_position in line:
            new_contents.append("purgeWrite      "+str(config.purge_write)+";")
            continue
        new_contents.append(line)

    with open('./OpenFOAM/system/controlDict', 'w') as file:
        file.writelines(new_contents)


def write_point_coordinates(file_path: str, res: int):

    insert_position = "/*---Insert points here---*/"
    x_range = (-0.5, 1.5)
    y_range = (-1, 1)
    new_contents = []

    with open(file_path, 'r') as file:
        contents = file.readlines()

    point_coordinates = "points\n(\n"

    for y in np.linspace(y_range[0], y_range[1], res, endpoint=False):
        for x in np.linspace(x_range[0], x_range[1], res, endpoint=False):
            point_coordinates += '(' + str(x) + ' ' + str(y) + ' 0.5)\n'

    point_coordinates += ');'

    for line in contents:
        if insert_position in line:
            new_contents.append('\n' + point_coordinates + '\n')
            continue
        new_contents.append(line)

    with open('./OpenFOAM/system/internalCloud', 'w') as file:
        file.writelines(new_contents)


def clean_res_dir(config: config_dict,res_dir: str):
    os.chdir(res_dir)
    os.mkdir("./pictures")
    os.mkdir("./data")
    os.mkdir("./data/train")
    os.mkdir("./data/validation")
    for idx in range(config.num_workers):
        for item in os.listdir("worker_{}/data_pictures".format(idx)):
            (shutil.move("worker_{}/data_pictures/{}".format(idx, item), "./pictures"))
        for item in os.listdir("worker_{}/train".format(idx)):
            shutil.move("worker_{}/train/{}".format(idx, item), "./data/train")
        shutil.rmtree("worker_{}".format(idx))

    files = os.listdir("./data/train")
    num_files_to_move = int(len(files) * config.validation_split)
    files_to_move = random.sample(files, num_files_to_move)

    for file_name in files_to_move:
        src_file = os.path.join("./data/train", file_name)
        dest_file = os.path.join("./data/validation", file_name)
        shutil.move(src_file, dest_file)


def split_files_train_val(source_dir, validation_split):
    all_files = os.listdir(source_dir)
    random.shuffle(all_files)
    num_validation_files = int(len(all_files) * validation_split)
    train_files = all_files[num_validation_files:]
    validation_files = all_files[:num_validation_files]

    train_dir = os.path.join(source_dir, "train")
    validation_dir = os.path.join(source_dir, "validation")
    os.mkdir(train_dir)
    os.mkdir(validation_dir)

    for file_name in train_files:
        shutil.move(os.path.join(source_dir,file_name), os.path.join(train_dir,file_name))

    for file_name in validation_files:
        shutil.move(os.path.join(source_dir,file_name), os.path.join(validation_dir,file_name))
