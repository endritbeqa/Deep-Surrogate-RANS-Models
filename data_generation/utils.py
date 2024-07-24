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

def generate_uniform_random_parameters(sample_times,name_airfoil=None,path_airfoil_database  = "./airfoil_database/",min_velocity=10,max_velocity=100,min_AoA=-22.5,max_AoA=22.5):
    seed = random.randint(0, 2**32 - 1)
    np.random.seed(seed)
    if name_airfoil is None:
        files = os.listdir(path_airfoil_database)
        if len(files)==0:
            print("error - no airfoils found in {}".format(path_airfoil_database))
            exit(1)
    results = []
    for i in range(sample_times):
        if name_airfoil is None:
            name = os.path.splitext(os.path.basename(files[np.random.randint(0, len(files))]))[0]
            name = name + ".dat"
        else:
            name = name_airfoil+'.dat'
        results.append([i,name,np.random.uniform(min_velocity, max_velocity),np.random.uniform(min_AoA, max_AoA) ])
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
    for idx in range(config.num_workers):
        for item in os.listdir("worker_{}/data_pictures".format(idx)):
            (shutil.move("worker_{}/data_pictures/{}".format(idx, item), "./pictures"))
        for item in os.listdir("worker_{}/train".format(idx)):
            shutil.move("worker_{}/train/{}".format(idx, item), "./data")
        shutil.rmtree("worker_{}".format(idx))


def move_files_by_percentage(source_dir, destination_dir, percentage):
    if not (0 <= percentage <= 100):
        raise ValueError("Percentage must be between 0 and 100")

    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    all_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    num_files_to_move = int(len(all_files) * (percentage / 100))

    files_to_move = random.sample(all_files, num_files_to_move)

    # Move the files
    for file_name in files_to_move:
        src_file = os.path.join(source_dir, file_name)
        dest_file = os.path.join(destination_dir, file_name)
        shutil.move(src_file, dest_file)

    print(f"Moved {num_files_to_move} files from {source_dir} to {destination_dir}")


