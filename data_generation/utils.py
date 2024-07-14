################
#
# Deep Flow Prediction - N. Thuerey, K. Weissenov, H. Mehrotra, N. Mainali, L. Prantl, X. Hu (TUM)
#
# Helpers for data generation
#
################

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

def imageOut(filename, outputs_param, targets_param, saveTargets=False):
    outputs = np.copy(outputs_param)
    targets = np.copy(targets_param)

    for i in range(3):
        min_value = min(np.min(outputs[i]), np.min(targets[i]))
        max_value = max(np.max(outputs[i]), np.max(targets[i]))
        outputs[i] -= min_value
        targets[i] -= min_value
        max_value -= min_value
        outputs[i] /= max_value
        targets[i] /= max_value

        suffix = ""
        if i==0:
            suffix = "_pressure"
        elif i==1:
            suffix = "_velX"
        else:
            suffix = "_velY"

        im = Image.fromarray(cm.magma(outputs[i], bytes=True))
        im = im.resize((512,512))
        im.save(filename + suffix + "_pred.png")

        if saveTargets:
            im = Image.fromarray(cm.magma(targets[i], bytes=True))
            im = im.resize((512,512))
            im.save(filename + suffix + "_target.png")

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
    if len(files)==0:
        print("error - no airfoils found in {}".format(path_airfoil_database))
        exit(1)
    results=[]
    for i in range(sample_times):
        if name_airfoil is None:
            name = os.path.splitext(os.path.basename(files[np.random.randint(0, len(files))]))[0]
            name = name + ".dat"
        else:
            name = name_airfoil+'.dat'
        results.append([i,name,np.random.uniform(min_velocity, max_velocity),np.random.uniform(min_AoA, max_AoA) ])
    return results




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
        for item in os.listdir("working_case_{}/data_pictures".format(idx)):
            (shutil.move("working_case_{}/data_pictures/{}".format(idx, item), "./pictures"))
        for item in os.listdir("working_case_{}/train".format(idx)):
            shutil.move("working_case_{}/train/{}".format(idx, item), "./data")
        shutil.rmtree("working_case_{}".format(idx))
