import os, signal, math, random
import shlex
import subprocess
import threading
import time

import numpy as np
import utils


samples           = 100           # no. of datasets to produce
freestream_angle  = math.pi / 8.  # -angle ... angle
freestream_length = 10.           # len * (1. ... factor)
freestream_length_factor = 10.    # length factor

airfoil_database  = "./airfoil_database/"
output_dir        = "./train/"

seed = random.randint(0, 2**32 - 1)
np.random.seed(seed)
print("Seed: {}".format(seed))


class Command(object):
    def __init__(self, cmd):
        self.cmd = cmd
        self.process = None

    def run(self, timeout):
        def target():
            print('Thread with following command {} started'.format(self.cmd))
            self.process = subprocess.Popen(self.cmd, shell=True, preexec_fn=os.setsid)
            self.process.communicate()
            print('Thread with following command {} finished'.format(self.cmd))

        thread = threading.Thread(target=target)
        thread.start()

        thread.join(timeout)
        if thread.is_alive():
            print('Terminating process')
            os.killpg(self.process.pid, signal.SIGTERM)
            thread.join()
            return self.process.returncode
        return self.process.returncode


def genMesh(airfoilFile):
    ar = np.loadtxt(airfoilFile, skiprows=1)

    # removing duplicate end point
    if np.max(np.abs(ar[0] - ar[(ar.shape[0]-1)]))<1e-6:
        ar = ar[:-1]

    output = ""
    pointIndex = 1000
    for n in range(ar.shape[0]):
        output += "Point({}) = {{ {}, {}, 0.00000000, 0.005}};\n".format(pointIndex, ar[n][0], ar[n][1])
        pointIndex += 1

    with open("airfoil_template.geo", "rt") as inFile:
        with open("airfoil.geo", "wt") as outFile:
            for line in inFile:
                line = line.replace("POINTS", "{}".format(output))
                line = line.replace("LAST_POINT_INDEX", "{}".format(pointIndex-1))
                outFile.write(line)


    command_gmsh = Command("gmsh -format msh2 airfoil.geo -3 -o airfoil.msh > /dev/null")
    if command_gmsh.run(15) != 0:
        print("gmsh timed out, moving on")
        return -1

    command_gmshToFoam = Command("gmshToFoam airfoil.msh > /dev/null")
    if command_gmshToFoam.run(40) !=0:
        print("gmshToFoam timed out, moving on")
        return -1

    with open("constant/polyMesh/boundary", "rt") as inFile:
        with open("constant/polyMesh/boundaryTemp", "wt") as outFile:
            inBlock = False
            inAerofoil = False
            for line in inFile:
                if "front" in line or "back" in line:
                    inBlock = True
                elif "aerofoil" in line:
                    inAerofoil = True
                if inBlock and "type" in line:
                    line = line.replace("patch", "empty")
                    inBlock = False
                if inAerofoil and "type" in line:
                    line = line.replace("patch", "wall")
                    inAerofoil = False
                outFile.write(line)
    os.rename("constant/polyMesh/boundaryTemp","constant/polyMesh/boundary")

    return 0

def runSim(freestreamX, freestreamY):
    with open("U_template", "rt") as inFile:
        with open("0/U", "wt") as outFile:
            for line in inFile:
                line = line.replace("VEL_X", "{}".format(freestreamX))
                line = line.replace("VEL_Y", "{}".format(freestreamY))
                outFile.write(line)

    command_simpleFoam = Command("./Allclean && simpleFoam > foam.log")
    if command_simpleFoam.run(200) != 0:
        print("simpleFoam timed out, moving on")
        return -1
    return 0



def outputProcessing(basename, freestreamX, freestreamY, dataDir=output_dir, pfile='OpenFOAM/postProcessing/internalCloud/500/cloud_p.xy', ufile='OpenFOAM/postProcessing/internalCloud/500/cloud_U.xy', res=32, imageIndex=0):
    # output layout channels:
    # [0] freestream field X + boundary
    # [1] freestream field Y + boundary
    # [2] binary mask for boundary
    # [3] pressure output
    # [4] velocity X output
    # [5] velocity Y output
    npOutput = np.zeros((6, res, res))

    ar = np.loadtxt(pfile)
    curIndex = 0

    for y in range(res):
        for x in range(res):
            xf = (x / res - 0.5) * 2 + 0.5
            yf = (y / res - 0.5) * 2
            if abs(ar[curIndex][0] - xf)<1e-4 and abs(ar[curIndex][1] - yf)<1e-4:
                npOutput[3][x][y] = ar[curIndex][3]
                curIndex += 1
                # fill input as well
                npOutput[0][x][y] = freestreamX
                npOutput[1][x][y] = freestreamY
            else:
                npOutput[3][x][y] = 0
                # fill mask
                npOutput[2][x][y] = 1.0

    ar = np.loadtxt(ufile)
    curIndex = 0

    for y in range(res):
        for x in range(res):
            xf = (x / res - 0.5) * 2 + 0.5
            yf = (y / res - 0.5) * 2
            if abs(ar[curIndex][0] - xf)<1e-4 and abs(ar[curIndex][1] - yf)<1e-4:
                npOutput[4][x][y] = ar[curIndex][3]
                npOutput[5][x][y] = ar[curIndex][4]
                curIndex += 1
            else:
                npOutput[4][x][y] = 0
                npOutput[5][x][y] = 0

    os.makedirs('data_pictures/%04d'%(imageIndex), exist_ok=True)
    utils.saveAsImage('data_pictures/%04d/pressured.png' % (imageIndex), npOutput[3])
    utils.saveAsImage('data_pictures/%04d/velXd.png' % (imageIndex), npOutput[4])
    utils.saveAsImage('data_pictures/%04d/velY.png' % (imageIndex), npOutput[5])
    utils.saveAsImage('data_pictures/%04d/inputX.png' % (imageIndex), npOutput[0])
    utils.saveAsImage('data_pictures/%04d/inputY.png' % (imageIndex), npOutput[1])

    #fileName = dataDir + str(uuid.uuid4()) # randomized name
    fileName = dataDir + "%s_%d_%d" % (basename, int(freestreamX*100), int(freestreamY*100) )
    print("\tsaving in " + fileName + ".npz")
    np.savez_compressed(fileName, a=npOutput)

def create_sample(idx:int, basename:str ,length:float, angle:float   ):

    print("Run {}:".format(idx))
    print("\tusing {}".format(basename))

    fsX = math.cos(angle) * length
    fsY = -math.sin(angle) * length

    os.chdir("OpenFOAM/")
    os.system("./PrepareDirectory")
    utils.makeDirs(["./constant/polyMesh/sets", "./constant/polyMesh"])
    if genMesh("../" + airfoil_database + basename) != 0:
        print("\tmesh generation failed, moving on")
        os.chdir("..")
        return

    if runSim(fsX, fsY) != 0:
        print("\tSimulation failed , moving on")
        os.chdir("..")
        return

    os.chdir("..")

    outputProcessing(basename, fsX, fsY, imageIndex=idx)
    print("\tdone")
    return


def generator(samples, working_directory):

    os.chdir(working_directory)
    utils.makeDirs(["./data_pictures", "./train", "./OpenFOAM/constant/polyMesh/sets", "./OpenFOAM/constant/polyMesh"])
    for params in samples:
        create_sample(params[0], params[1], params[2], params[3])









