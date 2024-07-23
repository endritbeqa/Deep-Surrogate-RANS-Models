import os, signal, math
import subprocess
import threading
import numpy as np
from ml_collections import config_dict
import utils


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


def genMesh(config: config_dict, airfoilFile):
    ar = np.loadtxt(airfoilFile, skiprows=1)

    # removing duplicate end point
    if np.max(np.abs(ar[0] - ar[(ar.shape[0] - 1)])) < 1e-6:
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
                line = line.replace("LAST_POINT_INDEX", "{}".format(pointIndex - 1))
                outFile.write(line)

    command_gmsh = Command("gmsh -format msh2 airfoil.geo -3 -o airfoil.msh > /dev/null")
    if command_gmsh.run(config.gmsh_timeout) != 0:
        print("gmsh timed out, moving on")
        return -1

    command_gmshToFoam = Command("gmshToFoam airfoil.msh > /dev/null")
    if command_gmshToFoam.run(config.gmshToFoam_timeout) != 0:
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
    os.rename("constant/polyMesh/boundaryTemp", "constant/polyMesh/boundary")

    return 0


def runSim(config: config_dict, freestreamX: float, freestreamY: float):
    with open("U_template", "rt") as inFile:
        with open("0/U", "wt") as outFile:
            for line in inFile:
                line = line.replace("VEL_X", "{}".format(freestreamX))
                line = line.replace("VEL_Y", "{}".format(freestreamY))
                outFile.write(line)

    command_simpleFoam = Command("./Allclean && simpleFoam > foam.log")
    if command_simpleFoam.run(config.simulation_timeout) != 0:
        print("simpleFoam timed out, moving on")
        return -1
    return 0


# TODO save input(freestreamX, freestreamY, mask) only once and the outputs(pressure, velocityX, velocityY) during time steps
def outputProcessing(config: config_dict, basename: str, freestreamX: float, freestreamY: float, imageIndex=0):
    for timeStep in config.save_timestep:
        pfile = 'OpenFOAM/postProcessing/internalCloud/{}/cloud_p.xy'.format(timeStep)
        ufile = 'OpenFOAM/postProcessing/internalCloud/{}/cloud_U.xy'.format(timeStep)

        # output layout channels:
        # [0] freestream field X + boundary
        # [1] freestream field Y + boundary
        # [2] binary mask for boundary
        # [3] pressure output
        # [4] velocity X output
        # [5] velocity Y output
        npOutput = np.zeros((6, config.res, config.res))
        ar_p = np.loadtxt(pfile)
        ar_v = np.loadtxt(ufile)
        curIndex = 0

        for y in range(config.res):
            for x in range(config.res):
                xf = (x / config.res - 0.5) * 2 + 0.5
                yf = (y / config.res - 0.5) * 2
                if abs(ar_p[curIndex][0] - xf) < 1e-4 and abs(ar_p[curIndex][1] - yf) < 1e-4:
                    npOutput[0][x][y] = freestreamX
                    npOutput[1][x][y] = freestreamY
                    npOutput[2][x][y] = 1.0
                    npOutput[3][x][y] = ar_p[curIndex][3]
                    npOutput[4][x][y] = ar_v[curIndex][3]
                    npOutput[5][x][y] = ar_v[curIndex][4]
                    curIndex += 1

        if config.save_images:
            os.makedirs('data_pictures/%04d' % (imageIndex), exist_ok=True)
            utils.saveAsImage(config.res, 'data_pictures/%04d/inputX_%d.png' % (imageIndex, timeStep), npOutput[0])
            utils.saveAsImage(config.res, 'data_pictures/%04d/inputY_%d.png' % (imageIndex, timeStep), npOutput[1])
            utils.saveAsImage(config.res, 'data_pictures/%04d/mask_%d.png' % (imageIndex, timeStep), npOutput[2])
            utils.saveAsImage(config.res, 'data_pictures/%04d/pressured_%d.png' % (imageIndex, timeStep), npOutput[3])
            utils.saveAsImage(config.res, 'data_pictures/%04d/velX_%d.png' % (imageIndex, timeStep), npOutput[4])
            utils.saveAsImage(config.res, 'data_pictures/%04d/velY_%d.png' % (imageIndex, timeStep), npOutput[5])


        fileName = config.output_dir + "%s_%d_%d_%d" % (
            basename, int(freestreamX * 100), int(freestreamY * 100), timeStep)
        print("\tsaving in " + fileName + ".npz")
        np.savez_compressed(fileName, a=npOutput)


def create_sample(config: config_dict, params: list):
    idx, basename, length, angle = params
    print("Run {}:".format(idx))
    print("\tusing {}".format(basename))

    fsX = math.cos(angle) * length
    fsY = -math.sin(angle) * length

    os.chdir("OpenFOAM/")
    os.system("./PrepareDirectory")
    utils.makeDirs(["./constant/polyMesh/sets", "./constant/polyMesh"])
    if genMesh(config, "../" + config.airfoil_database + basename) != 0:
        print("\tmesh generation failed, moving on")
        os.chdir("..")
        return

    if runSim(config, fsX, fsY) != 0:
        print("\tSimulation failed , moving on")
        os.chdir("..")
        return

    os.chdir("..")

    outputProcessing(config, basename, fsX, fsY, imageIndex=idx)
    print("\tdone")
    return


def generator(config: config_dict, samples: list, working_directory: str):
    os.chdir(working_directory)
    utils.makeDirs(["./data_pictures", "./train", "./OpenFOAM/constant/polyMesh/sets", "./OpenFOAM/constant/polyMesh"])
    for params in samples:
        create_sample(config, params)
