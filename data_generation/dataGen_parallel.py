################
#
# Deep Flow Prediction - N. Thuerey, K. Weissenov, H. Mehrotra, N. Mainali, L. Prantl, X. Hu (TUM)
#
# Generate training data via OpenFOAM
#
################

import math, random
import shutil
import threading
from asyncio import threads

import concurrent.futures
from sim_Funcs import *



samples           = 100           # no. of datasets to produce
freestream_angle  = math.pi / 8.  # -angle ... angle
freestream_length = 10.           # len * (1. ... factor)
freestream_length_factor = 10.    # length factor

airfoil_database  = "./airfoil_database/"
output_dir        = "../data/train/"

seed = random.randint(0, 2**32 - 1)
np.random.seed(seed)
print("Seed: {}".format(seed))



files = os.listdir(airfoil_database)
files.sort()
if len(files)==0:
	print("error - no airfoils found in %s" % airfoil_database)
	exit(1)

utils.makeDirs(["./data_pictures", "./train", "./OpenFOAM/constant/polyMesh/sets", "./OpenFOAM/constant/polyMesh"])




def create_sample(params):

    for sample_params in params:
        i, basename, length, angle = sample_params
        fsX =  math.cos(angle) * length
        fsY = -math.sin(angle) * length

        print("\tUsing len %5.3f angle %+5.3f " %( length,angle )  )
        print("\tResulting freestream vel x,y: {},{}".format(fsX,fsY))

        os.chdir("OpenFOAM/")
        if genMesh("../" + airfoil_database + basename) != 0:
            print("\tmesh generation failed, aborting")
            os.chdir("")

        runSim(fsX, fsY)
        os.chdir("")

        outputProcessing(basename, fsX, fsY, imageIndex=i)
        print("\tdone")


def create_sample_fault_tolerant(params):

    timeout = 500

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(create_sample, params)
        try:
            result = future.result(timeout=timeout)
            print(result)
        except concurrent.futures.TimeoutError:
            shutil.rmtree("OpenFOAM")
            shutil.copytree(src="OpenFOAMCopy", dst="OpenFOAM")
        except Exception as e:
            print(f"An error occurred: {e}")


num_processes = 4

samples = utils.generate_uniform_random_parameters(sample_times=1000)

k, m = divmod(len(samples), 4)

samples_grouped =  [samples[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(num_processes)]

def generator(params):
    create_sample_fault_tolerant(params)


for i in range(0, num_processes):
    dir_name = ""
    os.makedirs(dir_name, exist_ok=True)
    print(f"Directory '{dir_name}' created successfully.")
    shutil.copytree(src="OpenFOAM", dst=dir_name + "/OpenFOAM")
    shutil.copytree(src="OpenFOAMCopy", dst=dir_name + "/OpenFOAMCopy")

    thread = threading.Thread(target=generator, args=([samples_grouped[i]]))
    thread.start()
    threads.append(thread)


# Wait for all threads to complete
for thread in threads:
    thread.join()