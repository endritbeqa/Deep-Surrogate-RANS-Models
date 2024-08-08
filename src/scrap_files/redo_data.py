import numpy as np
import os
from PIL import Image
from matplotlib import cm


def process_npz_files(directory):

    npz_files = [f for f in os.listdir(directory) if f.endswith('.npz')]

    for npz_file in npz_files:

        data = np.load(os.path.join(directory, npz_file))

        arrays = data["a"]
        mask = arrays[2] != 0

        arrays[2][mask] = 1

        for i in [0,1,3,4,5]:
            arrays[i][mask] = 0

        # Save the modified arrays back into a .npz file
        save_path = os.path.join(directory,npz_file)
        np.savez(save_path, a=arrays)
        print(f"Processed and saved: {save_path}")



def save_arrays_as_images(npz_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    data = np.load(npz_file)
    array = data['a']


    for i in range(6):
        field = np.copy(array[i])
        field = np.flipud(field.transpose())

        min_value = np.min(field)
        max_value = np.max(field)
        field -= min_value
        max_value -= min_value
        field /= max_value

        img = Image.fromarray(cm.magma(field, bytes=True))

        img_filename = os.path.join(output_dir, f"{i}.png")
        img.save(img_filename)


if __name__ == '__main__':
    process_npz_files("/home/blin/PycharmProjects/Thesis/src/data_res_32_uncertainty/data/validation")
    #save_arrays_as_images("/home/blin/PycharmProjects/Thesis/src/data_res_32_uncertainty/data/test/2032c_5735_1406_2540.npz",
                 #         "/home/blin/PycharmProjects/Thesis/src/data_res_32_uncertainty/data/test")