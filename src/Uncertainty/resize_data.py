import os
import torch
import numpy as np


def process_npz_files(input_folder, output_folder, res):

    os.makedirs(output_folder, exist_ok=True)

    cases = [f for f in os.listdir(input_folder)]

    for case in cases:
        os.makedirs(os.path.join(output_folder,case))
        npz_files = [f for f in os.listdir(os.path.join(input_folder,case)) if f.endswith('.npz')]
        
        for npz_file in npz_files:
            file_path = os.path.join(input_folder,case, npz_file)
            data = np.load(file_path)
        
            if 'a' in data:
                array = data['a']
        
                if array.shape == (6, 128, 128):
                    tensor = torch.tensor(array, dtype=torch.float32)
                    tensor = tensor.unsqueeze(0)
                    interpolated_tensor = torch.nn.functional.interpolate(
                        tensor, size=(res, res), mode='bilinear', align_corners=False
                    )
                    interpolated_array = interpolated_tensor.squeeze(0).numpy()
                    output_file_path = os.path.join(output_folder,case, npz_file)
                    np.savez_compressed(output_file_path, a = interpolated_array)
                else:
                    print(f"Skipping {file_path}: Array shape is not (6, 128, 128)")
            else:
                print(f"Skipping {file_path}: No 'arr_0' found in the .npz file")


if __name__ == '__main__':
    process_npz_files("/home/blin/PycharmProjects/Thesis/src/Uncertainty/res_128","/home/blin/PycharmProjects/Thesis/src/Uncertainty/res_32", 32)