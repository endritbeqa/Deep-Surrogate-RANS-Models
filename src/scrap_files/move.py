import os
import shutil
import random


def move_files_to_parent_directory(root_dir):
    """
    Recursively move all files from subdirectories to their parent directory.

    Args:
        root_dir (str): The root directory to start from.
    """
    # Traverse the directory tree
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        for filename in filenames:
            # Construct the full file path
            file_path = os.path.join(dirpath, filename)
            # Move the file to the parent directory
            shutil.move(file_path, root_dir)

        # Once files are moved, remove the empty directory
        for dirname in dirnames:
            sub_dir_path = os.path.join(dirpath, dirname)
            try:
                os.rmdir(sub_dir_path)
            except OSError:
                # Directory is not empty, skip deletion
                pass


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



if __name__ == "__main__":
    #root_directory = "/home/blin/PycharmProjects/Thesis/src/data_res_32_uncertainty/res_32"  # Replace with your root directory
    #move_files_to_parent_directory(root_directory)
    split_files_train_val("/home/blin/PycharmProjects/Thesis/src/data_res_32_uncertainty/res_32",0.2)
