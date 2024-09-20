import os
from src.data import  dataset

class Model_Test(object):
    def __init__(self, config: ):
        self.model = load




test_dataset = dataset.Airfoil_Dataset(self.config, mode='train')
        os.mkdir(output_dir)
        for dir in [os.path.join(self.output_dir, "checkpoints"),
                    os.path.join(self.output_dir, "logs"),
                    os.path.join(self.output_dir, "config"),
                    os.path.join(self.output_dir, "images"),
                    os.path.join(self.output_dir, "images/predictions"),
                    os.path.join(self.output_dir, "images/targets")]:
            os.mkdir(dir)





