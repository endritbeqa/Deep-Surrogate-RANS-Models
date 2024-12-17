from src import trainers
from src.train_config import get_config_parametrized


def run():
    output_dir = '/home/blin/endrit/tests/uncertainty/run_all'
    models = ["Swin_UNet", "ViT_"]
    loss_functions = ['l1', 'mse']
    datasets_res = ['res_32']  #, 'res_64', 'res_128']

    for res in datasets_res:
        for model in models:
            for loss in loss_functions:
                study_name = "{}_{}_{}".format(model, res, loss)
                data_dir = '/home/blin/endrit/dataset/uncertainty/preprocessed/{}/full/train_val_split'.format(res)
                train_config = get_config_parametrized(output_dir, study_name, model, data_dir, loss)
                trainer = trainers.DiffusionTrainer(train_config)
                trainer.train_model()


if __name__ == '__main__':
    run()

