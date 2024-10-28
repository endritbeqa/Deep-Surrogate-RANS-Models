from src import utils

if __name__ == '__main__':
    output_dir = "/media/blin/VOL REC Blin/endrit/tests/uncertainty"
    labels = ['Ground Truth', 'ViT', 'Hierarchical VAE']
    ground_truth = [0.0017055667703971267,
        0.023249272257089615,
        0.029772255569696426,
        0.0331595353782177,
        0.03530581295490265,
        0.03692104294896126]
    ViT = [ 0.06705654412508011,
        0.04393627494573593,
        0.042940497398376465,
        0.04164035990834236,
        0.04032750427722931,
        0.04319944977760315]
    Hierachical_VAE = [ 0.012471460737287998,
        0.009918782860040665,
        0.021430429071187973,
        0.020813483744859695,
        0.024133365601301193,
        0.021037179976701736]
    x_values = [0.5, 2.5, 4.5, 6.5, 8.5, 10.5]

    lines = [ground_truth, ViT, Hierachical_VAE]


    utils.plot_std_curves(lines, x_values, labels, 1, 9, output_dir)