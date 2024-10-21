import numpy as np


if __name__ == '__main__':
    MSE_file = '/media/blin/VOL REC Blin/endrit/tests/uncertainty/test_swin_NVAE_Gaussian_128_full/evaluation/Inter_Extrapolation/MSE_interpolation.txt'
    target_means_file = '/media/blin/VOL REC Blin/endrit/tests/uncertainty/test_swin_NVAE_Gaussian_128_full/evaluation/Inter_Extrapolation/target_means_interpolation.txt'


    mse_data = np.genfromtxt(MSE_file, delimiter=',', skip_header=1)
    target_means_data = np.genfromtxt(target_means_file, delimiter=',', skip_header=1)

    mse_data = (mse_data[:, 1:7]).astype(np.float32)
    target_means_data = (target_means_data[:, 1:]).astype(np.float32)

    target_means_low_mask = target_means_data[:, 5] < 5e-3
    target_means_low_mask = np.tile(target_means_low_mask, 6).reshape((-1, 6))

    all_std_MSEs = np.mean(mse_data[:, 5])
    all_mean_MSEs = np.mean(mse_data[:, 2])


    low_std_MSEs = np.mean(mse_data[:, 5][target_means_low_mask[:, 5]])
    low_mean_MSEs = np.mean(mse_data[:, 2][target_means_low_mask[:, 2]])

    high_std_MSEs = np.mean(mse_data[:, 5][~target_means_low_mask[:, 5]])
    high_mean_MSEs = np.mean(mse_data[:, 2][~target_means_low_mask[:, 2]])

    print()

