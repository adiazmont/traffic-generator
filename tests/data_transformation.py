from sklearn.preprocessing import PowerTransformer
import numpy as np
import matplotlib.pyplot as plt

# description_files_dir = 'wdg-functions/'
# description_files = {'wdg1': 'wdg1.txt',
#                      'wdg2': 'wdg2.txt',
#                      'wdg1_yeo_johnson': 'wdg1_yeo_johnson.txt',
#                      'wdg2_yeo_johnson': 'wdg2_yeo_johnson.txt'}
#
#
# def load_wavelength_dependent_gain(wavelength_dependent_gain_id):
#     """
#     :param wavelength_dependent_gain_id: file name id (see top of script) - string
#     :return: Return wavelength dependent gain array
#     """
#     wdg_file = description_files[wavelength_dependent_gain_id]
#     with open(description_files_dir + wdg_file, "r") as f:
#         return [float(line) for line in f]


pt = PowerTransformer(standardize=False)

description_files_dir = 'wdg-functions/'
description_files = {'wdg1': 'wdg1.txt', 'wdg2': 'wdg2.txt'}
wdg_file1 = description_files['wdg1']
with open(description_files_dir + wdg_file1, "r") as f:
    wdg1 = [float(line) for line in f]
wdg_file2 = description_files['wdg2']
with open(description_files_dir + wdg_file2, "r") as f:
    wdg2 = [float(line) for line in f]
del wdg1[-1]

# Need to convert to Numpy arrays
wdg1_array = np.array(wdg1).reshape(-1, 1)
wdg2_array = np.array(wdg2).reshape(-1, 1)
# Fit into the power transformator
pt.fit(wdg1_array)
wdg1_transform = pt.transform(wdg1_array)
pt.fit(wdg2_array)
wdg2_transform = pt.transform(wdg2_array)

# Save Numpy arrays into text files
np.savetxt(description_files_dir + 'wdg1_yeo_johnson.txt', wdg1_transform)
np.savetxt(description_files_dir + 'wdg2_yeo_johnson.txt', wdg2_transform)

# Save figures for referencing
plt.plot(wdg1_transform)
plt.savefig(description_files_dir + 'wdg1_yeo_johnson.png')
# Clear the plots
plt.clf()

plt.plot(wdg2_transform)
plt.savefig(description_files_dir + 'wdg2_yeo_johnson.png')
plt.clf()
