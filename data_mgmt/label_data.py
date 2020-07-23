import utils_module
import numpy as np

data_folder = '/Users/rosagradilla/Documents/summer20/Manatee/Dataset_merged/'

#SiameseDataset = utils_module.SiameseNetworkDataset(data_folder, IMG_SIZE=105)
#SiameseDataset.make_data('out_jul23')

"""Check Data

training_data = np.load('/Users/rosagradilla/Documents/summer20/Manatee/out_jul23.npy', allow_pickle=True)
print(training_data.shape[0])

counter = 0
for labeled_pair in range(1659):
    img0_shape = training_data[labeled_pair][0].shape
    img1_shape = training_data[labeled_pair][1].shape
    img2_shape = training_data[labeled_pair][2].shape
    if img0_shape == (105,105) and img1_shape == (105,105) and img2_shape == (2,):
        counter += 1

print(counter)
"""