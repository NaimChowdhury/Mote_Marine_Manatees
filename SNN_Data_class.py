import os
import cv2
import numpy as np

dataset_folder = '/Users/rosagradilla/Documents/summer20/Manatee/Dataset/MMLDUs_BatchA/'

class SiameseNetworkDataset(object):
    """ 
    This class outputs a .npy file with the images and label as (img0,img1,label)
    it takes dataset path and image size as arguments
    """
    def __init__(self, PATH, IMG_SIZE):
        self.path = PATH
        self.img_size = IMG_SIZE
    
    LABELS = {'similar': 1, 'not_similar':0}
    
    training_data = []
    
    def make_data(self, out_name):
        # provide a name for .npy file being written
        i = 0
        file_names = []

        # First get all file names in list file_names
        for file in sorted(os.listdir(self.path)):
            file_names.append(str(file))

        for i in range(len(file_names)): 
            # Check if they are pairs
            if file_names[i][-5] == 'A' and file_names[i+1][-5] == 'B':
                img0 = os.path.join(self.path, file_names[i])
                img1 = os.path.join(self.path, file_names[i+1])
                img2 = os.path.join(self.path, file_names[np.random.randint(0, len(file_names))])

                # Read the image
                img0 = cv2.imread(img0, cv2.IMREAD_UNCHANGED)
                img1 = cv2.imread(img1, cv2.IMREAD_UNCHANGED)
                img2 = cv2.imread(img2, cv2.IMREAD_UNCHANGED)

                # Resize according to IMG_SIZE
                img0 = cv2.resize(img0, (self.img_size, self.img_size), interpolation= cv2.INTER_AREA)
                img1 = cv2.resize(img1, (self.img_size, self.img_size), interpolation= cv2.INTER_AREA)
                img2 = cv2.resize(img2, (self.img_size, self.img_size), interpolation= cv2.INTER_AREA)

                # append to training_data
                self.training_data.append([np.array(img0), np.array(img1), np.eye(2)[self.LABELS['similar'] ]])
                self.training_data.append([np.array(img0), np.array(img2), np.eye(2)[self.LABELS['not_similar'] ]])
                
            # else, continue
            i += 1
        
        np.save(out_name + '.npy', self.training_data )


siamese = SiameseNetworkDataset(dataset_folder, 128)
siamese.make_data('out2')

""" 
The Data can be loaded from any other file by loading it through numpy:
np.load('out.npy', allow_pickle=True)
"""

#training_data = np.load('/Users/rosagradilla/Documents/summer20/Manatee/out1.npy', allow_pickle=True)
