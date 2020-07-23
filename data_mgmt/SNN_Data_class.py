import os
import cv2
import numpy as np

""" 
To use this class:
initiate a class instance
siamese = SiameseNetworkDataset(data_folder, IMSIZE)
siamese.make_data('out_name')

The Data can be loaded from any other file by loading it through numpy:
np.load('out.npy', allow_pickle=True)
"""

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
    
    def num_data_items(self):
        i = 0
        file_names = []

        # First get all file names in list file_names
        for file in sorted(os.listdir(self.path)):
            file_names.append(str(file))
        return len(file_names)

    
    def make_data(self, out_name):
        # provide a name for .npy file being written
        i = 0
        file_names = []

        # First get all file names in list file_names
        for file in sorted(os.listdir(self.path)):
            file_names.append(str(file))

        for i in range(len(file_names)-2): 
            # Check if they are pairs
            if file_names[i][-5] == 'A' and file_names[i+1][-5] == 'B':
                img0_path = os.path.join(self.path, file_names[i])
                img1_path = os.path.join(self.path, file_names[i+1])
                img2_path = os.path.join(self.path, file_names[np.random.randint(1, len(file_names))])
                img3_path = os.path.join(self.path, file_names[np.random.randint(1, len(file_names))])

                # Read the image
                img0 = cv2.imread(img0_path, cv2.IMREAD_GRAYSCALE)
                img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
                img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
                img3 = cv2.imread(img3_path, cv2.IMREAD_GRAYSCALE)

                if img0 is None:
                    print(img0_path + 'is type None')
                if img1 is None:
                    print(img1_path + 'is type None')
                if img2 is None:
                    print(img2_path + 'is type None')
                if img3 is None:
                    print(img3_path + 'is type None')

                # Resize according to IMG_SIZE
                img0 = cv2.resize(img0, (self.img_size, self.img_size), interpolation= cv2.INTER_CUBIC)
                img1 = cv2.resize(img1, (self.img_size, self.img_size), interpolation= cv2.INTER_CUBIC)
                img2 = cv2.resize(img2, (self.img_size, self.img_size), interpolation= cv2.INTER_CUBIC)
                img3 = cv2.resize(img3, (self.img_size, self.img_size), interpolation= cv2.INTER_CUBIC)

                # append to training_data
                self.training_data.append([np.array(img0), np.array(img1), np.eye(2)[self.LABELS['similar'] ]])
                self.training_data.append([np.array(img0), np.array(img2), np.eye(2)[self.LABELS['not_similar'] ]])
                self.training_data.append([np.array(img0), np.array(img3), np.eye(2)[self.LABELS['not_similar'] ]])
                
            if file_names[i][-5] == 'A' and file_names[i+2][-5] == 'C':
                img0_path = os.path.join(self.path, file_names[i])
                img1_path = os.path.join(self.path, file_names[i+2])
                img2_path = os.path.join(self.path, file_names[np.random.randint(1, len(file_names))])
                img3_path = os.path.join(self.path, file_names[np.random.randint(1, len(file_names))])

                # Read the image
                img0 = cv2.imread(img0_path, cv2.IMREAD_GRAYSCALE)
                img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
                img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
                img3 = cv2.imread(img3_path, cv2.IMREAD_GRAYSCALE)

                # Resize according to IMG_SIZE
                img0 = cv2.resize(img0, (self.img_size, self.img_size), interpolation= cv2.INTER_CUBIC)
                img1 = cv2.resize(img1, (self.img_size, self.img_size), interpolation= cv2.INTER_CUBIC)
                img2 = cv2.resize(img2, (self.img_size, self.img_size), interpolation= cv2.INTER_CUBIC)
                img3 = cv2.resize(img3, (self.img_size, self.img_size), interpolation= cv2.INTER_CUBIC)

                # append to training_data
                self.training_data.append([np.array(img0), np.array(img1), np.eye(2)[self.LABELS['similar'] ]])
                self.training_data.append([np.array(img0), np.array(img2), np.eye(2)[self.LABELS['not_similar'] ]])
                self.training_data.append([np.array(img0), np.array(img3), np.eye(2)[self.LABELS['not_similar'] ]])

            # else, continue
            i += 1
        
        np.save(out_name + '.npy', self.training_data )

