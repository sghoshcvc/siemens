import os

import numpy as np
from skimage import io as img_io
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset

import scipy.io

from skimage.transform import resize

import sys
import torchvision.transforms as transforms
TARGET_IMAGE_SIZE = [448, 448]
CHANNEL_MEAN = [0.485, 0.456, 0.406]
CHANNEL_STD = [0.229, 0.224, 0.225]
data_transforms = transforms.Compose(
    [
        transforms.Resize(TARGET_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(CHANNEL_MEAN, CHANNEL_STD),
    ]
)

class CtRdaDataset(Dataset):
    '''
    PyTorch dataset class for the segmentation-based George Washington dataset
    '''

    def __init__(self, data_dir, image_extension='.png',
                 fixed_image_size=[70, 150]):
        '''
        Constructor

        :param data_dir: full path to the GW root dir
        :param image_extension: the extension of image files (default: png)

        '''

        # class members
        self.word_list = None
        self.word_string_embeddings = None
        self.query_list = None
        self.label_encoder = None

        self.fixed_image_size = fixed_image_size

        self.path = data_dir
        self.heights = []
        self.widths = []
        self.im_list = []

        gt_file = os.path.join(data_dir, 'image.txt')
        car_data = []
        train_split_ids = []
        test_split_ids = []
        cnt = 0
        for line in open(gt_file):
            if not line.startswith("#"):
                word_info = line.split()
                img_name = word_info[0]
                id = int(word_info[1])

                img_filename = os.path.join(data_dir, 'images', img_name)

                if not os.path.isfile(img_filename):
                    continue

                # print word_img_filename
                try:
                    car_img = img_io.imread(img_filename)
                    # print(word_img_filename)
                    # print(word_img.shape)
                    # ht, wd = word_img.shape
                    # ap = wd/np.float(ht)
                    # wd_mod = int(40.0 * ap)
                    # if wd_mod<=500:
                    # # self.heights.append(ht)
                    # # self.widths.append(wd_mod)
                    #     self.im_list.append(word_img_filename)

                except:
                    continue
                # scale black pixels to 1 and white pixels to 0

                # word_img = check_size(img=word_img, min_image_width_height=min_image_width_height, fixed_image_size=(wd_mod, 80))
                # sys.exit()
                car_data.append((car_img, id))
                # if len(car_data) == 100:
                #     break

                '''
                if '-'.join(img_paths[:-1]) in train_img_names:
                    train_split_ids.append(1)
                else:
                    train_split_ids.append(0)
                if '-'.join(img_paths[:-1]) in test_img_names:
                    test_split_ids.append(1)
                else:
                    test_split_ids.append(0)
                cnt += 1
                '''


        #self.train_ids = train_split_ids
        #self.test_ids = test_split_ids
        # create random partition
        total_images = len(car_data)
        ntrain_imges = int(np.floor(total_images * 0.8))
        nval_images =  int(np.floor(total_images * 0.1))
        ntest_images = total_images - ntrain_imges-nval_images
        indexes = list(range(total_images))
        indexes = np.random.permutation(indexes)
        self.train_ids = indexes[:ntrain_imges]
        self.val_ids = indexes[ntrain_imges:ntrain_imges+nval_images]
        self.test_ids = indexes[ntrain_imges+nval_images:]
        # self.nclasses = np.unique([x[1] for x in car_data])
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit([elem[1] for elem in car_data])
        self.nclasses = np.max(self.label_encoder.transform(self.label_encoder.classes_))+1


        # self.train_ids = [x[0] for x in train_test_mat.get('idxTrain')]
        # self.val_ids =
        # self.test_ids = [x[0] for x in train_test_mat.get('idxTest')]

        self.car_data = car_data

    def mainLoader(self, partition=None):


        if partition not in [None, 'train', 'test','val']:
            raise ValueError('partition must be one of None, train or test')

        if partition is not None:
            if partition == 'train':
                self.car_list = [self.car_data[x] for x in self.train_ids]
                # self.word_string_embeddings = [x for i, x in enumerate(self.word_embeddings) if self.train_ids[i] == 1]
                # self.length_embeddings = [x for i, x in enumerate(self.length_embeddings) if self.train_ids[i] == 1]
                # self.char_string_embeddings = [x for i, x in enumerate(self.char_embeddings) if self.train_ids[i] == 1]
            elif partition == 'val':
                self.car_list = [self.car_data[x] for x in self.val_ids]
            else:
                self.car_list = [self.car_data[x] for x in self.test_ids]
                # self.word_string_embeddings = [x for i, x in enumerate(self.word_embeddings) if self.test_ids[i] == 1]
                # self.length_embeddings = [x for i, x in enumerate(self.length_embeddings) if self.test_ids[i] == 1]
                # self.char_string_embeddings = [x for i, x in enumerate(self.char_embeddings) if self.test_ids[i] == 1]
        else:
            # use the entire dataset
            self.word_list = self.car_data
            # self.word_string_embeddings = self.word_embeddings
            # self.char_string_embeddings = self.char_embeddings
            # self.length_embeddings = self.length_embeddings

        # if partition == 'test':
        #     # create queries
        #     word_strings = [elem[1] for elem in self.word_list]
        #     unique_word_strings, counts = np.unique(word_strings, return_counts=True)
        #     qry_word_ids = unique_word_strings[np.where(counts > 1)[0]]
        #
        #     # remove stopwords if needed
        #     stopwords = []
        #     for line in open(os.path.join(self.path, 'iam-stopwords')):
        #         stopwords.append(line.strip().split(','))
        #     stopwords = stopwords[0]
        #
        #     qry_word_ids = [word for word in qry_word_ids if word not in stopwords]
        #
        #     query_list = np.zeros(len(word_strings), np.int8)
        #     qry_ids = [i for i in range(len(word_strings)) if word_strings[i] in qry_word_ids]
        #     query_list[qry_ids] = 1
        #
        #     self.query_list = query_list
        # else:
        #     word_strings = [elem[1] for elem in self.word_list]
        #     self.query_list = np.zeros(len(word_strings), np.int8)
        #
        # if partition == 'train':
        #     # weights for sampling
        #     #train_class_ids = [self.label_encoder.transform([self.word_list[index][1]]) for index in range(len(self.word_list))]
        #     #word_strings = [elem[1] for elem in self.word_list]
        #     unique_word_strings, counts = np.unique(word_strings, return_counts=True)
        #     ref_count_strings = {uword : count for uword, count in zip(unique_word_strings, counts)}
        #     weights = [1.0/ref_count_strings[word] for word in word_strings]
        #     self.weights = np.array(weights)/sum(weights)
        #
        #     # neighbors
        #     #self.nbrs = NearestNeighbors(n_neighbors=32+1, algorithm='ball_tree').fit(self.word_string_embeddings)
        #     #indices = nbrs.kneighbors(self.word_embeddings, return_distance= False)


    def __len__(self):
        return len(self.car_list)

    def __getitem__(self, index):
        car_img = self.car_list[index][0]
        # fixed size image !!!
        # car_img = self._image_resize(car_img, self.fixed_image_size)

        # word_img = word_img.reshape((1,) + word_img.shape)

        car_img = torch.from_numpy(car_img)
        car_img = car_img.permute(2, 0, 1)
        # embedding = self.word_string_embeddings[index]
        # embedding = torch.from_numpy(embedding)
        # embedding_char = self.char_string_embeddings[index]
        # embedding_char = torch.from_numpy(embedding_char)
        # len_embed = self.length_embeddings[index]
        # len_embed = torch.from_numpy(len_embed)
        class_id = self.label_encoder.transform([self.car_list[index][1]])

        return car_img, class_id[0]

    # fixed sized image
    @staticmethod
    def _image_resize(word_img, fixed_img_size):

        if fixed_img_size is not None:
            if len(fixed_img_size) == 1:
                scale = float(fixed_img_size[0]) / float(word_img.shape[0])
                new_shape = (int(scale * word_img.shape[0]), int(scale * word_img.shape[1]))

            if len(fixed_img_size) == 2:
                new_shape = (fixed_img_size[0], fixed_img_size[1])

            word_img = resize(image=word_img, output_shape=new_shape).astype(np.float32)

        return word_img
