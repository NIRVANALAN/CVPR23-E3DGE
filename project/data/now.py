import os
import torchvision.transforms as transforms
import numpy as np
import cv2
from skimage.io import imread
from skimage.transform import estimate_transform, warp
from torch.utils.data import Dataset


class NoWDataset(Dataset):

    def __init__(self,
                 ring_elements=6,
                 crop_size=224,
                 scale=1.6,
                 mode='val',
                 normalize_img=False):
        # folder = '/ps/scratch/yfeng/other-github/now_evaluation/data/NoW_Dataset'
        folder = '/mnt/lustre/share/yslan/3DFace/NoW/NoW_Dataset'
        data_paths = {
            'val': os.path.join(folder, 'imagepathsvalidation.txt'),
            'test': os.path.join(folder, 'imagepathstest.txt')
        }

        self.data_path = data_paths[mode]
        with open(self.data_path) as f:
            self.data_lines = f.readlines()

        self.imagefolder = os.path.join(folder, 'final_release_version',
                                        'iphone_pictures')
        self.bbxfolder = os.path.join(folder, 'final_release_version',
                                      'detected_face')

        self.crop_size = crop_size
        self.scale = scale

        self.normalize_img = normalize_img  # as in the training

        # self.transforms = [transforms.ToTensor()]
        # if self.normalize_img:
        #     self.transforms.append(
        #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
        #                              inplace=True))
        #     print('normalizing to [0.5,0.5,0.5]', flush=True)
        # self.transforms = transforms.Compose(self.transforms)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
                                 inplace=True)
        ])

    def __len__(self):
        return len(self.data_lines)

    def __getitem__(self, index):
        imagepath = os.path.join(self.imagefolder,
                                 self.data_lines[index].strip())  #+ '.jpg'
        bbx_path = os.path.join(
            self.bbxfolder,
            self.data_lines[index].strip().replace('.jpg', '.npy'))
        bbx_data = np.load(bbx_path, allow_pickle=True,
                           encoding='latin1').item()
        # box = np.array([[bbx_data['left'], bbx_data['top']], [bbx_data['right'], bbx_data['bottom']]]).astype('float32')
        left = bbx_data['left']
        right = bbx_data['right']
        top = bbx_data['top']
        bottom = bbx_data['bottom']

        imagename = imagepath.split('/')[-1].split('.')[0]
        image = imread(imagepath)[:, :, :3]  # rgb channel

        h, w, _ = image.shape
        old_size = (right - left + bottom - top) / 2
        center = np.array(
            [right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
        size = int(old_size * self.scale)

        # crop image
        src_pts = np.array([[center[0] - size / 2, center[1] - size / 2],
                            [center[0] - size / 2, center[1] + size / 2],
                            [center[0] + size / 2, center[1] - size / 2]])
        DST_PTS = np.array([[0, 0], [0, self.crop_size - 1],
                            [self.crop_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)

        image = image / 255.  # as in deca
        dst_image = warp(image,
                         tform.inverse,
                         output_shape=(self.crop_size,
                                       self.crop_size)).astype(np.float32)
        # dst_image *= 255

        #
        cv2_image = cv2.cvtColor(dst_image * 255,
                                 cv2.COLOR_RGB2BGR)  # for lms prediction
        dst_image = self.transform(
            dst_image
        )  # since dst_image is float32 here, scale to [0,1] does not check

        return {
            'cv2_image': cv2_image,
            'image': dst_image,
            'imagename': self.data_lines[index].strip().replace('.jpg', ''),
        }
        # images = torch.tensor(dst_image).float()
        # imagename = self.data_lines[index].strip().replace('.jpg', '')
        # return images, imagename
    def __getitem__old(self, index):
        imagepath = os.path.join(self.imagefolder,
                                 self.data_lines[index].strip())  #+ '.jpg'
        bbx_path = os.path.join(
            self.bbxfolder,
            self.data_lines[index].strip().replace('.jpg', '.npy'))
        bbx_data = np.load(bbx_path, allow_pickle=True,
                           encoding='latin1').item()
        # box = np.array([[bbx_data['left'], bbx_data['top']], [bbx_data['right'], bbx_data['bottom']]]).astype('float32')
        left = bbx_data['left']
        right = bbx_data['right']
        top = bbx_data['top']
        bottom = bbx_data['bottom']

        imagename = imagepath.split('/')[-1].split('.')[0]
        image = imread(imagepath)[:, :, :3]  # rgb channel

        h, w, _ = image.shape
        old_size = (right - left + bottom - top) / 2
        center = np.array(
            [right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
        size = int(old_size * self.scale)

        # crop image
        src_pts = np.array([[center[0] - size / 2, center[1] - size / 2],
                            [center[0] - size / 2, center[1] + size / 2],
                            [center[0] + size / 2, center[1] - size / 2]])
        DST_PTS = np.array([[0, 0], [0, self.crop_size - 1],
                            [self.crop_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)

        image = image / 255.  # as in deca
        dst_image = warp(image,
                         tform.inverse,
                         output_shape=(self.crop_size,
                                       self.crop_size)).astype(np.float32)
        # dst_image *= 255

        #
        cv2_image = cv2.cvtColor(dst_image * 255,
                                 cv2.COLOR_RGB2BGR)  # for lms prediction
        dst_image = self.transform(
            dst_image
        )  # since dst_image is float32 here, scale to [0,1] does not check

        return {
            'cv2_image': cv2_image,
            'image': dst_image,
            'imagename': self.data_lines[index].strip().replace('.jpg', ''),
        }
        # images = torch.tensor(dst_image).float()
        # imagename = self.data_lines[index].strip().replace('.jpg', '')
        # return images, imagename
