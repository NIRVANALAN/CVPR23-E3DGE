from torchvision import transforms
from pathlib import Path
import lmdb
import random
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
from io import BytesIO
from torch.utils.data import Dataset

from sorcery import dict_of


class MultiResolutionDataset(Dataset):

    def __init__(self, path, transform, resolution=256, nerf_resolution=64):

        # self._init_lmdb(path)
        self._init_local_dataset(path)
        self.resolution = resolution
        self.nerf_resolution = nerf_resolution
        self.transform = transform

    def _init_lmdb(self, path):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(
                txn.get('length'.encode('utf-8')).decode('utf-8'))

    def _read_lmdb_img(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        try:
            buffer = BytesIO(img_bytes)
            img = Image.open(buffer)
            return img
        except:
            print(index)

    def _init_local_dataset(self, path):
        dataset_path = Path(path)
        self.img_paths = list(dataset_path.rglob('*.jpg')) + list(
            dataset_path.rglob('*.png'))
        self.img_paths = sorted(self.img_paths)

        self.length = len(self.img_paths)

        print(self.length, flush=True)

    def _read_local_img(self, index):
        img_path = self.img_paths[index]
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            print(img_path, flush=True)
        return img

    def _read_img(self, index):
        # return self._read_lmdb_img(index)
        return self._read_local_img(index)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img = self._read_img(index)

        if random.random() > 0.5:
            img = TF.hflip(img)

        thumb_img = img.resize((self.nerf_resolution, self.nerf_resolution),
                               Image.HAMMING)
        img = self.transform(img)
        thumb_img = self.transform(thumb_img)

        return dict_of(img, thumb_img)


class MultiResolutionDatasetLMS(MultiResolutionDataset):

    def __init__(self,
                 path,
                 lms_path,
                 transform,
                 resolution=256,
                 nerf_resolution=64):
        super().__init__(path, transform, resolution, nerf_resolution)

        self.lms_path_root = None

    def __getitem__(self, index):

        img = self._read_img(index)
        img_idx = str(index).zfill(5)
        # with self.env.begin(write=False) as txn:
        #     img_idx = str(index).zfill(5)
        #     key = f'{self.resolution}-{img_idx}'.encode('utf-8')
        #     img_bytes = txn.get(key)

        # # try:
        #     buffer = BytesIO(img_bytes)
        #     img = Image.open(buffer)

        if self.lms_path_root is not None:
            lms_np = self.lms[img_idx].copy()  # self.num_landmarks, 2
            lms_np = np.concatenate(
                [lms_np, np.ones(
                    (self.num_landmarks, 1))], axis=-1)  # concat visible dim

            lms = self.heatmap_generator(lms_np, self.resolution)
        # except:
        #     print(index)

        if random.random() > 0.5:
            img = TF.hflip(img)
            if self.lms_path_root is not None:
                lms = np.flip(lms, axis=-1).copy()  # avoid retuen a view here

        thumb_img = img.resize((self.nerf_resolution, self.nerf_resolution),
                               Image.HAMMING)
        img = self.transform(img)
        thumb_img = self.transform(thumb_img)
        rec_dict = dict(img=img, thumb_img=thumb_img)
        if self.lms_path_root is not None:
            rec_dict.update(dict(lms=lms))

        # return dict_of(img, thumb_img, lms, thumb_lms)
        return rec_dict


class SingleResolutionDataset(Dataset):

    def __init__(self, path, transform, resolution=256, nerf_resolution=64):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(
                txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.nerf_resolution = nerf_resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        try:
            buffer = BytesIO(img_bytes)
            img = Image.open(buffer)
        except:
            print(index)

        if random.random() > 0.5:
            img = TF.hflip(img)

        thumb_img = img.resize((self.nerf_resolution, self.nerf_resolution),
                               Image.HAMMING)
        img = self.transform(img)
        thumb_img = self.transform(thumb_img)

        return img, thumb_img


class MultiResolutionDatasetNaive(Dataset):

    def __init__(self, path, transform, resolution=256, nerf_resolution=64):
        self.resolution = resolution
        self.nerf_resolution = nerf_resolution
        self.transform = transform
        # self.img_paths = sorted(Path(path).rglob('*.png'))
        dataset_path = Path(path)
        self.img_paths = list(dataset_path.rglob('*.jpg')) + list(
            dataset_path.rglob('*.png'))
        self.img_paths = sorted(self.img_paths)

        self.length = len(self.img_paths)
        print(self.length, flush=True)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        try:
            img = Image.open(img_path)
        except:
            print(img_path, flush=True)

        if random.random() > 0.5:
            img = TF.hflip(img)

        orig_img = img.resize((self.resolution, self.resolution),
                              Image.HAMMING)
        thumb_img = img.resize((self.nerf_resolution, self.nerf_resolution),
                               Image.HAMMING)
        orig_img = self.transform(orig_img)
        thumb_img = self.transform(thumb_img)

        return orig_img, thumb_img


# todo, eval dataset
class ImagesDatasetEval(Dataset):

    def __init__(
            self,
            source_root,
            transform=None,
            # img_name_order=False
            img_name_order=True
        #  source_transform=None
    ):
        from project.utils.utils_coach import data_utils

        try:
            self.source_paths = sorted(
                data_utils.make_dataset(source_root),
                key=lambda x: int(x.split('/')[-1].split('.')[0]
                                  ))  # img name order
        except:
            self.source_paths = sorted(
                data_utils.make_dataset(source_root))  # img name order

        # else:
        #     self.source_paths = sorted(data_utils.make_dataset(source_root)) # img name order
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
                                     inplace=True)
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        from_path = self.source_paths[index]
        from_im = Image.open(from_path).convert('RGB')

        # if self.source_transform:
        from_im = self.transform(from_im)

        # return from_im, from_path
        return dict(image=from_im, img_path=from_path)


class MultiResolutionDataset_CelebA(Dataset):

    def __init__(self, path, transform, resolution=256, nerf_resolution=64):
        self.resolution = resolution
        self.nerf_resolution = nerf_resolution
        self.transform = transform
        # self.img_paths = sorted(Path(path).rglob('*.png'))
        self.img_paths = sorted(Path(path).rglob('*.jpg')) + sorted(
            Path(path).rglob('*.png'))
        self.length = len(self.img_paths)
        self.thumb_resize = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path)
        # if random.random() > 0.5:
        #     img = TF.hflip(img)
        img = self.transform(img)  # align first

        orig_img = TF.resize(img, (self.resolution, self.resolution))
        thumb_img = TF.resize(img,
                              (self.nerf_resolution, self.nerf_resolution))

        # orig_img = img.resize((self.resolution, self.resolution),
        #                       Image.HAMMING)
        # thumb_img = img.resize((self.nerf_resolution, self.nerf_resolution),
        #                        Image.HAMMING)
        # orig_img = self.transform(orig_img)
        # thumb_img = self.transform(thumb_img)

        return orig_img, thumb_img


# for shapenet
def load_pose(filename):
    with open(filename) as f:
        lines = f.read().splitlines()
        if len(lines) == 1:
            pose = np.zeros((4, 4), dtype=np.float32)
            for i in range(16):
                pose[i // 4, i % 4] = lines[0].split(" ")[i]
            return pose.squeeze()
        else:
            lines = [[x[0], x[1], x[2], x[3]]
                     for x in (x.split(" ") for x in lines[:4])]
            return np.asarray(lines).astype(np.float32).squeeze()


class MultiResolutionDataset_ShapeNet(Dataset):

    def __init__(self, path, transform, resolution=256, nerf_resolution=64):
        self.resolution = resolution
        self.nerf_resolution = nerf_resolution
        self.transform = transform
        self.root = Path(path).parent  # root of the lsit
        with open(path) as f:
            self.img_paths = [
                str(self.root / x.strip()) for x in f.readlines()
            ]
        # self.img_paths = path.rglob('*.png')

        self.length = len(self.img_paths)
        self.thumb_resize = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)  # align first

        # load pose
        pose_file = Path(img_path).parent.parent / 'pose' / (
            Path(img_path).stem + '.txt')
        poses = load_pose(pose_file)

        if True:
            poses[:3, :3] = np.identity(3)
            poses[:3, 3] = np.array([0, 0, 0.])

        extrinsics = np.linalg.inv(poses)

        poses = poses[..., :3, :4]
        extrinsics = extrinsics[..., :3, :4]

        # ! debug
        # if True:
        #     poses[:3, :3] = np.identity(3)
        #     extrinsics[:3, :3] = np.identity(3)

        orig_img = TF.resize(img, (self.resolution, self.resolution))
        # thumb_img = TF.resize(img, (self.nerf_resolution, self.nerf_resolution))
        return dict(image=orig_img,
                    img_path=img_path,
                    cam_settings=dict(poses=poses, extrinsics=extrinsics))

        # return orig_img, thumb_img
