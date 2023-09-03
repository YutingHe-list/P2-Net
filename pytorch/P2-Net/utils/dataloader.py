from os.path import join
from os import listdir
import SimpleITK as sitk
from torch.utils import data
import numpy as np
import nibabel as nib
import pandas as pd


def load_nii(path):
    image = nib.load(path)
    affine = image.affine

    image = image.get_data()

    return image, affine


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.nii', '.mhd'])


def calc_at_risk(X, T, O):
    order = np.argsort(T.astype('float64'))
    sorted_T = T[order]
    # at_risk = np.asarray([list(sorted_T).index(x) for x in sorted_T]).astype('int32')

    O = O[order]
    X = X[order]

    return X, O

class DatasetFromFolder3D(data.Dataset):
    def __init__(self, dataset_csv_file, source_image_dir):
        super(DatasetFromFolder3D, self).__init__()
        self.dataset_csv_file = dataset_csv_file

        self.df = pd.read_csv(dataset_csv_file)

        self.source_image_dir = source_image_dir
        self.image_filenames = [x for x in listdir(join(source_image_dir, "cutImage")) if is_image_file(x)]

        self.labels = self.df[['event', 'time']].values
        MPA = self.df['MPA'].values
        self.MPA_mean = np.mean(MPA)
        self.MPA_std = np.std(MPA)

        self.batch_x = self.df['CT'].values

        self.batch_x1, *batch_y_ = calc_at_risk(self.batch_x, self.labels[:, 1], self.labels[:, 0])

    def __getitem__(self, index):
        image = sitk.GetArrayFromImage(
            sitk.ReadImage(self.source_image_dir + 'cutImage/' + str(self.batch_x1[index]) + '.mhd'))
        target = sitk.GetArrayFromImage(
            sitk.ReadImage(self.source_image_dir + 'cutLabel/' + str(self.batch_x1[index]) + '_label.nii'))

        image = image.astype(np.float32)
        target = target.astype(np.float32)
        target = np.where(target > 0, 1, 0)
        image = np.where(image < 0., 0., image)
        image = np.where(image > 2048., 2048., image)
        image = image / 2048.

        image = image[np.newaxis, :, :, :]
        target = target[np.newaxis, :, :, :]


        image = image * target


        events = self.df[self.df['CT'] == self.batch_x1[index]]['event'].values
        MPA = self.df[self.df['CT'] == self.batch_x1[index]]['MPA'].values
        MPA = (MPA - self.MPA_mean) / self.MPA_std
        times = self.df[self.df['CT'] == self.batch_x1[index]]['time'].values


        image = image.astype(np.float32)
        MPA = MPA.astype(np.float32)
        times = times.astype(np.float32)
        events = events.astype(np.float32)

        return image, events, MPA, times

    def __len__(self):
        return (len(open(self.dataset_csv_file).readlines()) - 1)
