import os
import torch
from torch.utils.data import Dataset, DataLoader
import pickle

testMask_dir = 'features/Mask'


class Custom_Dataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform

        # 在初始化时一次性加载数据
        with open(data_path, 'rb') as f:
            self.mDATA = pickle.load(f)

        self.slide_names = list(self.mDATA.keys())

    def __getitem__(self, idx):
        tumorSlides = os.listdir(testMask_dir)
        tumorSlides = [sst.split('.')[0] for sst in tumorSlides]

        slide_name = self.slide_names[idx]
        patch_data_list = self.mDATA[slide_name]

        spatial_info = []
        # "tumor_046_24576_110080_48_215_WW_191_HH_431.png"
        for tpatch in patch_data_list:
            patch_name = tpatch['file_name'].split('.')
            parts = patch_name[0].split('_')
            num1 = int(parts[4])  # 提取位置5的数字，即 '48'
            num2 = int(parts[5])  # 提取位置6的数字，即 '215'
            num3 = int(parts[7])  # 提取位置8的数字，即 '191'
            num4 = int(parts[9])  # 提取位置10的数字，即 '431'
            mytuple = (num1, num2, num3, num4)
            spatial_info.append(mytuple)

        featGroup = [torch.from_numpy(tpatch['feature']).unsqueeze(0) for tpatch in patch_data_list]
        featGroup = torch.cat(featGroup, dim=0)

        if slide_name.startswith('tumor'):
            label = 1
        elif slide_name.startswith('normal'):
            label = 0
        else:
            # raise RuntimeError('Undefined slide type')
            if slide_name in tumorSlides:
                label = 1
            else:
                label = 0

        if self.transform:
            featGroup = self.transform(featGroup)

        return slide_name, featGroup, label, spatial_info

    def __len__(self):
        return len(self.slide_names)