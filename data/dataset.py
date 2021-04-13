import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
from detectron2.data import DatasetMapper

from util import constants as C


class ImageClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, image_path=None, labels=None, transforms=None):
        self._image_path = image_path
        self._labels = labels
        self._transforms = transforms

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, index):
        label = torch.tensor(np.float64(self._labels[index]))
        image = Image.open(self._image_path[index]).convert('RGB')
        if self._transforms is not None:
            image = self._transforms(image)

        return image, label


class ImageClassificationDemoDataset(ImageClassificationDataset):
    def __init__(self):
        super().__init__(image_path=C.TEST_IMG_PATH, labels=[
            0, 1], transforms=T.Compose([T.Resize((224, 224)), T.ToTensor()]))


class ImageDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, image_path=None, annotations=None, augmentations=None):
        self._image_path = image_path
        self._annotations = annotations
        self._mapper = DatasetMapper(is_train=True,
                                     image_format="RGB",
                                     augmentations=augmentations
                                     )

    def __len__(self):
        return len(self._annotations)

    def __getitem__(self, index):
        sample = {}
        sample['annotations'] = self._annotations[index]
        sample['file_name'] = self._image_path[index]
        sample['image_id'] = index
        sample = self._mapper(sample)
        return sample


class ImageDetectionDemoDataset(ImageDetectionDataset):
    def __init__(self):
        super().__init__(image_path=C.TEST_IMG_PATH,
                         annotations=[[{'bbox': [438, 254, 455, 271], 'bbox_mode': 0, 'category_id': 0},
                                       {'bbox': [388, 259, 408, 279], 'bbox_mode': 0, 'category_id': 1}]] * 2,
                         augmentations=[])
