import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

class DigitDataset(Dataset):
    def __init__(self, path):

        # 影像路徑
        self.x = []
        # 影像類別
        self.y = []

        # 列出所有類別 [0,1,2,3,4,5,6,7,8,9]
        categories = os.listdir(path)
        
        # 去每個類別的路徑底下，列出所有檔案名稱
        for category in categories:
            subPath = os.path.join(path, category)
            filenames = os.listdir(subPath)

            # 列出所有檔案名稱
            for filename in filenames:
                filePath = os.path.join(subPath, filename)
                self.x.append(filePath)
                self.y.append(int(category))

    def __getitem__(self, index):
        filePath = self.x[index]
        tag = self.y[index]

        # 獲取檔案路徑，使用cv2讀取影像
        image = cv2.imread(filePath)
        # 使用numpy把cv2的image轉換為陣列
        image_array = np.array(image, dtype=np.float32)
        img = image_array[:, :, 0]

        tag = to_categorical(tag, 10)
        return img, tag # X, Y

    def __len__(self):
        return len(self.x)


if __name__ == "__main__":

    encoded = to_categorical(2, 10)
    print(encoded)