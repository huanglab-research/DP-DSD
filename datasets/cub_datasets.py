from torchvision.datasets import ImageFolder
import os
from PIL import Image




class IndexedImageFolder(ImageFolder):
    """
    ImageFolder with index returned for each sample.
    Output: (image, target, index)
    """
    def __getitem__(self, index):
        #img, target = super(IndexedImageFolder, self).__getitem__(index)
        # return img, target, index

        path, target = self.samples[index]  # path 是相对路径，比如 '001.Black_footed_Albatross/xxx.jpg'

        # 加载图像
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        # 返回：图像tensor，标签，image_id（相对路径）
        image_id = os.path.relpath(path, self.root)  # 相对路径，例如 '001.Black_footed_Albatross/xxx.jpg'
        return img, target, image_id


    def load_image_by_id(self, image_id):
        """
        根据 image_id（相对路径字符串）加载单张图像，返回 transform 后的 tensor。
        image_id 示例: '001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg'
        """
        image_path = os.path.join(self.root, image_id)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image path {image_path} 不存在，请检查 image_id 是否正确。")

        img = Image.open(image_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img
