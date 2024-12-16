import os
import torch
import tqdm
import json
import random
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import  Dataset
"""Dataset"""   
def min_max_normalize(image):
    np_image = np.array(image).astype(np.float32)
    np_image = (np_image - np.min(np_image)) / (np.max(np_image) - np.min(np_image))
    return torch.tensor(np_image)

def visual_result(input,filename):
    if len(input.shape)==4:
        np_image = input[0].cpu().permute(1,2,0).numpy()  # 将通道维度移到最后
    elif len(input.shape)==3:
        np_image = input.cpu().permute(1,2,0).numpy()  # 将通道维度移到最后
    if np_image.min()<0:    
        np_image = np_image * 0.5 + 0.5  # 假设图像已归一化为[-1, 1]
    plt.imshow(np_image)
    plt.axis('off')
    plt.savefig(filename)  # 在绘制图像后保存  
    
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    image=transform(image)
    #visual_result(image,"out.jpg")
    return image
    
def PreprocessCacheData(data_path,label_path,cache_file,cache=False,shuffle=True):
    if cache ==True and cache_file is not None and os.path.exists(cache_file):
        print("Loading features from cached file ", cache_file)
        features = torch.load(cache_file)
    else:
        print("Creating features from dataset at ", data_path)
        def processdata(data_path):
            print()
            return 0
        
        data=processdata(data_path)
        images,labels = [],[]
        for i  in range(len(data)):
            image = data[i]
            label = data[i]
            images.append(image)
            labels.append(label)    
        features=[]
        total_iterations = len(images) 
        for image,label in tqdm.tqdm(zip(images,labels),total=total_iterations):
            processed_image=preprocess_image(image)
            feature={
                "images":processed_image,
                "label":label
            }
            features.append(feature)
 
        if shuffle:
            random.shuffle(features)
        if cache==True and not os.path.exists(cache_file):
            print("Saving features into cached file ", cache_file)
            torch.save(features, cache_file)
    return features

class CacheDataset(Dataset):
    def __init__(self,features,num_instances):
        self.feature=features
        self.num_instances=num_instances
    
    def __len__(self):
        return int(self.num_instances)
    
    def __getitem__(self, index):
        feature = self.feature[index]
        image=feature["images"]
        label=feature["label"]
        return image,label
    
class OnlineCacheDataset(Dataset):   #only when json is standard json form,it will speed up
    def __init__(self, root_dir: str, label_path: str, transform=None):
        images_dir = root_dir
        self.label_dict=[]
        assert os.path.exists(images_dir), f"Image directory '{images_dir}' not found."
        assert os.path.exists(label_path), f"Label file '{label_path}' not found."
        with open(label_path, "r") as json_file:
            self.label_dict=json.load(json_file)  # 存储每一行的 JSON 数据

        # 从 label_dict 中提取数据（假设 JSON 文件格式包含 "filename" 和 "label" 字段）
        self.img_paths = [os.path.join(images_dir, item['file']) for item in self.label_dict]
        self.img_labels = [value["label"] for value in self.label_dict]
        self.total_num = len(self.img_paths)
        self.labels = set(self.img_labels)
        self.transform = transform or transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return self.total_num

    def __getitem__(self, item):
        img = Image.open(self.img_paths[item])
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.img_paths[item]))
        label = self.img_labels[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)  # 将图像堆叠成一个张量
        labels = torch.as_tensor(labels)    # 转换标签为张量
        return images, labels
