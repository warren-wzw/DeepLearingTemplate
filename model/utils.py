import os
import torch
import tqdm
import json
import math
import random
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from PIL import Image
from datetime import datetime
from torch.utils.data import  Dataset
from torch.optim.lr_scheduler import LambdaLR

"""Model info"""
def PrintModelInfo(model):
    """Print the parameter size and shape of model detail"""
    total_params = 0
    model_parments = sum(p.numel() for p in model.parameters() if p.requires_grad)
    for name, param in model.named_parameters():
        num_params = torch.prod(torch.tensor(param.shape)).item() * param.element_size() / (1024 * 1024)  # 转换为MB
        print(f"{name}: {num_params:.4f} MB, Shape: {param.shape}")
        total_params += num_params
    print(f"Traing parments {model_parments/1e6}M,Model Size: {total_params:.4f} MB")     
'''Log'''
def RecordLog(logfilename,message):
    """input:message=f"MODEL NAME:{model_name},EPOCH:{},Traing-Loss:{.3f},Acc:{(:.3f} %,Val-Acc:{:.3f} %" """
    with open(logfilename, 'a', encoding='utf-8') as logfile:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logfile.write(f"[{timestamp}] {message}\n")

"""Lr"""
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)

def ConsineAnnealing(optimizer,epochs,lrf=0.0001):
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # cosine
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    return scheduler

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

"""Save and load model"""
from datetime import datetime
def save_ckpt(save_path,model_name,model,epoch_index,scheduler,optimizer):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save({'epoch': epoch_index + 1,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler':scheduler.state_dict()},
                        '%s%s' % (save_path,model_name))
    print("->Saving model {} at {}".format(save_path+model_name, 
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    
"""Device"""
def get_gpus(num=None):
    gpu_nums = torch.cuda.device_count()
    if isinstance(num, list):
        devices = [torch.device(f'cuda:{i}')for i in num if i < gpu_nums]
    else:
        devices = [torch.device(f'cuda:{i}')for i in range(gpu_nums)][:num]
    return devices if devices else [torch.device('cpu')]   
"""Evaluate model"""
def CaculateAcc(output,label):
    print()
"""optimizer"""
from torch.optim.optimizer import Optimizer, required
class PIDAccOptimizer_SI_AAdRMS(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).
    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
    Example:
        >>> optimizer = PIDAccOptimizer_SI_AAdRMS(model.parameters(), lr=0.1)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.
        Considering the specific case of Momentum, the update can be written as
        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v
        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.
        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form
        .. math::
             v = \rho * v + lr * g \\
             p = p - v
        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, beta1=0.999, beta2=0.9, eps=1e-8, momentum=0.1, dampening=0,
                 weight_decay=0, nesterov=False, kp=5., ki=0.4, kd=8.):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, beta1=beta1, beta2=beta2, eps=eps,
                        weight_decay=weight_decay, nesterov=nesterov, kp=kp, ki=ki, kd=kd)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(PIDAccOptimizer_SI_AAdRMS, self).__init__(params, defaults)
        self.k = 1

    def __setstate__(self, state):
        super(PIDAccOptimizer_SI_AAdRMS, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']
            kp = group['kp']
            ki = group['ki']
            kd = group['kd']
            beta1 = group['beta1']
            beta2 = group['beta2']
            eps = group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]

                    if 'square_avg' not in param_state:
                        square_avg = param_state['square_avg'] = torch.zeros_like(p.data)
                        square_avg.mul_(beta1).addcmul_(d_p, d_p, value=1 - beta1)
                    else:
                        square_avg = param_state['square_avg']
                        square_avg.mul_(beta1).addcmul_(d_p, d_p, value=1 - beta1)
                        self.k += 1
                    avg = square_avg.clone().detach().mul_((1 - beta1 ** 2) ** -1).sqrt().add_(eps)

                    if 'z_buffer' not in param_state:
                        z_buf = param_state['z_buffer'] = torch.zeros_like(p.data)
                        z_buf.add_(d_p, alpha=lr)
                    else:
                        z_buf = param_state['z_buffer']
                        z_buf.add_(d_p, alpha=lr)
                    correct_z_buf = z_buf.clone().detach().div_(avg)
                    
                    if 'y_buffer' not in param_state:
                        param_state['y_buffer'] = torch.zeros_like(p.data)
                        y_buf = param_state['y_buffer']
                        y_buf.addcdiv_(d_p, avg, value=-lr * (kp - momentum * kd)). \
                            add_(correct_z_buf, alpha=-ki * lr)
                        y_buf.mul_((1 + momentum * lr) ** -1)
                    else:
                        y_buf = param_state['y_buffer']
                        y_buf.addcdiv_(d_p, avg, value=-lr * (kp - momentum * kd)). \
                            add_(correct_z_buf, alpha=-ki * lr)
                        y_buf.mul_((1 + momentum * lr) ** -1)

                    d_p = torch.zeros_like(p.data).add_(y_buf, alpha=lr).addcdiv_(d_p, avg, value=-kd * lr)
                p.data.add_(d_p)

        return loss  
