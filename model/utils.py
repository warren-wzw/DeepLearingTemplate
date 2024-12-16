import os
import torch
import math
from datetime import datetime
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
"""Save and load model"""
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
