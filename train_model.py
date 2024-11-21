import os
import sys
import torch
import tqdm 
import torch.nn as nn
os.chdir(sys.path[0])
os.environ["CUDA_VISIBLE_DEVICES"]='0'
from model.template import TEMPLATE
from torch.utils.data import (DataLoader)
from datetime import datetime
from model.utils import CacheDataset,OnlineCacheDataset,PreprocessCacheData,get_linear_schedule_with_warmup,\
    PrintModelInfo,CaculateAcc,save_ckpt
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter
    
LR=1e-5
EPOCH=200
BATCH_SIZE=100
CACHE=False
TENSORBOARDSTEP=500
TF_ENABLE_ONEDNN_OPTS=0
MODEL_NAME=f"model.ckpt"
LAST_MODEL_NAME=f"model_last.ckpt"
SAVE_PATH='./output/output_model/'
PRETRAINED_MODEL_PATH=SAVE_PATH+LAST_MODEL_NAME
Pretrain=False if PRETRAINED_MODEL_PATH ==" " else True
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""dataset"""
train_type="train"
data_path_train=f"./dataset/train/train"
cached_file=f"./dataset/cache/{train_type}.pt"
val_type="val"
data_path_val=f"./dataset/test/test"
cached_file_val=f"./dataset/cache/{val_type}.pt"

def CreateDataloader(image_path,label_path,cached_file):
    if CACHE:
        features = PreprocessCacheData(image_path,label_path,cached_file,cache=CACHE,shuffle=True)  
        num_features = len(features)
        num_train = int(1* num_features)
        train_features = features[:num_train]
        dataset = CacheDataset(features=train_features,num_instances=num_train)
        loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    else:
        dataset = OnlineCacheDataset(image_path,label_path,shuffle=True)
        num_work = min([os.cpu_count(), BATCH_SIZE if BATCH_SIZE > 1 else 0, 8])  # number of workers
        loader = DataLoader(dataset=dataset, 
                            batch_size=BATCH_SIZE, 
                            shuffle=True,
                            pin_memory=True,
                            num_workers=num_work,
                            collate_fn=dataset.collate_fn)
    return loader

def main():
    global_step=0
    """Define Model"""
    model=TEMPLATE()
    model.to(DEVICE)
    model_name=model.__class__.__name__
    PrintModelInfo(model)
    """Pretrain"""
    if Pretrain:
        model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH),strict=False)
    """Create dataloader"""
    dataloader_train=CreateDataloader(data_path_train,cached_file)
    dataloader_val=CreateDataloader(data_path_val,cached_file_val)
    total_steps = len(dataloader_train) * EPOCH
    """Loss function"""
    criterion = nn.CrossEntropyLoss()
    """Optimizer"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    """ Train! """
    best_accuarcy=0 
    model.train()
    torch.cuda.empty_cache()
    scheduler = get_linear_schedule_with_warmup(optimizer, 0.1 * total_steps , total_steps)
    tb_writer = SummaryWriter(log_dir='./output/tflog/') 
    print("  ************************ Running training ***********************")
    print("  Num Epochs = ", EPOCH)
    print("  Batch size per node = ", BATCH_SIZE)
    print("  Num examples = ", dataloader_train.sampler.data_source.num_instances)
    print(f"  Pretrained Model is ")
    print(f"  Save Model as {SAVE_PATH}")
    print("  ****************************************************************")
    start_time=datetime.now()
    for epoch_index in range(EPOCH):
        loss_sum=0
        sum_test_accuarcy=0
        train_iterator = tqdm.tqdm(dataloader_train, initial=0,desc="Iter", disable=False)
        for step, (image,label) in enumerate(train_iterator):
            image,label= image.to(DEVICE),label.to(DEVICE)
            optimizer.zero_grad()
            output=model(image)
            accuarcy=CaculateAcc()
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            model.zero_grad()
            loss_sum=loss_sum+loss.item()
            sum_test_accuarcy=sum_test_accuarcy+accuarcy
            current_lr= scheduler.get_last_lr()[0]
            """ tensorbooard """
            if  global_step % TENSORBOARDSTEP== 0 and tb_writer is not None:
                tb_writer.add_scalar('train/lr', current_lr, global_step=global_step)
                tb_writer.add_scalar('train/loss', loss.item(), global_step=global_step)
            global_step=global_step+1
            scheduler.step()
            train_iterator.set_description('Epoch=%d, Acc= %3.3f %%,loss=%.6f, lr=%9.7f' 
                                           % (epoch_index,(sum_test_accuarcy/(step+1))*100, loss_sum/(step+1), current_lr))
        """ validation """
        sum_accuarcy=0
        model.eval()
        with torch.no_grad():
            validation_iterator = tqdm.tqdm(dataloader_val, initial=0,desc="Iter", disable=False)
            for i,  (image,label) in enumerate(validation_iterator):
                image,label= image.to(DEVICE),label.to(DEVICE) 
                output=model(image)
                accuarcy=CaculateAcc()
                sum_accuarcy=sum_accuarcy+ accuarcy
                validation_iterator.set_description('ValAcc= %3.3f %%' % (sum_accuarcy*100/(i+1)))
        """save model"""
        if loss_sum/(step+1) < best_loss:
            best_loss = loss_sum/(step+1)
            save_ckpt(SAVE_PATH,model_name+'.ckpt',model,epoch_index,scheduler,optimizer)
        else:
            save_ckpt(SAVE_PATH,model_name+'_last.ckpt',model,epoch_index,scheduler,optimizer) 
    end_time=datetime.now()
    print("Training consume :",(end_time-start_time)/60,"minutes")
    
if __name__=="__main__":
    main()