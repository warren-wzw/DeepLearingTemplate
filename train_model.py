import os
import sys
os.chdir(sys.path[0])
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import torch
import tqdm 
import torch.nn as nn

from datetime import datetime
from model.template import TEMPLATE
from torch.utils.data import DataLoader
from model.utils import TemplateDataset
from model.utils import load_and_cache_withlabel,get_linear_schedule_with_warmup,\
    PrintModelInfo,CaculateAcc, load_checkpoint, save_checkpoint

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

LR=1e-5   
EPOCH=200 
BATCH_SIZE=100
TENSORBOARDSTEP=500
PRETRAINED_MODEL_PATH=" "
PRETRAINED=True if PRETRAINED_MODEL_PATH != "" and os.path.exists(PRETRAINED_MODEL_PATH) else False
SAVE_PATH='./output/output_model/'
MODEL_NAME=""
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""dataset"""
train_type="train"
data_path_train=f"./dataset/train/train"
cached_file=f"./dataset/cache/{train_type}.pt"
val_type="val"
data_path_val=f"./dataset/test/test"
cached_file_val=f"./dataset/cache/{val_type}.pt"

def CreateDataloader(image_path,cached_file):
    features = load_and_cache_withlabel(image_path,cached_file,shuffle=True)  
    num_features = len(features)
    num_features = int(num_features)
    train_features = features[:num_features]
    dataset = TemplateDataset(features=train_features,num_instances=num_features)
    loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    return loader

def main():
    global_step=0
    """Define Model"""
    model=TEMPLATE().to(DEVICE)
    PrintModelInfo(model)
    """Create dataloader"""
    dataloader_train=CreateDataloader(data_path_train,cached_file)
    dataloader_val=CreateDataloader(data_path_val,cached_file_val)
    total_steps = len(dataloader_train) * EPOCH
    """Loss function"""
    criterion = nn.CrossEntropyLoss()
    """Optimizer"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    """Lr"""
    scheduler = get_linear_schedule_with_warmup(optimizer, 0.1 * total_steps , total_steps)
    """tensorboard"""
    tb_writer = SummaryWriter(log_dir='./output/tflog/') 
    """Pretrain"""
    start_ep=0
    if PRETRAINED:
        ckpt = load_checkpoint(PRETRAINED_MODEL_PATH)
        start_ep = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
    """ Train! """
    model.train()
    best_accuarcy=0 
    start_time=datetime.now()
    print("  ************************ Running training ***********************")
    print("  Num Epochs = ", EPOCH)
    print("  Batch size per node = ", BATCH_SIZE)
    print("  Num examples = ", dataloader_train.sampler.data_source.num_instances)
    print(f"  Pretrained Model is {PRETRAINED_MODEL_PATH}")
    print(f"  Save Model as {SAVE_PATH+MODEL_NAME}")
    print("  ****************************************************************")
    for epoch_index in range(start_ep,EPOCH):
        loss_sum=0
        sum_test_accuarcy=0
        torch.cuda.empty_cache()
        train_iterator = tqdm.tqdm(dataloader_train, initial=0,desc="Iter", disable=False)
        for step, (image,label) in enumerate(train_iterator):
            image,label= image.to(DEVICE),label.to(DEVICE)
            optimizer.zero_grad()
            output=model(image)
            accuarcy=CaculateAcc(output,label)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            model.zero_grad()
            """cal loss and acc"""
            loss_sum=loss_sum+loss.item()
            sum_test_accuarcy=sum_test_accuarcy+accuarcy
            """ tensorbooard """
            current_lr= scheduler.get_last_lr()[0]
            if  global_step % TENSORBOARDSTEP== 0 and tb_writer is not None:
                tb_writer.add_scalar('train/lr', current_lr, global_step=global_step)
                tb_writer.add_scalar('train/loss', loss.item(), global_step=global_step)
            """show progress bar"""
            train_iterator.set_description('Epoch=%d, Acc= %3.3f %%,loss=%.6f, lr=%9.7f' 
                                           % (epoch_index,(sum_test_accuarcy/(step+1))*100, loss_sum/(step+1), current_lr))
            global_step=global_step+1
            scheduler.step()
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
        """save the best"""
        if sum_accuarcy/(i+1) > best_accuarcy:
            best_accuarcy = sum_accuarcy/(i+1)
            if not os.path.exists(SAVE_PATH):
                os.makedirs(SAVE_PATH)
            save_checkpoint({'epoch': epoch_index + 1,
                                'model': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'scheduler':scheduler.state_dict()},
                                '%s%s.ckpt' % (SAVE_PATH,MODEL_NAME),
                                max_keep=2) 
            print("->Saving model {} at {}".format(SAVE_PATH+MODEL_NAME, 
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    end_time=datetime.now()
    print("Training consume :",(end_time-start_time)/60,"minutes")
    
if __name__=="__main__":
    main()
