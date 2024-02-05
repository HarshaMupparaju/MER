import os 
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import time
from tempfile import TemporaryDirectory
import random
import time
import json
import tqdm as tqdm

from torchmetrics.classification import MulticlassF1Score, MulticlassRecall

torch.manual_seed(0)
cudnn.benchmark = True
plt.ion()   # interactive mode


def show_video(video, emotion):
    """Show video"""
    plt.figure()
    plt.title(emotion)
    plt.imshow(video)
    plt.show()



class CASME3Dataset(Dataset):
    def __init__(self, data_root, MAE_data, ME_data, expression_type, transform=None, sequence_length=16):
        self.data_root = data_root
        self.transform = transform
        self.sequence_length = sequence_length
        self.expression_type = expression_type
        #Load annotations
        self.annotations_macro = MAE_data
        self.annotations_micro = ME_data
        #TODO:Figure out a way to merge both annotations
        if self.expression_type == 'macro':
            self.annotations = self.annotations_macro
        else:    
            self.annotations = self.annotations_micro

        
        # sps = ['spNO.1', 'spNO.2', 'spNO.3', 'spNO.4', 'spNO.5', 'spNO.6', 'spNO.7']
        # #Filter out subjects not in sps
        # self.annotations = self.annotations[self.annotations['Subject'].isin(sps)]

        # Filter out sequences with less than 16 frames
        self.filter_sequences()
        # Define a mapping from string labels to numerical values
        self.label_mapping = {
            'disgust': 0,
            'surprise': 1,
            'happy': 2,
            'fear': 3,
            'anger': 4,
            'sad': 5,
            'others': 6,
            'Others': 6,
            'happiness': 2,
            'sadness': 5,

        }

    def __len__(self):
        return len(self.annotations)
    

    def __getitem__(self, index):
        subject = self.annotations.iloc[index]['Subject']
        sequence = self.annotations.iloc[index]['Filename']
        onset_frame = self.annotations.iloc[index]['Onset']
        apex_frame = self.annotations.iloc[index]['Apex']
        offset_frame = self.annotations.iloc[index]['Offset']
        emotion = self.annotations.iloc[index]['emotion']

        emotion = self.label_mapping[emotion]


        # print(f'{subject} {sequence} {onset_frame} {apex_frame} {offset_frame} {emotion}')

        #Picking 13 frames randomly 
        video_frames = []

        while len(video_frames) < 13:
            potential_frame = random.randint(onset_frame+1, offset_frame)
            potential_rgb_frame_path = self.data_root / subject / sequence / 'color' / f'{potential_frame}.jpg'
            potential_depth_frame_path = self.data_root / subject / sequence / 'depth' / f'{potential_frame}.png'
            if (potential_frame not in video_frames) and (potential_frame != apex_frame) and (potential_rgb_frame_path.is_file()) and (potential_depth_frame_path.is_file()):
                video_frames.append(potential_frame)
        


        # #Picking 16 frames around apex frame
        # video_frames = random.sample(range(onset_frame+1, offset_frame), 13)
        onset_frame_temp = onset_frame
        while len(video_frames) < 14:
            potential_onset_rgb_frame_path = self.data_root / subject / sequence / 'color' / f'{onset_frame_temp}.jpg'
            potential_onset_depth_frame_path = self.data_root / subject / sequence / 'depth' / f'{onset_frame_temp}.png'
            if (onset_frame_temp not in video_frames) and (potential_onset_rgb_frame_path.is_file()) and (potential_onset_depth_frame_path.is_file()):
                video_frames.append(onset_frame_temp)
            onset_frame_temp += 1

        offset_frame_temp = offset_frame

        while len(video_frames) < 15:
            potential_offset_rgb_frame_path = self.data_root / subject / sequence / 'color' / f'{offset_frame_temp}.jpg'
            potential_offset_depth_frame_path = self.data_root / subject / sequence / 'depth' / f'{offset_frame_temp}.png'
            if (offset_frame_temp not in video_frames) and (potential_offset_rgb_frame_path.is_file()) and (potential_offset_depth_frame_path.is_file()):
                video_frames.append(offset_frame_temp)
            offset_frame_temp -= 1

        apex_frame_temp = apex_frame



        while len(video_frames) < 16:
            potential_apex_rgb_frame_path = self.data_root / subject / sequence / 'color' / f'{apex_frame_temp}.jpg'
            potential_apex_depth_frame_path = self.data_root / subject / sequence / 'depth' / f'{apex_frame_temp}.png'
            if (apex_frame_temp not in video_frames) and (potential_apex_rgb_frame_path.is_file()) and (potential_apex_depth_frame_path.is_file()):
                video_frames.append(apex_frame_temp)
            if(apex_frame - onset_frame > offset_frame - apex_frame):
                apex_frame_temp -= 1
            else:
                apex_frame_temp += 1

           

        #Adding onset, apex and offset frames
        # video_frames.append(onset_frame)
        # video_frames.append(apex_frame)
        # video_frames.append(offset_frame)
        video_frames.sort()



        depth_video_frames = np.array([np.array(Image.open(self.data_root / subject / sequence / 'depth' / f'{frame}.png')) for frame in video_frames])

        depth_video_frames = depth_video_frames.astype(np.uint8)
        depth_video_frames = depth_video_frames.reshape(depth_video_frames.shape[0], depth_video_frames.shape[1], depth_video_frames.shape[2], 1)


        




        numpy_video_frames = np.array([np.array(Image.open(self.data_root / subject / sequence / 'color' / f'{frame}.jpg')) for frame in video_frames])
        


        numpy_depth_color_frames = np.concatenate((numpy_video_frames, depth_video_frames), axis=-1)

        numpy_video_frames = numpy_depth_color_frames




        
        torch_video_frames = torch.from_numpy(numpy_video_frames)
        torch_video_frames = torch_video_frames.permute(0, 3, 1, 2)
        # y_label = torch.tensor(float(emotion))
        y_label = torch.tensor(emotion)

        
        if len(video_frames) >= 16:
            if self.transform:
                torch_video_frames = self.transform(torch_video_frames)

            return torch_video_frames, y_label
        else:
            return None, None
        
    def filter_sequences(self):
        self.annotations = self.annotations[
            (self.annotations['Offset'] - self.annotations['Onset'] + 1) >= 16]


        if(self.expression_type == 'micro'):
            # self.annotations.remove(self.annotations[self.annotations['Subject'] == 'spNO.171'] & self.annotations[self.annotations['Filename'] == 'e'])
            # self.annotations.drop(self.annotations[self.annotations['Subject'] == 'spNO.171'] & self.annotations[self.annotations['Filename'] == 'e'], inplace=True)
            a = self.annotations['Subject'] == 'spNO.171'
            b = self.annotations['Filename'] == 'e'
            c = self.annotations['Onset'] == 2403

            self.annotations.drop(self.annotations[a & b & c].index, inplace=True)

            a = self.annotations['Subject'] == 'spNO.171'
            b = self.annotations['Filename'] == 'h'
            c = self.annotations['Onset'] == 1340

            self.annotations.drop(self.annotations[a & b & c].index, inplace=True)

            a = self.annotations['Subject'] == 'spNO.208'
            b = self.annotations['Filename'] == 'c'
            c = self.annotations['Onset'] == 2334

            self.annotations.drop(self.annotations[a & b & c].index, inplace=True)

            a = self.annotations['Subject'] == 'spNO.215'
            b = self.annotations['Filename'] == 'j'
            c = self.annotations['Onset'] == 463

            self.annotations.drop(self.annotations[a & b & c].index, inplace=True)

            a = self.annotations['Subject'] == 'spNO.216'
            b = self.annotations['Filename'] == 'e'
            c = self.annotations['Onset'] == 0

            self.annotations.drop(self.annotations[a & b & c].index, inplace=True)


            #Depth Missing Frame cases

            # a = self.annotations['Subject'] == 'spNO.214'
            # b = self.annotations['Filename'] == 'c'
            # c = self.annotations['Offset'] == 3460     #Missing frame case

            # self.annotations.drop(self.annotations[a & b & c].index, inplace=True)    

            
        
        if(self.expression_type == 'macro'):
            # self.annotations.remove(self.annotations[self.annotations['Subject'] == 'spNO.171'] & self.annotations[self.annotations['Filename'] == 'e'])
            # self.annotations.drop(self.annotations[self.annotations['Subject'] == 'spNO.171'] & self.annotations[self.annotations['Filename'] == 'e'], inplace=True)
            a = self.annotations['Subject'] == 'spNO.150'
            b = self.annotations['Filename'] == 'h'
            c = self.annotations['Onset'] == 2239

            self.annotations.drop(self.annotations[a & b & c].index, inplace=True)

            a = self.annotations['Subject'] == 'spNO.151'
            b = self.annotations['Filename'] == 'd'
            c = self.annotations['Onset'] == 2069

            self.annotations.drop(self.annotations[a & b & c].index, inplace=True)

            a = self.annotations['Subject'] == 'spNO.148'
            b = self.annotations['Filename'] == 'j'
            c = self.annotations['Onset'] == 868

            self.annotations.drop(self.annotations[a & b & c].index, inplace=True)

            a = self.annotations['Subject'] == 'spNO.148'
            b = self.annotations['Filename'] == 'j'
            c = self.annotations['Onset'] == 945

            self.annotations.drop(self.annotations[a & b & c].index, inplace=True)

            a = self.annotations['Subject'] == 'spNO.148'
            b = self.annotations['Filename'] == 'j'
            c = self.annotations['Apex'] == 1111

            self.annotations.drop(self.annotations[a & b & c].index, inplace=True)

            a = self.annotations['Subject'] == 'spNO.167'
            b = self.annotations['Filename'] == 'b'
            c = self.annotations['Apex'] == 226

            self.annotations.drop(self.annotations[a & b & c].index, inplace=True)

            a = self.annotations['Subject'] == 'spNO.171'
            b = self.annotations['Filename'] == 'f'
            c = self.annotations['Onset'] == 1404 

            self.annotations.drop(self.annotations[a & b & c].index, inplace=True)    

            a = self.annotations['Subject'] == 'spNO.186'
            b = self.annotations['Filename'] == 'a'
            c = self.annotations['Apex'] == 53

            self.annotations.drop(self.annotations[a & b & c].index, inplace=True)

            a = self.annotations['Subject'] == 'spNO.186'
            b = self.annotations['Filename'] == 'a'
            c = self.annotations['Onset'] == 246

            self.annotations.drop(self.annotations[a & b & c].index, inplace=True)

            a = self.annotations['Subject'] == 'spNO.186'
            b = self.annotations['Filename'] == 'k'
            c = self.annotations['Apex'] == 340

            self.annotations.drop(self.annotations[a & b & c].index, inplace=True)            

            a = self.annotations['Subject'] == 'spNO.203'
            b = self.annotations['Filename'] == 'm'
            c = self.annotations['Apex'] == 436

            self.annotations.drop(self.annotations[a & b & c].index, inplace=True)   

            a = self.annotations['Subject'] == 'spNO.207'
            b = self.annotations['Filename'] == 'j'
            c = self.annotations['Apex'] == 702

            self.annotations.drop(self.annotations[a & b & c].index, inplace=True)  

            a = self.annotations['Subject'] == 'spNO.210'
            b = self.annotations['Filename'] == 'd'
            c = self.annotations['Apex'] == 460

            self.annotations.drop(self.annotations[a & b & c].index, inplace=True)   

            a = self.annotations['Subject'] == 'spNO.210'
            b = self.annotations['Filename'] == 'e'
            c = self.annotations['Apex'] == 1672

            self.annotations.drop(self.annotations[a & b & c].index, inplace=True)  

            a = self.annotations['Subject'] == 'spNO.210'
            b = self.annotations['Filename'] == 'h'
            c = self.annotations['Onset'] == 2849

            self.annotations.drop(self.annotations[a & b & c].index, inplace=True)

            a = self.annotations['Subject'] == 'spNO.215'
            b = self.annotations['Filename'] == 'd'
            c = self.annotations['Apex'] == 3734

            self.annotations.drop(self.annotations[a & b & c].index, inplace=True) 

            a = self.annotations['Subject'] == 'spNO.215'
            b = self.annotations['Filename'] == 'g'
            c = self.annotations['Apex'] == 2096

            self.annotations.drop(self.annotations[a & b & c].index, inplace=True) 

            a = self.annotations['Subject'] == 'spNO.215'
            b = self.annotations['Filename'] == 'l'
            c = self.annotations['Onset'] == 0     #Missing frame case

            self.annotations.drop(self.annotations[a & b & c].index, inplace=True) 



            #Depth Missing Frame cases

            # a = self.annotations['Subject'] == 'spNO.11'
            # b = self.annotations['Filename'] == 'd'
            # c = self.annotations['Offset'] == 3652     

            # self.annotations.drop(self.annotations[a & b & c].index, inplace=True) 


            # a = self.annotations['Subject'] == 'spNO.207'
            # b = self.annotations['Filename'] == 'i'
            # c = self.annotations['Offset'] == 4328    

            # self.annotations.drop(self.annotations[a & b & c].index, inplace=True) 

            # a = self.annotations['Subject'] == 'spNO.146'
            # b = self.annotations['Filename'] == 'e'
            # c = self.annotations['Offset'] == 3084    

            # self.annotations.drop(self.annotations[a & b & c].index, inplace=True) 

            # a = self.annotations['Subject'] == 'spNO.157'
            # b = self.annotations['Filename'] == 'a'
            # c = self.annotations['Offset'] == 1024    

            # self.annotations.drop(self.annotations[a & b & c].index, inplace=True) 

            # a = self.annotations['Subject'] == 'spNO.159'
            # b = self.annotations['Filename'] == 'a'
            # c = self.annotations['Offset'] == 1025  

            # self.annotations.drop(self.annotations[a & b & c].index, inplace=True) 

            # a = self.annotations['Subject'] == 'spNO.160'
            # b = self.annotations['Filename'] == 'a'
            # c = self.annotations['Offset'] == 1021

            # self.annotations.drop(self.annotations[a & b & c].index, inplace=True) 

            # a = self.annotations['Subject'] == 'spNO.160'
            # b = self.annotations['Filename'] == 'c'
            # c = self.annotations['Offset'] == 3472

            # self.annotations.drop(self.annotations[a & b & c].index, inplace=True) 

            # a = self.annotations['Subject'] == 'spNO.160'
            # b = self.annotations['Filename'] == 'f'
            # c = self.annotations['Offset'] == 2316

            # self.annotations.drop(self.annotations[a & b & c].index, inplace=True) 



def train_one_epoch(model, criterion, optimizer, train_loader, train_dataset, writer, device, epoch_number):
    running_loss = 0.0
    last_loss = 0.0

    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        # print(outputs)
        # print(labels)
        # print(1/0)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()


    last_loss = running_loss / len(train_dataset)
    print(f'Training Loss: {last_loss}')
    return last_loss



def train_model(model, criterion, optimizer, train_loader, train_dataset, test_loader, test_dataset, writer, device, num_epochs=25):
    since = time.time()


    best_vloss = 100000000
    best_vacc = 0
    best_vf1 = 0
    best_vrecall = 0

    for epoch in tqdm.tqdm(range(num_epochs)):
        print(f'Epoch {epoch}/{num_epochs-1}')

        model.train()
        avg_loss = train_one_epoch(model, criterion, optimizer, train_loader, train_dataset, writer, device, epoch)

        running_vloss = 0.0
        running_vcorrects = 0
        targets = []
        preds = []


        model.eval()

        with torch.no_grad():
            for i, data in enumerate(test_loader):
                vinputs, vlabels = data
                vinputs = vinputs.to(device)
                vlabels = vlabels.to(device)
                
                voutputs = model(vinputs)
                _, vpreds = torch.max(voutputs, 1)

                preds.append(vpreds)
                targets.append(vlabels.data)

                vloss = criterion(voutputs, vlabels)
                running_vloss += vloss.item()

                running_vcorrects += torch.sum(vpreds == vlabels.data)

        avg_vloss = running_vloss / len(test_dataset)
        avg_vacc = running_vcorrects.double() / len(test_dataset)
        f1 = MulticlassF1Score(num_classes=7, average='macro').to(device)
        recall = MulticlassRecall(num_classes=7, average='macro').to(device)

        predictions = torch.cat(preds)
        targets = torch.cat(targets)


        unweighted_f1_score = f1(predictions, targets)
        unweighted_recall = recall(predictions, targets)

        print(f'Validation Loss: {avg_vloss} Acc: {avg_vacc} UF1: {unweighted_f1_score} UAR: {unweighted_recall}')

        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Loss/val', avg_vloss, epoch)
        
        writer.flush()

        if(avg_vloss < best_vloss):
            best_vloss = avg_vloss
            torch.save(model.state_dict(), f'train/train{train_num:04}/model.pt')

            metrics_dict = {
                'Cross Entropy Loss': round(avg_vloss,4),
                'Accuracy': round(avg_vacc.item(),4),
                'Unweighted F1 Score': round(unweighted_f1_score.item(),4),
                'Unweighted Recall': round(unweighted_recall.item(),4)
            }

            json.dump(metrics_dict, open(f'train/train{train_num:04}/metrics.json', 'w'))


    
    end = time.time()
    print(f'Training complete in {(end-since)/60} minutes')






def common_subjects_macro_micro(annotations_macro, annotations_micro):
    macro_subjects = annotations_macro['Subject'].unique()
    micro_subjects = annotations_micro['Subject'].unique()

    common_subjects = [subject for subject in macro_subjects if subject in micro_subjects]

    return common_subjects

def train_test_subjects_split(common_subjects):
    train_subjects = random.sample(list(common_subjects), int(0.8*len(common_subjects)))
    test_subjects = [subject for subject in common_subjects if subject not in train_subjects]

    return train_subjects, test_subjects

def train_test_macro_micro_split(annotations_macro, annotations_micro, train_subjects, test_subjects):
    train_data_macro = annotations_macro[annotations_macro['Subject'].isin(train_subjects)]
    test_data_macro = annotations_macro[annotations_macro['Subject'].isin(test_subjects)]

    train_data_micro = annotations_micro[annotations_micro['Subject'].isin(train_subjects)]
    test_data_micro = annotations_micro[annotations_micro['Subject'].isin(test_subjects)]

    return train_data_macro, test_data_macro, train_data_micro, test_data_micro


def main(train_num):

    writer = SummaryWriter(f'train/train{train_num:04}')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_root = Path('/media/harshamupparaju/Expansion/Harsha/Databases/CASME3/part_A/data/Compressed_version1_seperate_compress')
    MAE_annotations_path = Path('/media/harshamupparaju/Expansion/Harsha/Databases/CASME3/part_A/annotation/cas(me)3_part_A_MaE_label_JpgIndex_v2_emotion.xlsx')
    ME_annotations_path = Path('/media/harshamupparaju/Expansion/Harsha/Databases/CASME3/part_A/annotation/cas(me)3_part_A_ME_label_JpgIndex_v2.xlsx')


    annotations_macro = pd.read_excel(MAE_annotations_path, sheet_name=None)['Sheet1']
    annotations_macro.rename(columns={'sub': 'Subject', 'seq': 'Filename', 'onset': 'Onset', 'apex': 'Apex', 'offset': 'Offset', 'emotion': 'emotion'}, inplace=True)
    annotations_micro = pd.read_excel(ME_annotations_path, sheet_name='label')


    common_subjects = common_subjects_macro_micro(annotations_macro, annotations_micro)

    train_subjects, test_subjects = train_test_subjects_split(common_subjects)

    train_data_macro, test_data_macro, train_data_micro, test_data_micro = train_test_macro_micro_split(annotations_macro, annotations_micro, train_subjects, test_subjects)

    a = models.video.MViT_V2_S_Weights.KINETICS400_V1.transforms()
    a.mean.append(0.45)
    a.std.append(0.225)
    # Add Random Rotation 

    data_transforms = transforms.Compose([
        transforms.RandomRotation((-90, 90)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        a
        ])
    # data_transforms = transforms.Compose([
    #     models.video.MViT_V2_S_Weights.KINETICS400_V1.transforms()
    # ])



    #Initialize the micro dataset
    train_dataset_micro = CASME3Dataset(data_root, train_data_macro, train_data_micro, expression_type='micro', transform=data_transforms)
    test_dataset_micro = CASME3Dataset(data_root, test_data_macro, test_data_micro, expression_type='micro', transform=data_transforms)

    train_loader_micro = DataLoader(train_dataset_micro, batch_size=1  , shuffle=True, num_workers=1)
    test_loader_micro = DataLoader(test_dataset_micro, batch_size=1  , shuffle=False, num_workers=1)

    #Initialize the macro dataset
    train_dataset_macro = CASME3Dataset(data_root, annotations_macro, train_data_micro, expression_type='macro', transform=data_transforms)
    # train_dataset_macro = CASME3Dataset(data_root, train_data_macro, train_data_micro, expression_type='macro', transform=data_transforms)
    test_dataset_macro = CASME3Dataset(data_root, test_data_macro, test_data_micro, expression_type='macro', transform=data_transforms)

    train_loader_macro = DataLoader(train_dataset_macro, batch_size=8  , shuffle=True, num_workers=8)
    test_loader_macro = DataLoader(test_dataset_macro, batch_size=8  , shuffle=False, num_workers=8)

    #Load model
    # model = models.video.mvit_v2_s(weights='DEFAULT')
    # print(model)
    # print(1/0)


    # #Freeze all layers
    # for param in model.parameters():
    #     param.requires_grad = False

    # #Unfreeze all layers
    # # for param in model.parameters():
    # #     param.requires_grad =True

    # # Unfreeze last 3 blocks
    # for i in range(13,16):
    #     for param in model.blocks[i].parameters():
    #         param.requires_grad = True

    # #Unfreeze classifier
    # for param in model.head.parameters():
    #     param.requires_grad = True

    # model.conv_proj = nn.Conv3d(4, 96, kernel_size=(3, 7, 7), stride=(2, 4, 4), padding=(1, 3, 3))

    # num_features = model.head[1].in_features
    # model.head[1] = torch.nn.Linear(num_features, 7) 

    # model = model.to(device)
    # # print(model)
    # # print(1/0)

    # criterion = nn.CrossEntropyLoss()

    # # Observe that all parameters are being optimized
    # # optimizer = optim.SGD(model.parameters(), lr=1e-5)
    # optimizer = optim.Adam(model.parameters(), lr=1e-5)

    # # Decay LR by a factor of 0.1 every 7 epochs
    # # exp_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # #Finetuning on Macro
    # # model = train_model(model, criterion, optimizer, exp_scheduler, train_loader_macro, train_dataset_macro, test_loader_macro, test_dataset_macro, writer, device, num_epochs=25)


    # # model = train_model(model, criterion, optimizer, exp_scheduler, train_loader_micro, train_dataset_micro, test_loader_micro, test_dataset_micro, writer, device, num_epochs=25)
    # model = train_model(model, criterion, optimizer, train_loader_micro, train_dataset_micro, test_loader_micro, test_dataset_micro, writer, device, num_epochs=500)

    
    # # torch.save(model, f'train/train{train_num:04}/train_model.pt')
    # torch.save(model.state_dict(), f'train/train{train_num:04}/train_model_params.pt')
    # writer.flush()
    # writer.close()


    model = models.video.mvit_v2_s()
    model.conv_proj = nn.Conv3d(4, 96, kernel_size=(3, 7, 7), stride=(2, 4, 4), padding=(1, 3, 3))
    num_features = model.head[1].in_features
    model.head[1] = torch.nn.Linear(num_features, 7)
    model.load_state_dict(torch.load(f'/mnt/2tb-hdd/Harsha/MER/train/train0012/model.pt'))

    model = model.to(device)
    model.eval()

    for i,data in enumerate(test_loader_micro):
    
        inputs, labels = data
        if(labels == 2):

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            print(outputs)
            break

    for i,data in enumerate(train_loader_micro):
        
        inputs, labels = data
        if(labels == 2):

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            print(outputs)
            break


        

    
if __name__ == '__main__':
    train_num = 13
    main(train_num=train_num)