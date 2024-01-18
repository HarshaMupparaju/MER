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
            potential_frame_path = self.data_root / subject / sequence / 'color' / f'{potential_frame}.jpg'
            if (potential_frame not in video_frames) and (potential_frame != apex_frame) and potential_frame_path.is_file():
                video_frames.append(potential_frame)
        


        # #Picking 16 frames around apex frame
        # video_frames = random.sample(range(onset_frame+1, offset_frame), 13)


                

        #Adding onset, apex and offset frames
        video_frames.append(onset_frame)
        video_frames.append(apex_frame)
        video_frames.append(offset_frame)
        video_frames.sort()



        # depth_video_frames = np.array([np.array(Image.open(self.data_root / subject / sequence / 'depth' / f'{frame}.png')) for frame in video_frames])

        # depth_video_frames = depth_video_frames.astype(np.uint8)
        # depth_video_frames = depth_video_frames.reshape(depth_video_frames.shape[0], depth_video_frames.shape[1], depth_video_frames.shape[2], 1)


        




        numpy_video_frames = np.array([np.array(Image.open(self.data_root / subject / sequence / 'color' / f'{frame}.jpg')) for frame in video_frames])
        


        # numpy_depth_color_frames = np.concatenate((numpy_video_frames, depth_video_frames), axis=-1)

        # numpy_video_frames = numpy_depth_color_frames




        
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

            # self.annotations.drop(self.annotations[(self.annotations['Subject'] == 'spNO.171') & (self.annotations[self.annotations['Filename'] == 'e'])].index, axis=0, inplace=True)
        
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



def train_model(model, criterion, optimizer, scheduler,train_loader, train_dataset, test_loader, test_dataset, writer, device, num_epochs=25):
    since = time.time()

    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0
        best_loss = 100000000
        best_Uf1 = 0.0
        best_UArecall = 0.0
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)
            
            # Each epoch has a training and validation phase
            for phase in ['train', 'test']:
                if phase == 'train':
                    model.train()
                    dataloader = train_loader
                    dataset = train_dataset
                else:
                    model.eval()
                    dataloader = test_loader
                    dataset = test_dataset

                running_loss = 0.0
                running_corrects = 0
                targets = []
                predictions = []
                for inputs, labels in dataloader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        
                        outputs = model(inputs)

                        _, preds = torch.max(outputs, 1)
                        predictions.append(preds)
                        targets.append(labels.data)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()



                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                predictions = torch.cat(predictions)
                targets = torch.cat(targets)

                f1 = MulticlassF1Score(num_classes=7, average='macro').to(device)
                Recall = MulticlassRecall(num_classes=7, average='macro').to(device)
                unweighted_f1_score = f1(predictions, targets)
                unweighted_average_recall = Recall(predictions, targets)
                # unweighted_f1_score = f1_score(predictions, targets, num_classes=8, average='macro')
                # print(unweighted_f1_score)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / len(dataset)
                if phase == 'train':
                    writer.add_scalar("Loss/train", epoch_loss, epoch)
                else:
                    writer.add_scalar("Loss/test", epoch_loss, epoch)
                epoch_acc = running_corrects.double() / len(dataset)
                epoch_f1_score = unweighted_f1_score
                epoch_recall = unweighted_average_recall
                
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} UF1: {epoch_f1_score:.4f} UAR: {epoch_recall:.4f}')

                if(phase == 'test'):
                    if(epoch_loss < best_loss):
                        best_loss = epoch_loss
                    if(epoch_acc > best_acc):
                        best_acc = epoch_acc
                        torch.save(model.state_dict(), best_model_params_path)
                    if(epoch_f1_score > best_Uf1):
                        best_Uf1 = epoch_f1_score
                    if(epoch_recall > best_UArecall):
                        best_UArecall = epoch_recall
                    

                # deep copy the model
                # if phase == 'test' and epoch_acc > best_acc:
                #     torch.save(model.state_dict(), best_model_params_path)
                #     best_acc = epoch_acc

                # print()
        
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        # print(f'Best val Acc: {best_acc:4f}')

        metrics_dictionary = {
            'Cross Entropy Loss': round(best_loss, 4),
            'Accuracy': round(best_acc.item(), 4),
            'Unweighted F1 Score': round(best_Uf1.item(), 4),
            'Unweighted Average Recall': round(best_UArecall.item(), 4)
        }

        # with open(f'train/train{train_num:04}/metrics.json', 'w') as fp:
        #     json.dump(metrics_dictionary, fp)
        json.dump(metrics_dictionary, open(f'train/train{train_num:04}/metrics.json', 'w'), indent=4)


        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model


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

    # a = models.video.MViT_V2_S_Weights.KINETICS400_V1.transforms()
    # a.mean.append(0.45)
    # a.std.append(0.225)
    # data_transforms = transforms.Compose([a])
    data_transforms = transforms.Compose([
        models.video.MViT_V2_S_Weights.KINETICS400_V1.transforms()
    ])



    #Initialize the micro dataset
    train_dataset_micro = CASME3Dataset(data_root, train_data_macro, train_data_micro, expression_type='micro', transform=data_transforms)
    test_dataset_micro = CASME3Dataset(data_root, test_data_macro, test_data_micro, expression_type='micro', transform=data_transforms)

    train_loader_micro = DataLoader(train_dataset_micro, batch_size=8  , shuffle=True, num_workers=8)
    test_loader_micro = DataLoader(test_dataset_micro, batch_size=8  , shuffle=False, num_workers=8)

    #Initialize the macro dataset
    train_dataset_macro = CASME3Dataset(data_root, annotations_macro, train_data_micro, expression_type='macro', transform=data_transforms)
    # train_dataset_macro = CASME3Dataset(data_root, train_data_macro, train_data_micro, expression_type='macro', transform=data_transforms)
    test_dataset_macro = CASME3Dataset(data_root, test_data_macro, test_data_micro, expression_type='macro', transform=data_transforms)

    train_loader_macro = DataLoader(train_dataset_macro, batch_size=8  , shuffle=True, num_workers=8)
    test_loader_macro = DataLoader(test_dataset_macro, batch_size=8  , shuffle=False, num_workers=8)

    #Load model
    model = models.video.mvit_v2_s(weights='DEFAULT')


    #Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    #Unfreeze all layers
    for param in model.parameters():
        param.requires_grad =True

    # Unfreeze last 3 blocks
    # for i in range(13,16):
    #     for param in model.blocks[i].parameters():
    #         param.requires_grad = True

    #Unfreeze classifier
    # for param in model.head.parameters():
    #     param.requires_grad = True

    num_features = model.head[1].in_features
    model.head[1] = torch.nn.Linear(num_features, 7)

    model = model.to(device)
    # print(model)
    # print(1/0)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.001)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    #Finetuning on Macro
    model = train_model(model, criterion, optimizer, exp_scheduler, train_loader_macro, train_dataset_macro, test_loader_macro, test_dataset_macro, writer, device, num_epochs=25)


    model = train_model(model, criterion, optimizer, exp_scheduler, train_loader_micro, train_dataset_micro, test_loader_micro, test_dataset_micro, writer, device, num_epochs=25)

    
    # torch.save(model, f'train/train{train_num:04}/train_model.pt')
    torch.save(model.state_dict(), f'train/train{train_num:04}/train_model_params.pt')
    writer.flush()
    writer.close()

    
if __name__ == '__main__':
    train_num = 4
    main(train_num=train_num)