import os 
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import time
from tempfile import TemporaryDirectory
import random
import time

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
    def __init__(self, data_root, MAE_annotations_path, ME_annotations_path, transform=None, sequence_length=16):
        self.data_root = data_root
        self.transform = transform
        self.sequence_length = sequence_length

        #Load annotations
        self.annotations_macro = pd.read_excel(MAE_annotations_path, sheet_name=None)['Sheet1']
        self.annotations_micro = pd.read_excel(ME_annotations_path, sheet_name='label')
        #TODO:Figure out a way to merge both annotations
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
            'Others': 7
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


        print(f'{subject} {sequence} {onset_frame} {apex_frame} {offset_frame} {emotion}')

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





        numpy_video_frames = np.array([np.array(Image.open(self.data_root / subject / sequence / 'color' / f'{frame}.jpg')) for frame in video_frames])
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
data_root = Path('/media/harshamupparaju/Expansion/Harsha/Databases/CASME3/part_A/data/Compressed_version1_seperate_compress')
MAE_annotations_path = Path('/media/harshamupparaju/Expansion/Harsha/Databases/CASME3/part_A/annotation/cas(me)3_part_A_MaE_label_JpgIndex_v2_emotion.xlsx')
ME_annotations_path = Path('/media/harshamupparaju/Expansion/Harsha/Databases/CASME3/part_A/annotation/cas(me)3_part_A_ME_label_JpgIndex_v2.xlsx')

data_transforms = transforms.Compose([
    models.video.MViT_V2_S_Weights.KINETICS400_V1.transforms()
])

#Initialize the dataset
facial_expression_dataset = CASME3Dataset(data_root, MAE_annotations_path, ME_annotations_path, transform=data_transforms)

print(len(facial_expression_dataset))
print(1/0)

train_size = int(0.8 * len(facial_expression_dataset))
test_size = len(facial_expression_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(facial_expression_dataset, [train_size, test_size])

dataloader = DataLoader(facial_expression_dataset, batch_size=2  , shuffle=False, num_workers=0)

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)
            
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

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

                f1 = MulticlassF1Score(num_classes=8, average='macro').to(device)
                Recall = MulticlassRecall(num_classes=8, average='macro').to(device)
                unweighted_f1_score = f1(predictions, targets)
                unweighted_average_recall = Recall(predictions, targets)
                # unweighted_f1_score = f1_score(predictions, targets, num_classes=8, average='macro')
                print(unweighted_f1_score)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / len(facial_expression_dataset)
                epoch_acc = running_corrects.double() / len(facial_expression_dataset)
                epoch_f1_score = unweighted_f1_score
                epoch_recall = unweighted_average_recall
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} UF1: {epoch_f1_score:.4f} UAR: {epoch_recall:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    torch.save(model.state_dict(), best_model_params_path)
                    best_acc = epoch_acc

            print()
        
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model

#Load model
model = models.video.mvit_v2_s(weights='DEFAULT')

for param in model.parameters():
    param.requires_grad = False

for param in model.head.parameters():
    param.requires_grad = True

num_features = model.head[1].in_features
model.head[1] = torch.nn.Linear(num_features, 8)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer = optim.SGD(model.head.parameters(), lr=1e-3, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model = train_model(model, criterion, optimizer, exp_scheduler, num_epochs=1)



    
