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
import cv2
import glob

from torchmetrics.classification import MulticlassF1Score, MulticlassRecall
from utils.face_alignment import face_aligner
from utils.face_detect_crop import face_cropper

torch.manual_seed(0)
cudnn.benchmark = True
plt.ion()  # interactive mode


def show_video(video, emotion):
    """Show video"""
    plt.figure()
    plt.title(emotion)
    plt.imshow(video)
    plt.show()


class CASME3Dataset(Dataset):
    def __init__(self, data_root, MAE_data, ME_data, expression_type, use_optical_flow_masks, transform=None, num_emotions=7, sequence_length=16):
        self.data_root = data_root
        self.data_dirpath = data_root / 'processed_data'
        self.optical_masks_dirpath = data_root / 'flow_masks'
        self.transform = transform
        self.sequence_length = sequence_length
        self.expression_type = expression_type
        # Load annotations
        self.annotations_macro = MAE_data
        self.annotations_micro = ME_data
        self.use_optical_flow_masks = use_optical_flow_masks
        # TODO:Figure out a way to merge both annotations
        if self.expression_type == 'macro':
            self.annotations = self.annotations_macro
        else:
            self.annotations = self.annotations_micro

        # Filter out sequences with less than 16 frames
        # self.filter_sequences()
        # Define a mapping from string labels to numerical values
        if(num_emotions == 7):
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
        subject = self.annotations.iloc[index]['subject']
        sequence = self.annotations.iloc[index]['filename']
        onset_frame = self.annotations.iloc[index]['onset']
        apex_frame = self.annotations.iloc[index]['apex']
        offset_frame = self.annotations.iloc[index]['offset']
        emotion = self.annotations.iloc[index]['emotion']

        emotion = self.label_mapping[emotion]

        print(f'{subject} {sequence} {onset_frame} {apex_frame} {offset_frame} {emotion}')

        datapoint_dirpath = self.data_dirpath / f'{subject}_{sequence}_{onset_frame}_{offset_frame}'
        rgb_dirpath = datapoint_dirpath / 'rgb'
        depth_dirpath = datapoint_dirpath / 'depth'

        rgb_frames = [frame.stem for frame in rgb_dirpath.glob('*.jpg')]
        depth_frames = [frame.stem for frame in depth_dirpath.glob('*.npy')]

        #Read the frames
        # rgb_video = np.array([np.array(Image.open(rgb_dirpath / f'{frame}.jpg')) for frame in rgb_frames])
        # depth_video = np.array([np.load(depth_dirpath / f'{frame}.npy') for frame in depth_frames])

        rgb_video = []
        depth_video = []
        masks_video = []

        for frame in rgb_frames:
            frame_num = int(frame.split('_')[-1])
            rgb_frame = np.array(Image.open(rgb_dirpath / f'{frame}.jpg'))
            depth_frame = np.load(depth_dirpath / f'{frame}.npy')
            onset_apex_optical_flow_mask = None
            apex_offset_optical_flow_mask = None
            if(self.use_optical_flow_masks):
                if(self.optical_masks_dirpath / f'{subject}_{sequence}_{onset_frame}_{offset_frame}/onset_apex.jpg').exists() == True:
                    onset_apex_optical_flow_mask = np.array(Image.open(self.optical_masks_dirpath / f'{subject}_{sequence}_{onset_frame}_{offset_frame}/onset_apex.jpg'))
                if(self.optical_masks_dirpath / f'{subject}_{sequence}_{onset_frame}_{offset_frame}/apex_offset.jpg').exists() == True:
                    apex_offset_optical_flow_mask = np.array(Image.open(self.optical_masks_dirpath / f'{subject}_{sequence}_{onset_frame}_{offset_frame}/apex_offset.jpg'))

                if(onset_apex_optical_flow_mask is None and apex_offset_optical_flow_mask is None):
                    mask = np.ones_(rgb_frame.shape[:2]) * 255
                elif(onset_apex_optical_flow_mask is None):
                    mask = apex_offset_optical_flow_mask
                elif(apex_offset_optical_flow_mask is None):
                    mask = onset_apex_optical_flow_mask
                else:
                    if(frame_num <= apex_frame):
                        mask = onset_apex_optical_flow_mask
                    else:
                        mask = apex_offset_optical_flow_mask
        
                # plt.imshow(rgb_frame)
                # plt.show()                                                                                                                                                                                             
            rgb_video.append(rgb_frame)
            depth_video.append(depth_frame)
            masks_video.append(mask)
        
        rgb_video = np.array(rgb_video)
        depth_video = np.expand_dims(np.array(depth_video), axis=-1)
        masks_video = np.expand_dims(np.array(masks_video), axis=-1)

        rgbd_video = np.concatenate((rgb_video, depth_video, masks_video), axis=-1)


        torch_rgbd_video = torch.from_numpy(rgbd_video).float()
        torch_rgbd_video = torch_rgbd_video.permute(0, 3, 1, 2)
        y_label = torch.tensor(emotion)


        if self.transform:
            torch_rgbd_video = self.transform(torch_rgbd_video)

        return torch_rgbd_video, y_label




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


def train_model(model, criterion, optimizer, train_loader, train_dataset, test_loader, test_dataset, writer, device,
                num_epochs=25):
    since = time.time()

    best_vloss = 100000000
    best_vacc = 0
    best_vf1 = 0
    best_vrecall = 0

    for epoch in tqdm.tqdm(range(num_epochs)):
        print(f'Epoch {epoch}/{num_epochs - 1}')

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

        if (avg_vloss < best_vloss):
            best_vloss = avg_vloss
            torch.save(model.state_dict(), f'train/train{train_num:04}/model.pt')

            metrics_dict = {
                'Cross Entropy Loss': round(avg_vloss, 4),
                'Accuracy': round(avg_vacc.item(), 4),
                'Unweighted F1 Score': round(unweighted_f1_score.item(), 4),
                'Unweighted Recall': round(unweighted_recall.item(), 4)
            }

            json.dump(metrics_dict, open(f'train/train{train_num:04}/metrics.json', 'w'))

    end = time.time()
    print(f'Training complete in {(end - since) / 60} minutes')
    # torch.save(model.state_dict(), f'train/train{train_num:04}/train_model_params.pt')
    return model


def common_subjects_macro_micro(annotations_macro, annotations_micro):
    macro_subjects = annotations_macro['subject'].unique()
    micro_subjects = annotations_micro['subject'].unique()

    common_subjects = [subject for subject in macro_subjects if subject in micro_subjects]

    return common_subjects


def train_test_subjects_split(common_subjects):
    train_subjects = random.sample(list(common_subjects), int(0.8 * len(common_subjects)))
    test_subjects = [subject for subject in common_subjects if subject not in train_subjects]

    return train_subjects, test_subjects


def train_test_macro_micro_split(annotations_macro, annotations_micro, train_subjects, test_subjects):
    train_data_macro = annotations_macro[annotations_macro['subject'].isin(train_subjects)]
    test_data_macro = annotations_macro[annotations_macro['subject'].isin(test_subjects)]

    train_data_micro = annotations_micro[annotations_micro['subject'].isin(train_subjects)]
    test_data_micro = annotations_micro[annotations_micro['subject'].isin(test_subjects)]

    return train_data_macro, test_data_macro, train_data_micro, test_data_micro


def main(train_num, epochs, num_emotions, batch_size, use_optical_flow_masks):
    writer = SummaryWriter(f'train/train{train_num:04}')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_root = Path('../../CASME3/')
    MAE_annotations_path = Path('../../CASME3/processed_annotations/me_annotations.csv')
    ME_annotations_path = Path('../../CASME3/processed_annotations/me_annotations.csv')

    annotations_macro = pd.read_csv(MAE_annotations_path)

    annotations_micro = pd.read_csv(ME_annotations_path)

    common_subjects = common_subjects_macro_micro(annotations_macro, annotations_micro)

    train_subjects, test_subjects = train_test_subjects_split(common_subjects)

    train_data_macro, test_data_macro, train_data_micro, test_data_micro = train_test_macro_micro_split(
        annotations_macro, annotations_micro, train_subjects, test_subjects)

    a = models.video.MViT_V2_S_Weights.KINETICS400_V1.transforms()
    a.mean.append(0.45)
    a.std.append(0.225)
    if(use_optical_flow_masks):
        a.mean.append(0.5)
        a.std.append(0.5)


    data_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        a
    ])


    # Initialize the micro dataset
    train_dataset_micro = CASME3Dataset(data_root, train_data_macro, train_data_micro, expression_type='micro', use_optical_flow_masks=use_optical_flow_masks, transform=data_transforms, num_emotions=num_emotions)
    test_dataset_micro = CASME3Dataset(data_root, test_data_macro, test_data_micro, expression_type='micro', use_optical_flow_masks=use_optical_flow_masks, transform=data_transforms, num_emotions=num_emotions)

    train_loader_micro = DataLoader(train_dataset_micro, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader_micro = DataLoader(test_dataset_micro, batch_size=batch_size, shuffle=False, num_workers=1)

    # Initialize the macro dataset
    train_dataset_macro = CASME3Dataset(data_root, annotations_macro, train_data_micro, expression_type='macro', use_optical_flow_masks=use_optical_flow_masks, transform=data_transforms, num_emotions=num_emotions)
    # train_dataset_macro = CASME3Dataset(data_root, train_data_macro, train_data_micro, expression_type='macro', transform=data_transforms)
    test_dataset_macro = CASME3Dataset(data_root, test_data_macro, test_data_micro, expression_type='macro', use_optical_flow_masks=use_optical_flow_masks, transform=data_transforms, num_emotions=num_emotions)

    train_loader_macro = DataLoader(train_dataset_macro, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader_macro = DataLoader(test_dataset_macro, batch_size=batch_size, shuffle=False, num_workers=1)

    # Load model
    model = models.video.mvit_v2_s(weights='DEFAULT')

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze all layers
    for param in model.parameters():
        param.requires_grad = True

    # Unfreeze last 3 blocks
    # for i in range(13,16):
    #     for param in model.blocks[i].parameters():
    #         param.requires_grad = True

    # Unfreeze classifier
    for param in model.head.parameters():
        param.requires_grad = True

    model.conv_proj = nn.Conv3d(4, 96, kernel_size=(3, 7, 7), stride=(2, 4, 4), padding=(1, 3, 3))
    if(use_optical_flow_masks):
        model.conv_proj = nn.Conv3d(5, 96, kernel_size=(3, 7, 7), stride=(2, 4, 4), padding=(1, 3, 3))

    num_features = model.head[1].in_features
    model.head[1] = torch.nn.Linear(num_features, 7)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    # Decay LR by a factor of 0.1 every 7 epochs
    # exp_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Finetuning on Macro
    # model = train_model(model, criterion, optimizer, exp_scheduler, train_loader_macro, train_dataset_macro, test_loader_macro, test_dataset_macro, writer, device, num_epochs=25)

    # model = train_model(model, criterion, optimizer, exp_scheduler, train_loader_micro, train_dataset_micro, test_loader_micro, test_dataset_micro, writer, device, num_epochs=25)
    model = train_model(model, criterion, optimizer, train_loader_micro, train_dataset_micro, test_loader_micro,
                        test_dataset_micro, writer, device, num_epochs=epochs)

    # torch.save(model, f'train/train{train_num:04}/train_model.pt')
    torch.save({
        
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
    }, f'train/train{train_num:04}/model.pt')
    writer.flush()
    writer.close()


if __name__ == '__main__':
    train_num = 9
    epochs = 100
    num_emotions = 7
    batch_size = 1
    use_optical_flow_masks = True
    main(train_num=train_num, epochs=epochs, num_emotions=num_emotions, batch_size=batch_size, use_optical_flow_masks=use_optical_flow_masks)