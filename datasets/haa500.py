"""
Version 0.1
Author: Raymond
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from vidaug import augmentors as va
from PIL import Image
import cv2
import pdb
import torchvision.transforms as transforms
import os


# Need to handle label
class HAA500Dataset(Dataset):
    def __init__(self, video_dir, splitfile, actionclassffile, randomAugmentationFactor, mode='totalframes',
                 mode_val=10):
        super().__init__()
        self.video_dir = video_dir
        self.mode = mode
        self.mode_val = mode_val
        self.index_list = self.readSplitFile(splitfile)
        self.actionClass_dict = self.readActionFile(actionclassffile)
        self.randomAugmentationFactor = randomAugmentationFactor

    def readSplitFile(self, splitfile):
        index_list = []
        f = open(splitfile, 'r')
        for line in f:
            index_list.append(line.strip())
        f.close()
        return index_list

    def readActionFile(self, actionclassffile):
        actionClass_dict = {}
        f = open(actionclassffile, 'r')
        for i, line in enumerate(f):
            actionClass_dict[line.strip()] = i
        f.close()
        return actionClass_dict

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        findex = self.index_list[idx]
        video_file_path = os.path.join(self.video_dir, findex)
        # Read video
        frames = []
        cap = cv2.VideoCapture(video_file_path)
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frames.append(frame)
            else:
                cap.release()
                break

        # Video augmentation
        chance = lambda aug: va.Sometimes(self.randomAugmentationFactor, aug)
        seq = va.Sequential([
            chance(va.RandomRotate(degrees=10)),
            chance(va.HorizontalFlip()),
        ])
        augmented_video = seq(frames)

        # Select frame from video
        selected_frames = []
        if self.mode == 'totalframes':
            total_num_of_frames = len(frames)
            sample_interval = total_num_of_frames // self.mode_val

            # Edge case: When the number of frames in the video does not contain enough number for the required frame
            num_of_actions = self.mode_val
            if (sample_interval < 1):
                sample_interval = 1
                num_of_actions = len(frames)

            for i in range(num_of_actions):
                selected_frames.append(augmented_video[i * sample_interval])
        else:
            sample_interval = self.mode_val
            for i in range(0, len(frames) - 1, sample_interval):
                selected_frames.append(augmented_video[i])

        # Resize and normalize
        data_transforms = transforms.Compose([
            transforms.Resize((720, 1280)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        transformed_selected_frames = []
        for f in selected_frames:
            transformed_selected_frames.append(data_transforms(f))

        # Output labels
        action_name = findex.split('/')[-2]
        labels = [self.actionClass_dict[action_name] for i in range(len(transformed_selected_frames))]

        # Convert and concat the frames
        output_image = None
        for f in transformed_selected_frames:
            one_frame = f.unsqueeze_(0)
            if output_image == None:
                output_image = one_frame
            else:
                output_image = torch.cat((output_image, one_frame), dim=0)

        return output_image, labels


def genTrainTestTxt(source):
    files = []
    dictionary = {}
    for root, dirnames, filenames in os.walk(source):
        for filename in filenames:
            if filename.endswith('mp4'):
                name = os.path.join(root, filename).split('/')
                action_name = name[-2]
                file_name = name[-1]
                # files.append(os.path.join(action_name, filename))
                if action_name in dictionary:
                    dictionary[action_name].append(filename)
                else:
                    dictionary[action_name] = []
                    dictionary[action_name].append(filename)

    trainval_f = open('trainval.txt', 'w')
    test_f = open('test.txt', 'w')
    class_f = open('actionClass.txt', 'w')
    trainval = 6.0  # 6/10
    test = 4.0  # 4/10
    import random
    for k in dictionary:
        videos = dictionary[k]
        random.shuffle(videos)
        total_length = len(videos)
        seperation_line = int(total_length * (trainval / (trainval + test)))
        trainval_videos = videos[:seperation_line]
        test_videos = videos[seperation_line:]

        for i in trainval_videos:
            trainval_f.write(k + '/' + i + '\n')

        for i in test_videos:
            test_f.write(k + '/' + i + '\n')

        class_f.write(k + '\n')
    class_f.close()
    trainval_f.close()
    test_f.close()
    return files


def build_hdd500_dataset(split, **kwargs):
    # TODO(Shuqin): Fix this function
    # specify video_dir, splitfile path,
    datapath = 'data/haa500/'
    video_dir = f'{datapath}/haa500_v1_1/video'
    splitfile = f'{datapath}/trainval.txt'
    actionclassffile = f'{datapath}/actionClass.txt'
    dataset = HAA500Dataset(video_dir, splitfile, actionclassffile, randomAugmentationFactor, mode = 'totalframes',
    mode_val = 10)

    return dataset


if __name__ == '__main__':
    video_dir = './haa500_v1_1/video'
    splitfile = './trainval.txt'
    actionclassffile = './actionClass.txt'
    mode = 'other'

    """
    #Unit test
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    params = {'batch_size': 1,
            'shuffle': True,
            'num_workers': 4}
    max_epochs = 100
    training_set = HAA500Dataset(video_dir, splitfile, actionclassffile, 0.2, mode='other', mode_val=5)
    training_generator = torch.utils.data.DataLoader(training_set, **params)
    # Loop over epochs
    for epoch in range(max_epochs):
        # Training
        for local_batch, local_labels in training_generator:
            print(len(local_batch))
            print(len(local_labels))
    """

    """
    #Check output
    training_set = HAA500Dataset(video_dir, splitfile, actionclassffile, 0.5, mode='other', mode_val=5)
    temp = training_set[2]
    pdb.set_trace()
    """
