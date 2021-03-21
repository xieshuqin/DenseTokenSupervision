from torch.utils.data import Dataset
from collections import defaultdict
import json
import cv2
import torch
from IPython import embed
import os
from PIL import Image
import numpy as np


class FineGym(Dataset):
    def __init__(self, path, split, num_frames_per_video, frame_size):
        super().__init__()
        assert(split in ['train', 'val'])
        categories = {}
        for line in open('%s/gym288_categories.txt' % path):
            line = line.replace(';', '').split()
            categories[line[1]] = (int(line[3]), int(line[1]))
        name_to_event = json.load(
            open('%s/finegym_annotation_info_v1.1.json' % path))
        events = defaultdict(lambda: {'actions': []})
        for line in open(path+'/gym288_%s_element%s.txt' % (split, '_v1.1'if split == 'train'else None)):
            k, v = line.split()
            k = k.split('_')
            v_id = k[0]
            e_id = 'E_'+k[2]+'_'+k[3]
            events[e_id]['video'] = path+'/videos/'+v_id+'/'+e_id+'.mp4'
            events[e_id]['actions'].append(
                (int(k[5]), int(k[6])+(k[5] == k[6]))+categories[v])
            events[e_id]['event'] = name_to_event[v_id][e_id]['event']
        self.events = events
        self.event_names = list(sorted(events.keys()))
        self.event_names = [
            n for n in self.event_names if os.path.exists(events[n]['video'])]
        self.num_frames_per_video = num_frames_per_video
        self.frame_size = frame_size

    def __len__(self):
        return len(self.event_names)

    def __getitem__(self, index):
        event_name = self.event_names[index]
        event = self.events[event_name]
        useful_time = 0
        actions = event['actions']
        for a in actions:
            useful_time += a[1]-a[0]
        self.fps = self.num_frames_per_video/useful_time
        cap = cv2.VideoCapture(event['video'])
        assert(cap.isOpened())
        sample_factor = 1
        old_fps = cap.get(cv2.CAP_PROP_FPS)  # fps of video
        sample_factor = int(old_fps / self.fps)
        assert(sample_factor >= 1)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        time_len = int(num_frames / sample_factor)
        good_frames = []
        good_labels = []
        good_label = torch.tensor(event['event'])
        for index in range(time_len):
            frame_index = sample_factor * index
            true_time = frame_index/old_fps
            label = None
            for a in actions:
                if a[0] <= true_time <= a[1]:
                    label = a[3]
            if not label:
                continue
            label = torch.tensor(label)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            assert(ret)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.frame_size, self.frame_size))
            frame = torch.from_numpy(frame).float()
            frame = frame.permute(2, 0, 1)/255
            good_frames.append(frame)
            good_labels.append(label)
        cap.release()
        while len(good_frames) < self.num_frames_per_video:
            good_frames.append(good_frames[-1])
            good_labels.append(good_labels[-1])
        return torch.stack(good_frames[:self.num_frames_per_video]), torch.stack(good_labels[:self.num_frames_per_video]), good_label


if __name__ == '__main__':
    dataset = FineGym('data', 'train', 4, 224)
    all_labels = set()
    all_frame_labels = set()
    all_has_multiple = []
    for i in range(len(dataset)):
        frames, frame_labels, label = dataset[i]
        for j in []:
            Image.fromarray((frames[j].numpy().transpose(
                1, 2, 0)*255).astype(np.uint8)).show()
        all_labels.add(label.item())
        frame_labels = {l.item()for l in frame_labels}
        for l in frame_labels:
            all_frame_labels.add(l)
        all_has_multiple.append(len(frame_labels) > 1)
        print('num_videos', len(all_has_multiple))
        print('num_video_classes', len(all_labels))
        print('num_videos_has_more_than_one_subactions', sum(all_has_multiple))
        print('num_subactions', len(all_frame_labels))
