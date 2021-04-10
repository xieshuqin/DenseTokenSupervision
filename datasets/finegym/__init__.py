from torch.utils.data import Dataset
from collections import defaultdict
import json
import cv2
import torch
from IPython import embed
import os
from PIL import Image
import numpy as np
import random
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class FineGym(Dataset):
    def __init__(self, path='data', split='train', num_frames_per_video=4, frame_size=224):
        super().__init__()
        assert(split in ['train', 'val', 'test'])
        categories = {}
        for line in open('%s/gym288_categories.txt' % path):
            line = line.replace(';', '').split()
            categories[line[1]] = (int(line[3]), int(line[1]))
        name_to_event = json.load(
            open('%s/finegym_annotation_info_v1.1.json' % path))
        events = defaultdict(lambda: {'actions': []})
        for line in open(path+'/gym288_%s_element%s.txt' % (split, '_v1.1'if split == 'train'else '')):
            k, v = line.split()
            k = k.split('_')
            v_id = k[0]
            e_id = v_id+'_E_'+k[2]+'_'+k[3]
            events[e_id]['video'] = path+'/videos/' + \
                v_id+'/'+'E_'+k[2]+'_'+k[3]+'.mp4'
            events[e_id]['actions'].append(
                (int(k[5]), int(k[6])+(k[5] == k[6]))+categories[v])
            events[e_id]['event'] = name_to_event[v_id]['E_' +
                                                        k[2]+'_'+k[3]]['event']
        self.events = events
        self.event_names = list(sorted(events.keys()))
        self.event_names = [
            n for n in self.event_names if os.path.exists(events[n]['video']) and '1JsRXIoR3C0' not in n]
        self.num_frames_per_video = num_frames_per_video
        self.frame_size = frame_size

    def __len__(self):
        return len(self.event_names)

    def __getitem__(self, index):
        event_name = self.event_names[index]
        event = self.events[event_name]
        actions = event['actions']
        action_hit = [0 for i in range(len(actions))]
        for i in range(self.num_frames_per_video):
            action_hit[i % len(action_hit)] += 1
        cap = cv2.VideoCapture(event['video'])
        assert(cap.isOpened())
        good_frames = []
        good_labels = []
        good_label = torch.tensor(event['event'])
        fps = cap.get(cv2.CAP_PROP_FPS)
        mx = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        for i in range(len(actions)):
            a = actions[i]
            for j in range(action_hit[i]):
                true_time = (j+1)/(action_hit[i]+1)*(a[1]-a[0])+a[0]
                frame_index = min(fps*true_time, mx)
                label = torch.tensor(a[3])
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()
                assert(ret)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.frame_size, self.frame_size))
                frame = torch.from_numpy(frame).float()
                frame /= 255
                frame -= torch.tensor(IMAGENET_DEFAULT_MEAN)
                frame /= torch.tensor(IMAGENET_DEFAULT_STD)
                frame = frame.permute(2, 0, 1)
                good_frames.append(frame)
                good_labels.append(label)
        cap.release()
        assert(len(good_frames) == self.num_frames_per_video)
        return torch.stack(good_frames), torch.stack(good_labels), good_label-1


def build_finegym_dataset(split, num_frames_per_video=2, frame_size=224):
    path = '/home/shuqin/hdd/datasets/finegym/data'
    return FineGym(path=path, split=split, num_frames_per_video=num_frames_per_video, frame_size=frame_size)


if __name__ == '__main__':
    # dataset = FineGym(path='data', split='train', num_frames_per_video=2, frame_size=224)
    dataset = build_finegym_dataset(split='train', num_frames_per_video=2, frame_size=224)
    all_labels = set()
    all_frame_labels = set()
    all_has_multiple = []
    random.shuffle(dataset.event_names)
    print(len(dataset))
    max_frame_label, min_frame_label = -1, 290
    max_video_label, min_video_label = -1, 290
    from tqdm import tqdm
    for i in tqdm(range(len(dataset))):
        frames, frame_labels, label = dataset[i]
        max_frame_label = max(frame_labels.max(), max_frame_label)
        min_frame_label = min(frame_labels.min(), min_frame_label)
        max_video_label = max(label.max(), max_video_label)
        min_video_label = min(label.min(), min_video_label)

        print(f'frame label max {max_frame_label} min {min_frame_label}')
        print(f'video label max {max_video_label}, min {min_video_label}')

        # all_labels.add(label.item())
        # frame_labels = {l.item()for l in frame_labels}
        # for l in frame_labels:
        #     all_frame_labels.add(l)
        # all_has_multiple.append(len(frame_labels) > 1)
        # if all_has_multiple[-1] and False:
        #     for j in range(4):
        #         Image.fromarray((frames[j].numpy().transpose(
        #             1, 2, 0)*255).astype(np.uint8)).show()
        #     print(frame_labels)
        #     exit(0)
        # print('num_videos', len(all_has_multiple))
        # print('num_video_classes', len(all_labels))
        # print('num_videos_has_more_than_one_subactions', sum(all_has_multiple))
        # print('num_subactions', len(all_frame_labels))
