from torch.utils.data import Dataset
from collections import defaultdict
import json
import cv2


class FineGym(Dataset):
    def __init__(self, path, split, num_frames_per_video, frame_size):
        super().__init__()
        assert(split in ['train', 'val'])
        categories = {}
        for line in open('%s/gym288_categories.txt' % path):
            line = line.replace(';', '').split()
            categories[line[1]] = (int(line[3]), int(line[5]))
        name_to_event = json.load(
            open('%s/finegym_annotation_info_v1.1.json' % path))
        self.events = defaultdict(lambda: {'actions': []})
        for line in open(path+'/gym288_%s_element%s.txt' % (split, '_v1.1'if split == 'train'else None)):
            k, v = line.split()
            k = k.split('_')
            v_id = k[0]
            e_id = 'E_'+k[2]+'_'+k[3]
            events[e_id]['video'] = path+'/'+v_id+'/'+e_id+'.mp4'
            events[e_id]['actions'].append(
                (int(k[5]), int(k[6])+categories[v]))
            events[e_id]['event'] = name_to_event[v_id][e_id]['event']
        self.event_names = list(sorted(events.keys()))
        self.num_frames_per_video = num_frames_per_video
        self.frame_size = frame_size

    def __getitem__(self, index):
        event_name = self.event_names[index]
        event = self.events[event_name]
        useful_time = 0
        for a in event_name['actions']:
            useful_time += max(a[1]-a[0], 1)
        frame_rate = useful_time/self.num_frames_per_video
        video = cv2.VideoCapture(event['video'])
        # for frame sampling, learn from this: https://github.com/YuxinZhaozyx/pytorch-VideoDataset/blob/master/transforms.py
