import torch as th
import numpy as np
from PIL import Image
# pytorch=1.7.1
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
# pip install opencv-python
import cv2
import torchvision.transforms as transforms
import os
from numpy.random import randint

class RawFramesExtractorCV2():
    def __init__(self, centercrop=False, size=224, num_segments=-1, dense_sample=False, random_shift=False, strategy=1):
        self.centercrop = centercrop
        self.size = size
        self.num_segments = num_segments
        self.dense_sample = dense_sample
        self.random_shift = random_shift
        self.strategy = strategy
        self.transform = self._transform(self.size)
        self.image_tmpl ='img_{:05d}.jpg'
        self.image_tmpn ='image_{:06d}.jpg'

        if self.strategy == 1:
            print('[uniform sampling with 30 fps]')
        elif self.strategy == 2 : 
            print('[uniform sampling 30 fps with a random offset]')
        else: raise NotImplementedError


    def _transform(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])



    def _get_val_indices(self, video_list):
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + len(video_list) - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % len(video_list) for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:
            if len(video_list) > self.num_segments:
                tick = (len(video_list)) / float(self.num_segments)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1


    def _sample_indices(self, video_list):
        if not self.dense_sample: # TSN uniformly sampling
            average_duration = (len(video_list)) // self.num_segments

            offsets = []
            if average_duration > 0:
                offsets += list(np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,size=self.num_segments))
            elif len(video_list) > self.num_segments:
                offsets += list(np.sort(randint(len(video_list), size=self.num_segments)))
            else:
                offsets += list(np.zeros((self.num_segments,)))
            offsets = np.array(offsets)
            return offsets + 1
        else: # i3d type sampling for training
            sample_pos = max(1, 1 + len(video_list) - 64)
            t_stride = 64 // self.num_segments
            start_idx1 = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx1) % len(video_list) for idx in range(self.num_segments)]
            return np.array(offsets) + 1


    def video_to_tensor_fps30(self, frames_folder,  start_time=None, end_time=None):
        if start_time is not None or end_time is not None:
            assert isinstance(start_time, int) and isinstance(end_time, int) \
                   and start_time > -1 and end_time > start_time

        frames_list = os.listdir(frames_folder)
        frameCount = len(frames_list)

        # for didemo dataset, considering start sec and end sec
        total_duration = frameCount // 30   # 30 fps

        start_sec, end_sec = 0, total_duration
        if start_time is not None:
            start_sec, end_sec = start_time, end_time if end_time <= total_duration else total_duration


        # compared to reading videos (same sampling strategy)
        if self.strategy == 1:  # for val & test
            frame_indexs = np.arange(0, frameCount, 30) + 1
            if start_sec and end_sec:
                frame_indexs = [x for x in frame_indexs if x>=start_sec*30 and x<=end_sec*30]
            # frame_indexs = self._get_val_indices(frames_list)
            import pdb; pdb.set_trace()
        elif self.strategy == 2:    # for train
            offset = randint(0, 30 - 1)
            frame_indexs = np.arange(1, frameCount, 30) + offset
            if start_sec and end_sec:
                frame_indexs = [x for x in frame_indexs if x>=start_sec*30 and x<=end_sec*30]
            if frame_indexs[-1] >= frameCount:
                frame_indexs = frame_indexs[:-1]
        else:
            raise NotImplementedError
        import pdb; pdb.set_trace();
        # load images
        images = []
        for ind in frame_indexs:
            p = int(ind)
            segs_imgs = [self.transform(Image.open(os.path.join(frames_folder, self.image_tmpl.format(p))).convert("RGB"))]
            images.extend(segs_imgs)
            if p < len(frames_list):
                p += 1

        if len(images) > 0:
            video_data = th.tensor(np.stack(images))
        else:
            video_data = th.zeros(1)
        return {'video': video_data}



    def video_to_tensor_fps1(self, frames_folder,  start_time, end_time, max_frames):
        if start_time is not None or end_time is not None:
            assert isinstance(start_time, int) and isinstance(end_time, int) \
                   and start_time > -1 and end_time > start_time

        frames_list = os.listdir(frames_folder)
        frameCount = len(frames_list)

        # for didemo dataset, considering start sec and end sec
        total_duration = frameCount // 1  # fps1

        start_sec, end_sec = 0, total_duration
        if start_time is not None:
            start_sec, end_sec = start_time, end_time if end_time <= total_duration else total_duration


        # compared to reading videos (same sampling strategy)
        if self.strategy == 1:  # for val & test
            if start_sec is not None and end_sec is not None:
                start_sec = int(start_sec)
                end_sec = int(start_sec)
                frame_indexs = [x for x in range(frameCount) if x>=start_sec]
                if len(frame_indexs) >= max_frames:
                    frame_indexs = frame_indexs[: max_frames]
        else:
            raise NotImplementedError

        # load images
        images = []
        for ind in frame_indexs:
            p = int(ind) + 1
            # if os.path.exists(os.path.join(frames_folder, self.image_tmpl.format(p))):
            segs_imgs = [self.transform(Image.open(os.path.join(frames_folder, self.image_tmpn.format(p))).convert("RGB"))]
            images.extend(segs_imgs)

            if p < len(frames_list):
                p += 1

        if len(images) > 0:
            video_data = th.tensor(np.stack(images))
        else:
            video_data = th.zeros(1)
        return {'video': video_data}



    def video_to_tensor(self, frames_folder, preprocess,  start_time=None, end_time=None):
        if start_time is not None or end_time is not None:
            assert isinstance(start_time, int) and isinstance(end_time, int) \
                   and start_time > -1 and end_time > start_time

        frames_list = os.listdir(frames_folder)
        frameCount = len(frames_list)

        # compared to reading videos (same sampling strategy)
        if self.strategy == 1:
            frame_indexs = np.arange(0, frameCount, 30) + 1       # fixed with 30 fps without offset
        elif self.strategy == 2:
            offset = randint(0, 30 - 1)
            frame_indexs = np.arange(1, frameCount, 30) + offset
            if frame_indexs[-1] >= frameCount:
                frame_indexs = frame_indexs[:-1]
        else:
            raise NotImplementedError
        
        
        # load images
        images = []
        for ind in frame_indexs:
            p = int(ind)
            segs_imgs = [preprocess(Image.open(os.path.join(frames_folder, self.image_tmpl.format(p))).convert("RGB"))]
            images.extend(segs_imgs)
            if p < len(frames_list):
                p += 1

        if len(images) > 0:
            video_data = th.tensor(np.stack(images))
        else:
            video_data = th.zeros(1)
        return {'video': video_data}


    def get_video_data(self, video_path, start_time=None, end_time=None):
        image_input = self.video_to_tensor(video_path, self.transform, start_time=start_time, end_time=end_time)
        # import pdb; pdb.set_trace()
        return image_input

    def process_raw_data(self, raw_video_data):
        tensor_size = raw_video_data.size()
        tensor = raw_video_data.view(-1, 1, tensor_size[-3], tensor_size[-2], tensor_size[-1])
        return tensor

    def process_frame_order(self, raw_video_data, frame_order=0):
        # 0: ordinary order; 1: reverse order; 2: random order.
        if frame_order == 0:
            pass
        elif frame_order == 1:      # reverse frames
            reverse_order = np.arange(raw_video_data.size(0) - 1, -1, -1)
            raw_video_data = raw_video_data[reverse_order, ...]
        elif frame_order == 2:      # shuffle frames
            random_order = np.arange(raw_video_data.size(0))
            np.random.shuffle(random_order)
            raw_video_data = raw_video_data[random_order, ...]

        return raw_video_data

# An ordinary video frame extractor based CV2
RawFramesExtractor = RawFramesExtractorCV2