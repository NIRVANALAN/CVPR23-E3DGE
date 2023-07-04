'''
script to produce gallary video in the E3DGE demo video. 

To use it in your code, simply modify 

    gt_root: the image ground truth.
    video_trajectory_root: texture video inversion output path.
    geo_video_trajectory_root: geometry video inversion output path.
    video_output_path: the final gallary output path.
'''


import imageio
import traceback
from PIL import Image
from pathlib import Path
import cv2
import skvideo.io
import numpy as np
from ipdb import set_trace as st
import random

#read videos

tex_videos = []
geo_videos = []

# random.shuffle(tex_videos)
# random.shuffle(geo_videos) # shall be paired

# properties
vid_size = (1024, 1024)
speed = 5
video_length = 18  # seconds # 
input_video_frames = 250 // speed
fps = 60
res_h, res_w = 1080, 1920
candidate_rows = 13  # vertical videos concatenated.
right_res = 90  # 12 videos, minimum size
mini_ratio = right_res / res_h

total_frames = fps * video_length  #

gt_root = Path('datasets/test_img')

toonify=False # whether this is toonified video. Different save format, default=False
# toonify = True

if toonify:
    # ! Toonify settings
    video_trajectory_root = Path(
        'logs/toonify/no_local_branch_N48/Toonify400_1024x1024/val/videos/')
    geo_video_trajectory_root = video_trajectory_root
    video_output_path = 'logs/final_organized/demo-video/gallary_toonify.mp4'

    toonify_ids = [322, 360, 445, 540, 591, 597, 327, 438, 524,
                   532]  # * for debug
    random_ids = list(range(300))
    random.shuffle(random_ids)
    video_ids_for_iter = toonify_ids + toonify_ids + random_ids
    video_ids = iter(video_ids_for_iter)
else:
    # ! celeba hq settings
    video_trajectory_root = Path(
        'logs/final_organized/demo-video/output_video_V9_fixbug/ffhq1024x1024/val/videos/'
    )
    geo_video_trajectory_root = Path(
        'logs/final_organized/demo-video/output_video/ffhq1024x1024/val/videos/'
    )
    video_output_path = 'logs/final_organized/demo-video/gallary_real.mp4'

    csv_file_path = Path('logs/final_organized/demo-video/200.csv')  # 600 ids
    with open(csv_file_path, 'r') as f:
        content = f.readlines()[:]
    video_ids = [content[i].split(',')[0] for i in range(len(content))]
    random.shuffle(video_ids)
    pre_defined_ids = '529 838 647 485 726 800'.split(
    ) + '220 463 568 706 366 184'.split()
    pre_defined_ids = pre_defined_ids + pre_defined_ids + pre_defined_ids
    # random.shuffle(pre_defined_ids)
    video_ids_for_iter = pre_defined_ids + video_ids
    video_ids_for_iter.remove('560')
    video_ids_for_iter.remove('1')
    video_ids = iter(video_ids_for_iter)


def create_writer(video_output_path, crf=18):
    writer = skvideo.io.FFmpegWriter(
        video_output_path,
        outputdict={
            '-pix_fmt': 'yuv420p',
            '-crf': str(crf)  # 11 
        })
    return writer


def mix_tex_geo_vid(tex_vid, geo_vid):
    _, h, w, _ = tex_vid.shape
    _, geo_h, geo_w, _ = geo_vid.shape
    # vid_w = tex_vid.shape[-2]
    # st()
    if geo_h != h:
        # resize geo videos
        geo_vid = np.stack(
            [cv2.resize(geo_vid[i], (w, h)) for i in range(geo_vid.shape[0])])
    tex_vid[:, :, -w // 2:, :] = geo_vid[:, :, w // 2:, :]
    return tex_vid  # ! debug


def read_video(path, size, speed=1):
    video = np.stack([
        cv2.resize(frame[:, -frame.shape[0]:, :], size)
        for i, frame in enumerate(imageio.get_reader(path, 'ffmpeg'))
        if i % speed == 0
    ], 0)

    return video


def sample_videos(concat_h=True):

    def sample_single_vid(return_stem=False):
        # img_stem = video_ids[random.randint(0, len(video_ids) - 1)]
        # st()
        global video_ids, video_ids_for_iter
        try:
            img_stem = next(video_ids)
        except StopIteration as e:
            video_ids = iter(video_ids_for_iter)
            img_stem = next(video_ids)

        try:
            tex_vid = read_video(
                str(video_trajectory_root / f'{img_stem}.jpg' /
                    'sample_video__elipsoid.mp4'), vid_size, speed)
            geo_vid = read_video(
                str(geo_video_trajectory_root / f'{img_stem}.jpg' /
                    'sample_depth_video__elipsoid.mp4'), vid_size, speed)
            video = mix_tex_geo_vid(tex_vid, geo_vid)
            if return_stem:
                return video, img_stem
            else:
                return video
        except Exception as e:
            traceback.print_exc()
            print(img_stem)
            return sample_single_vid(return_stem)
            # return None
            # return sample_single_vid()

    def sample_quater_video():
        # return 2*2 video grid to show gt image
        def concat_img_video():
            video, img_stem = sample_single_vid(True)
            print('stem: {}'.format(img_stem))
            gt_img = np.array(Image.open(gt_root / f'{img_stem}.jpg'))
            gt_img = cv2.resize(gt_img, (video.shape[2], video.shape[1]))
            video = np.stack(
                [np.concatenate([gt_img, frame], 1) for frame in video], 0)
            return video

        return np.concatenate([concat_img_video() for _ in range(2)], 1)

    # sample_video_fn = sample_single_vid
    sample_video_fn = sample_quater_video

    # videos = [sample_single_vid() for _ in range(candidate_rows)]
    videos = [sample_video_fn() for _ in range(candidate_rows)]
    if concat_h:
        videos = np.concatenate(videos, 1)  # F H*15 W C
    return videos


def get_current_res(anchor_u_l):
    """
    anchor_w: w (u) coordinate of the left most largest frame, left u.
    """
    if anchor_u_l <= 0:
        # return (res_h, res_h + anchor_u_l)  # clip
        # return (res_h, res_h + anchor_u_l)  # clip
        return (res_h, res_h)  # clip
    else:
        current_res = res_h * (mini_ratio + (res_w - anchor_u_l) / res_w *
                               (1 - mini_ratio))  # right_res -> res_h
        current_res = int(current_res)
        return (current_res, current_res)
    # current_res = int(min(current_res, res_h)) # smaller than res_h


def composite_videos():
    # loop_num = int(500 / 5 * candidate_rows)
    frames = []

    videos_stack = [sample_videos() for _ in range(5)]  # init stack

    while (len(frames) < total_frames):

        # anchor_w = res_h # right frame position of the left most biggest frame
        for anchor_w in range(0, res_h, 8):  # 4 * 250 \approx 1080
            anchor_w *= -1  # left most staring position
            frame = np.zeros((res_h, res_w, 3)).astype(np.uint8)
            current_anchor_w = anchor_w

            # fill in a frame
            for video_idx in range(5):
                current_res_h_single, current_res_w = get_current_res(
                    current_anchor_w)  # resolution of single frame
                current_res_h = current_res_h_single * candidate_rows  # * shall be int
                video_frame = videos_stack[video_idx][
                    len(frames) % input_video_frames]  # loop frames

                video_frame = cv2.resize(
                    video_frame,
                    (current_res_w, current_res_h))  # (w, h) order

                # * clip w (left most)
                if current_anchor_w <= 0:  # left most clip w
                    video_frame = video_frame[:, -(res_h + current_anchor_w):,
                                              ...]
                else:
                    # * clip w, right most
                    video_frame = video_frame[:, 0:min(
                        res_w - current_anchor_w, current_res_h_single
                    )]  # min of available right margin and h_single

                # * clip h, split from the middle. total size = res_h
                lower_clip_idx = int(current_res_h / 2 - res_h / 2)
                upper_clip_idx = lower_clip_idx + res_h
                video_frame = video_frame[lower_clip_idx:upper_clip_idx, ...]

                try:
                    # frame[lower_clip_idx:upper_clip_idx, max(0, current_anchor_w):min(res_w, current_anchor_w+current_res_h_single)] = video_frame
                    frame[:,
                          max(0, current_anchor_w
                              ):min(res_w, current_anchor_w +
                                    current_res_h_single)] = video_frame
                except Exception as exception:
                    traceback.print_exc()
                    st()

                current_anchor_w += current_res_h_single
                if (current_anchor_w >= res_w):
                    break

                # print('current_anchor_w: {}'.format(current_anchor_w))
            frames.append(frame)
            print('frames: {}'.format(len(frames)))
            if (len(frames) > total_frames):
                break
                # frame_idx = (frame_idx+1) / input_video_frames

        videos_stack.pop(0)  # remove head video
        videos_stack.append(sample_videos())

    writer = create_writer(video_output_path, 18)  # 1024 output
    for frame in frames:
        writer.writeFrame(frame)
    writer.close()

    print('output video to :{}'.format(video_output_path))


composite_videos()
