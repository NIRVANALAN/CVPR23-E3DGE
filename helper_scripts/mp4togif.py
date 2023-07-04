import cv2
import skvideo.io
import imageio
from tqdm import tqdm
from pathlib import Path

from multiprocessing import Pool

# read ranked csv
# csv_file_path = Path('logs/2022-3dface-experiments - trajectory-identity-similarity.csv')
csv_file_path = Path('logs/final_organized/demo-video/200.csv')
# csv_file_path = Path('logs/final_organized/demo-video/400.csv')
# csv_file_path = Path('logs/final_organized/demo-video/600.csv')

with open(csv_file_path, 'r') as f:
    content = f.readlines()

identity = ['322'] + [content[i].split(',')[0] for i in range(len(content))]


def exportMp4Gif(identity_list):
    # get GT
    # video_trajectory_root = Path('logs/final_organized/ablations/N_samples24/occlusion_runner_cycle_v9_hybridAlign_fullTestSet/ffhq1024x1024/val/videos/')
    # video_trajectory_root = Path('logs/final_organized/demo-video/output_video_V9_fixbug/ffhq1024x1024/val/videos/')
    video_trajectory_root = Path(
        'logs/final_organized/demo-video/output_video/ffhq1024x1024/val/videos/'
    )

    # metrics_output_path = video_trajectory_root / 'trajectory_scores.csv'
    # test_dataset = ImagesDatasetEval(test_dataset_path,
    #                              transform=transform)  # size 2824
    # test_loader = torch.utils.data.DataLoader(test_dataset,
    #                               batch_size=1,
    #                               drop_last=False,
    #                               num_workers=1)

    def MP4toGif(mp4_path, gif_path):
        # print(video_path)
        video = skvideo.io.vread(str(video_path))

        with imageio.get_writer(gif_path, mode='I', duration=1 / 60) as writer:
            for frame_idx in range(0, video.shape[0], 3):
                frame = cv2.resize(video[frame_idx], (256, 256))
                writer.append_data(frame)
        print(gif_path)

    for idx, img_stem in enumerate(tqdm(identity_list)):
        video_path = video_trajectory_root / f'{img_stem}.jpg' / 'sample_video__elipsoid.mp4'
        geo_video_path = video_trajectory_root / f'{img_stem}.jpg' / 'sample_depth_video__elipsoid.mp4'

        for v_path in (video_path, geo_video_path):
            gif_path = Path(v_path).parent / Path(v_path).with_suffix(
                '.gif').name
            MP4toGif(v_path, gif_path)

        gif_path = video_trajectory_root / f'{img_stem}.jpg' / 'video.gif'

        # torch.cuda.empty_cache()


# def MP4toGif(mp4_path, gif_path):
def MP4toGif(paths):
    mp4_path, gif_path = paths
    try:
        video = skvideo.io.vread(str(mp4_path))
    except:
        print(mp4_path, 'failed')
        return

    with imageio.get_writer(gif_path, mode='I', duration=1 / 60) as writer:
        for frame_idx in range(0, video.shape[0], 3):

            if video[frame_idx].shape[0] != video[frame_idx].shape[
                    1]:  # maybe include other intermedate results here, crop the right most one
                # video[frame_idx] = video[frame_idx][:, -video[frame_idx].shape[0]:, :]
                frame = cv2.resize(
                    video[frame_idx][:, -video[frame_idx].shape[0]:, :],
                    (256, 256))
            else:
                frame = cv2.resize(video[frame_idx], (256, 256))
            writer.append_data(frame)
    print(gif_path)


def exportMp4GifMP(identity_list):
    # get GT
    # video_trajectory_root = Path('logs/final_organized/ablations/N_samples24/occlusion_runner_cycle_v9_hybridAlign_fullTestSet/ffhq1024x1024/val/videos/')
    # video_trajectory_root = Path('logs/final_organized/demo-video/output_video_V9_fixbug/ffhq1024x1024/val/videos/')
    video_trajectory_root = Path(
        'logs/final_organized/demo-video/output_video/ffhq1024x1024/val/videos/'
    )

    params = []
    for idx, img_stem in enumerate(tqdm(identity_list)):
        video_path = video_trajectory_root / f'{img_stem}.jpg' / 'sample_video__elipsoid.mp4'
        video_paths = [video_path]
        geo_video_path = video_trajectory_root / f'{img_stem}.jpg' / 'sample_depth_video__elipsoid.mp4'
        if geo_video_path.exists():
            video_paths.append(geo_video_path)

        # for v_path in (video_path, geo_video_path):
        for v_path in video_paths:
            # for v_path in (video_path, geo_video_path):
            gif_path = Path(v_path).parent / Path(v_path).with_suffix(
                '.gif').name
            # MP4toGif(v_path, gif_path)
            params.append([v_path, gif_path])

    with Pool(56) as p:
        # p.map(MP4toGif, params)
        p.map(MP4toGif, params)


# exportMp4Gif(identity)
exportMp4GifMP(identity)
