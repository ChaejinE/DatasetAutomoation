from collections import defaultdict
from unittest import loader
from extraxt_video import VideoExtractor
import os
from tqdm import tqdm
from datetime import datetime

def cvtTime2Interval(time1, time2):
    time_1 = datetime.strptime(time1,"%M:%S")
    time_2 = datetime.strptime(time2,"%M:%S")

    return (time_2 - time_1).seconds

def cvtTime2Second(time):
    d = datetime.strptime(time, '%M:%S')
    return d.second + d.minute * 60 + d.hour * 3600

def main(config):
    video_dir = config["video_dir"]
    image_save_dir = config["image_save_dir"]
    video_save_dir = config["video_save_dir"]
    
    videos = config["videos"]
    for video in videos:
        if video_dir is None:
            video_dir = "./"
        
        extractor = VideoExtractor(os.path.join(video_dir, video))
        mode = config["mode"]
        
        if "image" in mode:
            os.makedirs(image_save_dir, exist_ok=True)
            print(f"sections : {config['videos'][video]['sections']}")
            sections = config["videos"][video]["sections"]
            
            for section in tqdm(sections, desc=f"[{video}] Image Extracting..."):
                time1, time2 = section
                start, time_interval = cvtTime2Second(time1), cvtTime2Interval(time1, time2)
                save_dir = os.path.join(image_save_dir, video[:-4], f"{time1}to{time2}")
                os.makedirs(save_dir, exist_ok=True)
                extractor.generate_jpeg(out_file_dir=save_dir, start=start, time=time_interval, fps=config["fps"])
                
                
        if "video" in mode:
            os.makedirs(video_save_dir, exist_ok=True)
            sections = config["videos"][video]["sections"]
            for section in tqdm(sections, desc=f"[{video}] Video Extracting..."):
                start, time = section
                save_dir = os.path.join(video_save_dir, video[:-4])
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{video[:-4]}_{start}to{start+time}.mp4")
                extractor.generate_section(out_file_path=save_path, start=start, time=time)
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="./make_images.yml")
    args = parser.parse_args()
    
    with open(args.config_path, 'r') as f:
        import yaml
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    main(cfg)