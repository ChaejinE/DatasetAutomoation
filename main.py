from collections import defaultdict
from unittest import loader
from extraxt_video import VideoExtractor
import os
from tqdm import tqdm

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
            sections = config["videos"][video]["sections"]
            for section in tqdm(sections, desc=f"[{video}] Image Extracting..."):
                start, time = section
                save_dir = os.path.join(image_save_dir, video[:-4], f"{start}to{start+time}")
                os.makedirs(save_dir, exist_ok=True)
                extractor.generate_jpeg(out_file_dir=save_dir, start=start, time=time, fps=config["fps"])
                
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