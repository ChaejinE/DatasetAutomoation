from collections import defaultdict
from unittest import loader
from extraxt_video import VideoExtractor
from inference.inference import YOLOv5Inference
import os
import shutil
from tqdm import tqdm
from datetime import datetime

def cvtTime2Interval(time1, time2):
    time_1 = datetime.strptime(time1,"%M:%S")
    time_2 = datetime.strptime(time2,"%M:%S")

    return (time_2 - time_1).seconds

def cvtTime2Second(time):
    d = datetime.strptime(time, '%M:%S')
    return d.second + d.minute * 60 + d.hour * 3600

def link_softsymbolic(src, dst, img_names):
    try:
        [ os.symlink(src=os.path.join(os.getcwd(), os.path.relpath(os.path.join(src, file))), \
                    dst=os.path.join(os.getcwd(), os.path.relpath(os.path.join(dst, file)))) \
        for file in img_names ]
    except FileExistsError:
        shutil.rmtree(os.path.join(os.getcwd(), dst))
        os.makedirs(os.path.join(os.getcwd(), dst), exist_ok=True)
        link_softsymbolic(src, dst, img_names)
        

def main(config):
    video_dir = config["video_dir"]
    image_save_dir = config["image_save_dir"]
    video_save_dir = config["video_save_dir"]
    
    inference = YOLOv5Inference(exec_dir=os.path.join(os.getcwd(), "inference"), cfg_path=config["inference_yaml_path"]) \
                if "inference" in config["mode"] else None
                
    all_save_dir = os.path.join(image_save_dir, "all")
    os.makedirs(all_save_dir, exist_ok=True)
        
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
                
                extractor.generate_jpeg(out_file_dir=all_save_dir, start=start, time=time_interval, fps=config["fps"])
                cur_imgNames = extractor.get_current_image_names()
                
                link_dir_detail2 = os.path.join(image_save_dir, video[:-4], f"{time1}to{time2}")
                os.makedirs(link_dir_detail2, exist_ok=True)
                link_softsymbolic(all_save_dir, link_dir_detail2, cur_imgNames)
                if config["inference_detail"] and inference:
                    print(f"detail Inferencing {time1} to {time2}")
                    inference(link_dir_detail2, cvat_dir=config["path_detail_for_cvat"])
            
            link_dir_detail1 = os.path.join(image_save_dir, video[:-4], f"{video[:-4]}_all")
            os.makedirs(link_dir_detail1, exist_ok=True)
            link_softsymbolic(all_save_dir, link_dir_detail1, cur_imgNames)
            if config["inference_sub_all"] and inference:
                print(f"sub all Inferencing {video}")
                inference(link_dir_detail1, cvat_dir=config["path_sub_all_for_cvat"])
            
    if config["inference_all"] and inference:
        print(f"all Inferencing {videos}")
        inference(all_save_dir, cvat_dir=config["path_all_for_cvat"])
                
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
    parser.add_argument("--config_path", default="./make_images.yaml")
    args = parser.parse_args()
    
    with open(args.config_path, 'r') as f:
        import yaml
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    import time
    start = time.time()
    main(cfg)
    print(f"Prgram runtime : {time.time() - start}")