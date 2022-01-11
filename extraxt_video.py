import ffmpeg
import numpy as np
import cv2
import os

class VideoExtractor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)
        self.video_stream = self.get_video_stream()
        self.width = self.get_video_width()
        self.height = self.get_video_height()
    
    def load_video(self):
        video = ffmpeg.input(self.file_path)
        return video

    def get_video_stream(self):
        probe = ffmpeg.probe(self.file_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        return video_stream

    def get_video_width(self):
        return int(self.video_stream["width"])

    def get_video_height(self):
        return int(self.video_stream["height"])

    def generate_section(self, out_file_path, start, time):
        """
        file_name에 해당하는 비디오를 start 부터 time 시간동안 추출
        """
        (
            ffmpeg
            .input(self.file_path)
            .output(out_file_path, ss=start, t=time)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        
    def gen_section_stream(self, out_file_path, start, time):
        return (
                ffmpeg
                .input(self.file_path)
                .output(out_file_path, ss=start, t=time)
                .overwrite_output()
                )
        
    def generate_jpeg(self, out_file_dir, start, time, fps, resize=None, is_gray=None):
        out, _ = (
            ffmpeg
            .input(self.file_path, ss=start, t=time)
            .filter("fps", fps)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True, capture_stderr=True)
        )
        
        ch = 1 if is_gray else 3
        result = np.frombuffer(out, np.uint8).reshape([-1, self.height, self.width, ch])
        
        for idx, img in enumerate(result):
            img = cv2.resize(img, dsize=resize) if resize is not None else img
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f"{out_file_dir}/{self.file_name[:-4]}_{start}-{time}_{idx}.jpg", img)
        
    def get_jpeg_images(self, start, time, is_gray=None):
        out, _ = (
            ffmpeg
            .input(self.file_path, ss=start, t=time)
            .filter('select', 'mod(n\,{})'.format(16))
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True, capture_stderr=True)
        )
        
        ch = 1 if is_gray else 3
        return np.frombuffer(out, np.uint8).reshape([-1, self.height, self.width, ch])
            
if __name__ == "__main__":
    extractor = VideoExtractor("155633.mp4")
    # extractor.generate_section("test2.mp4", start=120, time=30)
    extractor.generate_jpeg("./test_save", start=120, time=2)