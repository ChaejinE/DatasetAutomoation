import unittest
import ffmpeg
import numpy as np
import cv2

def load_video(file_name):
    video = ffmpeg.input(file_name)
    return video

def get_video_stream(file_name):
    probe = ffmpeg.probe(file_name)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    return video_stream

def get_video_width(video_stream):
    return video_stream["width"]

def get_video_height(video_stream):
    return video_stream["height"]

def generate_section(file_name, out_file_name, start, time):
    """
    file_name에 해당하는 비디오를 start 부터 time 시간동안 추출
    """
    (
        ffmpeg
        .input(file_name)
        .output(out_file_name, ss=start, t=time)
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True)
    )
    
def gen_section_stream(file_name, out_file_name, start, time):
    return (
            ffmpeg
            .input(file_name)
            .output(out_file_name, ss=start, t=time)
            .overwrite_output()
           )
    
def generate_jpeg(file_name, start, time):
    out, err = (
        ffmpeg
        .input(file_name, ss=start, t=time)
        .filter('select', 'gte(n,{})'.format(5))
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')#, vframes=2)
        .run(capture_stdout=True)
    )
    
    result = np.frombuffer(out, np.uint8).reshape([-1, 1944, 2592, 3])
    
    for idx, img in enumerate(result):
        img = cv2.resize(img, dsize=(832, 832))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"./test_save/tt{idx}.jpg", img)

class ExtractTest(unittest.TestCase):
    def setUp(self):
        self.video_name = "155633.mp4"
        self.video = load_video(self.video_name)
        self.video_stream = get_video_stream(self.video_name)
        
    def tearDown(self):
        pass
    
    def test_load_video(self):
        "video load test"
        video = load_video(self.video_name)
        self.assertNotEqual(video, None)
        
    def test_get_video_stream(self):
        "video stream obj test"
        video_stream = get_video_stream(self.video_name)
        self.assertNotEqual(video_stream, None)
        self.assertNotEqual(video_stream, {})
        
    def test_get_video_width(self):
        """
        width info test
        """
        width = get_video_width(self.video_stream)
        self.assertIsInstance(width, int)
        
    def test_get_video_height(self):
        """
        height info test
        """
        height = get_video_height(self.video_stream)
        self.assertIsInstance(height, int)
        
    def test_generate_section(self):
        """
        generating section video  test
        """
        generate_section(self.video_name, "test.mp4", 100, 30)
        
    def test_gen_section_stream(self):
        """
        getting section video stream test
        """
        video_stream = gen_section_stream(self.video_name, "test.mp4", 100, 60)
        self.assertIsNotNone(video_stream)
        
        video_stream.run()
        
    def test_generate_jpeg(self):
        """
        gernerate jpeg (video to jpeg)
        """
        generate_jpeg(self.video_name, "test.mp4", 100, 2)
            
if __name__ == "__main__":
    unittest.main()