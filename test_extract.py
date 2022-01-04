import unittest
import ffmpeg

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

def generate_section(file_name, out_file_name, time, width):
    (
        ffmpeg
        .input(file_name, t=120)
        .output(out_file_name, t=60)
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True)
    )
    

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
        generate_section(self.video_name, "test.mp4", 100, 120)
            
if __name__ == "__main__":
    unittest.main()