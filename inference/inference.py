from os import popen
import os
import shutil
import subprocess
import zipfile

class Inference:
    def __init__(self, infrence_dir):
        self._inference_dir = infrence_dir
    
    def __call__(self):
        pass
    
class YOLOv5Inference(Inference):
    def __init__(self, cfg=None, cfg_path=None, exec_dir=os.getcwd()) -> None:
        super().__init__(exec_dir)
        if cfg is not None:
            self.cfg = cfg
        else:
            if cfg_path is not None:
                self.cfg = self._init_config(cfg_path)
            else:
                self.cfg = self._init_config()
        
        self._dir = os.path.join(self._inference_dir, self.cfg["inference_prgm"])
        self._data_dir = self.cfg["temp_root"]
        self._inference_folder_name = self.cfg["inference_folder_name"]
        self._data_yml_name = self.cfg["data_yaml"]
        self._cvat_folder_name = self.cfg["temp_save_folder_name"]
        
        self._num = self._check_sameFolder_num(os.path.join(self._data_dir, self._inference_folder_name))
        self._data_yml_name = self._data_yml_name[:-5] + f"{self._num+1}.yaml" if self._num > 1 else self._data_yml_name
        self._inference_folder_name = self._inference_folder_name + f"{self._num+1}" if self._num > 1 else self._inference_folder_name
        self._cvat_folder_name = self._cvat_folder_name + f"{self._num+1}" if self._num > 1 else self._cvat_folder_name
        
    def _init_config(self, config_path="./yolov5.yaml"):
        with open(config_path, 'r') as f:
            import yaml
            cfg = yaml.load(f, Loader=yaml.FullLoader)
            
        return cfg
    
    @staticmethod
    def _check_sameFolder_num(dir):
        import re
        upper, base = os.path.split(dir)
        check_list = []
        for file in os.listdir(upper):
            if base in file:
                result = re.findall("\d+", file)
                check_list.append(result[-1]) if len(result) > 0 else None
        
        check_list.sort()
        if len(check_list) > 0 and check_list[-1].isdigit():
            return int(check_list[-1])
        else:
            return 1
        
    def _make_data_yml(self, image_dir, save_dir=None):
        self._data_dir = save_dir if save_dir is not None else self._data_dir
        
        data_dict = {
                        "train": None, "val": image_dir, \
                        "nc": len(self.cfg["names"]), "names": self.cfg["names"]
                    }
        
        import yaml
        with open(os.path.join(self._data_dir, self._data_yml_name), "w") as f:
            yaml.safe_dump(data_dict, f, default_flow_style=False, sort_keys=False)
            
    def _make_file_ForLabel(self, image_dir, save_dirForCVAT, inference_dir=None, image_format="jpg"):
        cvat_dir = os.path.join(self._data_dir, self._cvat_folder_name)
        save_dirForCVAT = os.path.join(self.cfg["cvat_save_dir"], save_dirForCVAT)
        
        os.makedirs(cvat_dir, exist_ok=True)
        
        image_paths = [ os.path.join(save_dirForCVAT, file_name) for file_name in os.listdir(image_dir) \
                        if file_name.endswith(image_format) ]
        file1 = os.path.join(cvat_dir, "train.txt")
        with open(file1, 'w') as f:
            [ f.write(path+'\n') for path in sorted(image_paths) ]
        
        
        label_dir = os.path.join(cvat_dir, save_dirForCVAT[5:])
        os.makedirs(label_dir, exist_ok=True)
        
        inference_dir = os.path.join(self._data_dir, self._inference_folder_name, "labels") if inference_dir is None else inference
        inferences = [ os.path.join(inference_dir, file_name) for file_name in os.listdir(inference_dir) if file_name.endswith("txt") ]
        [ shutil.move(inference, os.path.join(label_dir, os.path.basename(inference))) for inference in inferences ]
        
        file2 = os.path.join(cvat_dir, "obj.names")
        with open(file2, 'w') as f:
            [ f.write(cls+'\n') for cls in self.cfg["names"] ]
            
        file3 = os.path.join(cvat_dir, "obj.data")
        with open(file3, 'w') as f:
            f.write(f"classes = {len(self.cfg['names'])}\n")
            f.write("train = data/train.txt\n")
            f.write("names = data/obj.names\n")
            f.write("backup = backup/")
            
        files_for_cvat = os.listdir(cvat_dir)
        [ self._move(os.path.join(cvat_dir, file), os.path.join(os.getcwd(), file)) for file in files_for_cvat ]
        
        with zipfile.ZipFile(os.path.join(image_dir, f"{self.cfg['cvat_zip_file_name']}.zip"), 'w') as my_zip:
            for file_for_cvat in files_for_cvat:
                if os.path.isdir(file_for_cvat):
                    [ my_zip.write(os.path.join(save_dirForCVAT[5:], f)) for f in os.listdir(save_dirForCVAT[5:]) ]
                else:
                    my_zip.write(file_for_cvat)
                
        [ os.remove(os.path.join(os.getcwd(), file)) if os.path.isfile(file)
          else shutil.rmtree(file) for file in files_for_cvat ]
            
        shutil.rmtree(cvat_dir)
        
    def _move(self, src=None, dst=None):
        import shutil
        if src is None or dst is None:
            print("src or dst is None")
            return
        
        try:
            shutil.move(src, dst)
        except shutil.Error:
            print("exist Error, So source directory remove and replay")
            shutil.rmtree(dst)
            shutil.move(src, dst)
            
    def __call__(self, image_dir, cvat_dir):
        self._make_data_yml(image_dir=image_dir)
        exec_file_path = os.path.join(self._dir, "test.py")
        data_path = os.path.join(self._data_dir, self._data_yml_name)
        
        self.popen = subprocess.Popen(["python", exec_file_path, \
                                       "--weights", self.cfg["weights"], \
                                       "--data", data_path, \
                                       "--device", str(self.cfg["gpu"]), \
                                       "--project", self._data_dir, \
                                       "--name", self._inference_folder_name, \
                                       "--img-size", str(self.cfg["img_size"]), \
                                       "--iou-thres", str(self.cfg["iou_thr"]), \
                                       "--conf-thres", str(self.cfg["conf_thr"]), \
                                       "--save-txt"], stdout = subprocess.PIPE)
        
        out, err = self.popen.communicate()
        # print(out.decode('utf-8'))
        if err is None:
            print("YOLOv5 Inferecing Success")
            self._make_file_ForLabel(image_dir=image_dir, save_dirForCVAT=cvat_dir)
            
            try:
                os.remove(os.path.join(self._data_dir, self._data_yml_name))
                shutil.rmtree(os.path.join(self._data_dir, self._inference_folder_name))
            except shutil.Error as e:
                print(f"Last Remove error\n {e}")
        else:
            print(err)
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="./yolov5.yaml")
    args = parser.parse_args()
    
    with open(args.config_path, 'r') as f:
        import yaml
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    inference = YOLOv5Inference(cfg=cfg)
    # inference._make_data_yml(image_dir="/home/chaejin/Desktop/cjlotto/git_clone/personal/DatasetAutomoation/test_save_img/150626/00:45to00:47")
    inference("/home/chaejin/Desktop/cjlotto/git_clone/personal/DatasetAutomoation/t_save_img/all", 
              cvat_dir="PocSite/Hanti/20220110/v1/train/images")
    
    # dst = "/home/chaejin/Desktop/cjlotto/git_clone/personal/DatasetAutomoation/test_save_img/150626/00:45to00:47"
    # inference._move_folder(dst=dst)
    
    # print(inference._check_sameFolder_num("/tmp/labels"))
    
    # inference._make_file_ForLabel(image_dir="/home/chaejin/Desktop/cjlotto/git_clone/personal/DatasetAutomoation/t_save_img/all", \
    #                               save_dirForCVAT="PoC/hanti/v1/train/images")
    