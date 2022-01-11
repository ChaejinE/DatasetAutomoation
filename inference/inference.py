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
    def __init__(self, dir=os.getcwd()) -> None:
        super().__init__(dir)
        self._dir = os.path.join(self._inference_dir, "AIT.DL.YOLOv5")
        self._data_dir = "/tmp"
        self._inference_folder_name = "labels"
        self._data_yml_name = "neubie_data.yaml"
        self._cvat_folder_name = "cvat"
        
        # self._num = self._check_sameFolder_num(os.path.join("/tmp", self._inference_folder_name))
        self._num = 4
        self._data_yml_name = self._data_yml_name[:-5] + f"{self._num+1}.yaml" if self._num > 1 else self._data_yml_name
        self._inference_folder_name = self._inference_folder_name + f"{self._num+1}" if self._num > 1 else self._inference_folder_name
        self._cvat_folder_name = self._cvat_folder_name + f"{self._num+1}" if self._num > 1 else self._cvat_folder_name
    
    @staticmethod
    def _check_sameFolder_num(dir):
        upper, base = os.path.split(dir)
        check_list = [x for x in os.listdir(upper) if base in x]
        if len(check_list) > 0 and check_list[-1][-1].isdigit():
            return int(check_list[-1][-1])
        else:
            return 1
        
    def _make_data_yml(self, image_dir, save_dir=None):
        self._data_dir = save_dir if save_dir is not None else self._data_dir
        
        data_dict = {"train": None, "val": image_dir, \
                     "nc": 9, "names": ['bicycle', \
                                        'car', \
                                        'motorcycle', \
                                        'person', \
                                        'scooter', \
                                        'dog', \
                                        'bollard', \
                                        'traffic-light-red', \
                                        'traffic-light-green'] \
                    }
        
        import yaml
        with open(os.path.join(self._data_dir, self._data_yml_name), "w") as f:
            yaml.safe_dump(data_dict, f, default_flow_style=False, sort_keys=False)
            
    def _make_file_ForLabel(self, image_dir, save_dirForCVAT, inference_dir=None, image_format="jpg"):
        cvat_dir = os.path.join(self._data_dir, self._cvat_folder_name)
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
        
        classes = ['bicycle', \
                   'car', \
                   'motorcycle', \
                   'person', \
                   'scooter', \
                   'dog', \
                   'bollard', \
                   'traffic-light-red', \
                   'traffic-light-green']
        file2 = os.path.join(cvat_dir, "obj.names")
        with open(file2, 'w') as f:
            [ f.write(cls+'\n') for cls in classes ]
            
        class_num = str(len(classes))
        file3 = os.path.join(cvat_dir, "obj.data")
        with open(file3, 'w') as f:
            f.write(class_num+'\n')
            f.write("train = data/train.txt\n")
            f.write("names = data/obj.naems\n")
            f.write("backup = backup/")
            
        files_for_cvat = os.listdir(cvat_dir)
        for file_for_cvat in files_for_cvat:
            shutil.move(os.path.join(cvat_dir, file_for_cvat), os.path.join(image_dir, file_for_cvat))
            
        with zipfile.ZipFile(os.path.join(image_dir, "for_cvat.zip"), 'w') as my_zip:
            for file_for_cvat in files_for_cvat:
                my_zip.write(os.path.join(image_dir, file_for_cvat))
            my_zip.close()
        
        [ shutil.move(os.path.join(label_dir, os.path.basename(inference)), inference) for inference in inferences ]
        shutil.rmtree(cvat_dir)
        
    def _move_folder(self, src=None, dst=None):
        import shutil
        src = os.path.join("/tmp", self._inference_folder_name, "labels") if src is None else src
        dst = os.path.join(dst, "labels")
        
        try:
            shutil.move(src, dst)
        except shutil.ExecError as e:
            print("exist Error, So source directory remove and replay")
            shutil.rmtree(dst)
            shutil.move(src, dst)
            
        shutil.rmtree(os.path.join(self._data_dir, self._inference_folder_name))
            
    def __call__(self, image_dir):
        self._make_data_yml(image_dir=image_dir)
        exec_file_path = os.path.join(self._dir, "test.py")
        data_path = os.path.join(self._data_dir, self._data_yml_name)
        
        self.popen = subprocess.Popen(["python", exec_file_path, \
                                       "--weights", "/home/chaejin/Desktop/cjlotto/neubility/weights/SongpaKT/best_80eph.pt", \
                                       "--data", data_path, \
                                       "--device", '0', \
                                       "--project", "/tmp", \
                                       "--name", self._inference_folder_name, \
                                       "--img-size", "832", \
                                       "--iou-thres", "0.6", \
                                       "--conf-thres", "0.4", \
                                       "--save-txt"], stdout = subprocess.PIPE)
        
        out, err = self.popen.communicate()
        print(out.decode('utf-8'))
        print(err)
        
        # self._move_folder(dst=image_dir)
        
if __name__ == "__main__":
    inference = YOLOv5Inference()
    # inference._make_data_yml(image_dir="/home/chaejin/Desktop/cjlotto/git_clone/personal/DatasetAutomoation/test_save_img/150626/00:45to00:47")
    # inference("/home/chaejin/Desktop/cjlotto/git_clone/personal/DatasetAutomoation/test_save_img/150626/00:45to00:47")
    
    # dst = "/home/chaejin/Desktop/cjlotto/git_clone/personal/DatasetAutomoation/test_save_img/150626/00:45to00:47"
    # inference._move_folder(dst=dst)
    
    # print(inference._check_sameFolder_num("/tmp/labels"))
    
    inference._make_file_ForLabel(image_dir="/home/chaejin/Desktop/cjlotto/git_clone/personal/DatasetAutomoation/test_save_img/150626/00:45to00:47", \
                                  save_dirForCVAT="data/obj_train_data/od/PocSite/Songpa/20211214/v1/test/images")
    