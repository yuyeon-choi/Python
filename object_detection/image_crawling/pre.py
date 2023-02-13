
import os
from distutils.dir_util import copy_tree

fromDir = "/kaggle/input/yolo-drone-detection-dataset"
toDir = "./temp"

copy_tree(fromDir,toDir)