import cv2 as cv
import numpy as np
import os
from filters import PencilSketch

m = 480
n = 640
DIR_PATH = '../../ZuBuD+/training'
SKETCH_PATH = '../../ZuBuD_Sketch/train'

for file in os.listdir(DIR_PATH):
    file_path = os.path.join(DIR_PATH, file)
    img = cv.imread(file_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    height, width, channels = img.shape
    if (height, width, channels) != (m, n, 3):
	    img = np.resize(img, (m, n, 3))
    pencil = PencilSketch(width, height)
    sketch = pencil.render(img)
    os.makedirs(SKETCH_PATH, exist_ok=True)
    write_path = os.path.join(SKETCH_PATH, file)
    cv.imwrite(write_path, sketch)
