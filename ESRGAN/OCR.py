import easyocr
import pandas as pd
import glob
import re
import cv2
import torch

def Excl():
    list_Master_OCR = []
    ocr = easyocr.Reader(['en'])
    print('OCR Process Started!!')
    test_img_folder = 'ESRGAN/results/*'
    for path in glob.glob(test_img_folder):
        IDD = re.search('_[\d]+',path)
        IDD = path[IDD.start():IDD.end()]
        ocr_result = ocr.readtext(path)
        conc = ''
        for iocr in ocr_result:
            conc = conc + iocr[1]
        list_Master_OCR.append([IDD, conc])

    df = pd.DataFrame(list_Master_OCR, columns=['ID', 'Number_Plate'])
    df.to_csv('D:/Virtual_Env/GitHub_projects/ATCC_Yolov5/NB_plate_SR.csv',index=False)
    print('OCR Process Complete!!')