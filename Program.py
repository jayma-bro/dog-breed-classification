# chargement des librairies
import cv2 as cv
import numpy as np
import math
from typing import Optional
import tensorflow as tf
from tensorflow.keras.models import load_model

# définition des fonction de preprocessing
def equalize_img(img: np.ndarray) -> np.ndarray:
    img_YUV = cv.cvtColor(img, cv.COLOR_BGR2YUV)
    img_YUV[:,:,0] = cv.equalizeHist(img_YUV[:,:,0])
    img_equ = cv.cvtColor(img_YUV, cv.COLOR_YUV2RGB)
    img_RGB = cv.cvtColor(img_equ, cv.COLOR_BGR2RGB)
    return img_RGB

def resize_img(img: np.ndarray, resolution: int = 200) -> np.ndarray:
    ratio = img.shape[0]/img.shape[1]
    borne_min = 0.75
    borne_max = 1.33334
    if ratio >= borne_min and ratio <= borne_max:
        resize: np.ndarray = cv.resize(img, (resolution, resolution), interpolation=cv.INTER_CUBIC)
    elif ratio < borne_min:
        presize = math.ceil(resolution*ratio*borne_max)
        resize: np.ndarray = cv.resize(img, (resolution, presize), interpolation=cv.INTER_CUBIC)
        rajout = (resolution - presize) /2
        resize: np.ndarray = cv.copyMakeBorder(resize, top=math.ceil(rajout), bottom=math.floor(rajout), left=0, right=0, borderType=cv.BORDER_REPLICATE)
    elif ratio > borne_max:
        presize = math.ceil(resolution*1/(ratio*borne_min))
        resize: np.ndarray = cv.resize(img, (presize, resolution), interpolation=cv.INTER_CUBIC)
        rajout = (resolution - presize) /2
        resize: np.ndarray = cv.copyMakeBorder(resize, top=0, bottom=0, left=math.ceil(rajout), right=math.floor(rajout), borderType=cv.BORDER_REPLICATE)
    else:
        resize = np.array(0)
    return resize

def recadre_inbox(img: np.ndarray, bndbox: dict, square=True) -> np.ndarray:
    if square:
        loop = True
        switch = True
        while loop:
            box_ratio = (bndbox['ymax'] - bndbox['ymin'])/(bndbox['xmax'] - bndbox['xmin'])
            if box_ratio > 1:
                if switch:
                    if bndbox['xmax'] < img.shape[1]:
                        bndbox['xmax'] += 1
                    switch = False
                else:
                    if bndbox['xmin'] > 0:
                        bndbox['xmin'] -= 1
                    switch = True
                if (bndbox['xmax'] >= img.shape[1]) & (bndbox['xmin'] <= 0):
                    loop = False
            elif box_ratio < 1:
                if switch:
                    if bndbox['ymax'] < img.shape[0]:
                        bndbox['ymax'] += 1
                    switch = False
                else:
                    if bndbox['ymin'] > 0:
                        bndbox['ymin'] -= 1
                    switch = True
                if (bndbox['ymax'] >= img.shape[0]) & (bndbox['ymin'] <= 0):
                    loop = False
            else:
                loop = False
    return img[bndbox['ymin']:bndbox['ymax'],bndbox['xmin']:bndbox['xmax']]

def img_process(imgs, bndbox = None, resolution: int = 200, equalize: bool = False, denois: Optional[int] = None) -> np.ndarray:
    imgs_out = []
    for i, img in enumerate(imgs):
        if bndbox != None:
            img = recadre_inbox(img=img, bndbox=bndbox[i])
        img = resize_img(img=img, resolution=resolution)
        if equalize:
            img = equalize_img(img=img)
        if denois != None:
            img = cv.fastNlMeansDenoisingColored(src=img, h=denois)
        imgs_out.append(img)
    return np.array(imgs_out)

# chargement du modèle
model = load_model('model.h5')
labels = np.load('labels.npy')


path = input('Le chemin de la photo : ')
# calcule de prédiction
img = cv.cvtColor(cv.imread(path), cv.COLOR_BGR2RGB)
img = tf.keras.applications.xception.preprocess_input(img_process(imgs=[img], resolution=299))
pred : np.ndarray = model.predict(img)

# retour de prédiction
print('\nc\'est : '+labels[pred.argmax()]+'\navec une certitude de ' + str(pred.max()))