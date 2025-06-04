import os
import pydicom
import nibabel as nib
import numpy as np
import pandas as pd 
import glob
from nipype.interfaces import fsl
import nibabel.orientations as nio
import subprocess
import torchio as tio


df = pd.read_csv('data/train_augmented.csv')

output_folder = 'adni'
os.makedirs(output_folder, exist_ok=True)

for path in df['mri_link']:
    s = path.split('/')[1:]
    save_path = output_folder
    for f in s:
        save_path = os.path.join(save_path, f)
        os.makedirs(save_path, exist_ok=True)

    image_path = os.path.join(*save_path.split('/')[:-1])
    image_path = os.path.join(image_path, 'image.nii.gz')
    save_path = os.path.join(save_path, 'image.nii.gz')

    subject = tio.Subject(dti=tio.ScalarImage(image_path))
    transform = tio.RandomAffine(translation=2)
    augmented = transform(subject)

    augmented.dti.save(save_path)

    print("Done", save_path)