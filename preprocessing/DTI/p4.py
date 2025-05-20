import os
import pydicom
import nibabel as nib
import numpy as np
import pandas as pd 
import glob
from nipype.interfaces import fsl
import nibabel.orientations as nio
import subprocess


# df = pd.read_csv('data/train.csv')
# df = pd.read_csv('data/val.csv')
df = pd.read_csv('data/test.csv')


def dicom_to_nifti(path, save_path):
    dcm_files = glob.glob(os.path.join(path, "*.dcm"))
    slices = [pydicom.dcmread(f) for f in dcm_files]

    if not all(hasattr(s, "ImagePositionPatient") and hasattr(s, "ImageOrientationPatient") for s in slices):
        raise ValueError("Thiếu ImagePositionPatient hoặc ImageOrientationPatient trong DICOM")

    orientation = np.array(slices[0].ImageOrientationPatient).reshape(2, 3)
    row_cosine, col_cosine = orientation
    normal = np.cross(row_cosine, col_cosine)

    positions = np.array([s.ImagePositionPatient for s in slices])
    distances = positions @ normal
    sorted_slices = [s for _, s in sorted(zip(distances, slices), key=lambda x: x[0])]

    img3d = np.stack([s.pixel_array for s in sorted_slices]).astype(np.int16)
    img3d = np.transpose(img3d, (2, 1, 0))

    spacing = list(map(float, slices[0].PixelSpacing))
    try:
        slice_thickness = float(slices[0].SliceThickness)
    except:
        slice_thickness = np.abs(distances[1] - distances[0])

    affine = np.eye(4)
    affine[:3, 0] = row_cosine * spacing[1]
    affine[:3, 1] = col_cosine * spacing[0]
    affine[:3, 2] = normal * slice_thickness
    affine[:3, 3] = slices[0].ImagePositionPatient

    # Chuẩn hóa hướng ảnh về RAS
    affine[0, :] *= -1  # L->R
    affine[1, :] *= -1  # P->A

    nii_img = nib.Nifti1Image(img3d, affine)

    # Tự động chuyển ảnh về chuẩn RAS (tránh xoay/lật)
    nii_img = reorient_to_RAS(nii_img)

    nib.save(nii_img, save_path)

def reorient_to_RAS(img):
    current_ornt = nio.io_orientation(img.affine)
    ras_ornt = nio.axcodes2ornt(('R', 'A', 'S'))
    transform = nio.ornt_transform(current_ornt, ras_ornt)
    reoriented_data = nio.apply_orientation(img.get_fdata(), transform)
    new_affine = img.affine @ nio.inv_ornt_aff(transform, img.shape)
    return nib.Nifti1Image(reoriented_data, new_affine)


def register_to_mni(nni_path, template_path, save_path):
    # Bước 1: Linear registration về MNI bằng FLIRT
    flirt = fsl.FLIRT()
    flirt.inputs.in_file = nni_path
    flirt.inputs.reference = template_path
    flirt.inputs.output_type = "NIFTI_GZ"
    flirt.inputs.out_file = os.path.join(save_path, "image.nii.gz")
    flirt.inputs.out_matrix_file = os.path.join(save_path, "flirt_matrix.mat")
    flirt.inputs.dof = 6  # Bằng 12 nếu ảnh là T1, dùng 6 nếu FA
    flirt.run()

    # Xóa file tạm
    os.remove(os.path.join(save_path, "flirt_matrix.mat"))


def clean_background_with_bet(nifti_path, cleaned_path):
    # Tạo ảnh não đã loại bỏ nền với BET
    subprocess.run(['bet', nifti_path, cleaned_path, '-f', '0.35', '-g', '0', '-m'])

    mask_path = cleaned_path.replace(".nii.gz", "_mask.nii.gz")
    if os.path.exists(mask_path):
        os.remove(mask_path)


output_folder = 'adni'
os.makedirs(output_folder, exist_ok=True)

for path in df['dti_link']:
    s = path.split('/')[1:]
    save_path = output_folder
    for f in s:
        save_path = os.path.join(save_path, f)
        os.makedirs(save_path, exist_ok=True)

    raw_nifti_path = os.path.join(save_path, "image_raw.nii")
    template_path = 'template/FMRIB58_FA_1mm.nii.gz'
    
    dicom_to_nifti(path, raw_nifti_path)
    cleaned_path = os.path.join(save_path, "image2.nii.gz")
    clean_background_with_bet(raw_nifti_path, cleaned_path)
    register_to_mni(cleaned_path, template_path, save_path)

    os.remove(raw_nifti_path)
    os.remove(cleaned_path)
    print(f"Done {path}")



