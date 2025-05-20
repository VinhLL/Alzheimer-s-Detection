import os
import glob
import pydicom
import nibabel as nib
import numpy as np
import pandas as pd
import subprocess
from scipy.ndimage import affine_transform
from nipype.interfaces import fsl
import nibabel.orientations as nio

def safe_remove(path):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        print(f"Cannot remove {path}: {e}")

def reorient_to_RAS(img):
    current_ornt = nio.io_orientation(img.affine)
    ras_ornt = nio.axcodes2ornt(('R', 'A', 'S'))
    transform = nio.ornt_transform(current_ornt, ras_ornt)
    reoriented_data = nio.apply_orientation(img.get_fdata(), transform)
    new_affine = img.affine @ nio.inv_ornt_aff(transform, img.shape)
    return nib.Nifti1Image(reoriented_data, new_affine)

def dicom_to_nifti(path, save_path):
    path = path.replace('\\', '/')
    dcm_files = glob.glob(os.path.join(path, "*.dcm"))
    slices = [pydicom.dcmread(f) for f in dcm_files]

    if not all(hasattr(s, "ImagePositionPatient") and hasattr(s, "ImageOrientationPatient") for s in slices):
        raise ValueError("Missing orientation or position metadata in DICOM files.")

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
    affine[0, :] *= -1
    affine[1, :] *= -1

    nii_img = nib.Nifti1Image(img3d, affine)
    nii_img = reorient_to_RAS(nii_img)
    nib.save(nii_img, save_path)

def register_to_mni(in_path, template_path, out_path):
    flirt = fsl.FLIRT()
    flirt.inputs.in_file = in_path
    flirt.inputs.reference = template_path
    flirt.inputs.output_type = "NIFTI_GZ"
    flirt.inputs.out_file = out_path
    flirt.inputs.dof = 12
    flirt.inputs.out_matrix_file = out_path.replace(".nii.gz", "_matrix.mat")
    flirt.run()
    safe_remove(flirt.inputs.out_matrix_file)

def clean_background_and_save(input_path, output_path):
    subprocess.run(['bet', input_path, output_path, '-f', '0.4', '-g', '0', '-m'])
    mask_path = output_path.replace(".nii.gz", "_mask.nii.gz")
    safe_remove(mask_path)

def apply_translation_after_register(input_path, output_paths, max_shift_mm=5.0):
    img = nib.load(input_path)
    data = img.get_fdata()
    affine = img.affine
    header = img.header
    voxel_sizes = header.get_zooms()[:3]

    for out_path in output_paths:
        shifts_mm = np.random.uniform(-max_shift_mm, max_shift_mm, size=3)
        shifts_voxels = shifts_mm / np.array(voxel_sizes)

        translated_data = affine_transform(
            input=data,
            matrix=np.eye(3),
            offset=-shifts_voxels,
            order=3,
            mode='constant',
            cval=0
        )

        translated_img = nib.Nifti1Image(translated_data, affine)
        nib.save(translated_img, out_path)

df = pd.read_csv('data/train.csv')
template_path = 'template/MNI152_T1_1mm.nii.gz'


for i in range(43):  # Điều chỉnh chỉ số dòng tùy theo bộ dữ liệu
    row = df.iloc[i]
    dicom_path = row['mri_link']
    print(f"🔄 Processing [{i}]: {dicom_path}")

    subpath = "/".join(dicom_path.replace("\\", "/").split("/")[1:]) 
    output_dir = os.path.join("adni", subpath)
    os.makedirs(output_dir, exist_ok=True)

    raw_nii = os.path.join(output_dir, "image_raw.nii")
    dicom_to_nifti(dicom_path, raw_nii)

    registered_nii = os.path.join(output_dir, "image_reg.nii.gz")
    register_to_mni(raw_nii, template_path, registered_nii)
    safe_remove(raw_nii)

    final_nii = os.path.join(output_dir, "image.nii.gz")
    clean_background_and_save(registered_nii, final_nii)
    safe_remove(registered_nii)

    print(f"BET completed: {final_nii}")
    print(f"Done original [{i}]: {dicom_path}")
