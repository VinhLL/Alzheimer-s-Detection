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
import multiprocessing
import dicom2nifti
import warnings
warnings.filterwarnings("ignore") 

def safe_remove(path):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        print(f"Không thể xóa {path}: {e}")

def reorient_to_RAS(img):
    current_ornt = nio.io_orientation(img.affine)
    ras_ornt = nio.axcodes2ornt(('R', 'A', 'S'))
    transform = nio.ornt_transform(current_ornt, ras_ornt)
    reoriented_data = nio.apply_orientation(img.get_fdata(), transform)
    new_affine = img.affine @ nio.inv_ornt_aff(transform, img.shape)
    return nib.Nifti1Image(reoriented_data, new_affine)

def dicom_to_nifti(dicom_dir, save_path):
    try:
        dicom_files = glob.glob(os.path.join(dicom_dir.replace('\\', '/'), "*.dcm")) or \
                      glob.glob(os.path.join(dicom_dir.replace('\\', '/'), "I*"))
        if len(dicom_files) != 1:
            raise ValueError(f"Thư mục {dicom_dir} không chứa đúng một file DICOM: {len(dicom_files)} files")
        
        dicom_file = dicom_files[0]
        if not os.path.exists(dicom_file):
            raise FileNotFoundError(f"File DICOM không tồn tại: {dicom_file}")
        try:
            pydicom.dcmread(dicom_file)
        except Exception as e:
            raise ValueError(f"File DICOM không hợp lệ: {dicom_file}, lỗi: {str(e)}")

        output_dir = os.path.dirname(save_path)
        dicom2nifti.convert_directory(dicom_dir, output_dir, compression=True)
        
        nifti_files = glob.glob(os.path.join(output_dir, "*.nii.gz"))
        if not nifti_files:
            raise ValueError(f"Không tìm thấy file NIfTI sau khi chuyển đổi từ {dicom_dir}")
        
        os.rename(nifti_files[0], save_path)
        
        nii_img = nib.load(save_path)
        nii_img = reorient_to_RAS(nii_img)
        nib.save(nii_img, save_path)
        
        return True
    except Exception as e:
        print(f"Lỗi khi chuyển DICOM sang NIfTI cho {dicom_dir}: {str(e)}")
        return False

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
    subprocess.run(['bet', input_path, output_path, '-f', '0.38', '-g', '0', '-m'])
    mask_path = output_path.replace(".nii.gz", "_mask.nii.gz")
    safe_remove(mask_path)


def process_subject(row, template_path):
    dicom_path = row['mri_link']
    print(f"🔄 Đang xử lý: {dicom_path}")

    subpath = "/".join(dicom_path.replace("\\", "/").split("/")[1:])
    output_dir = os.path.join("adni", subpath)
    os.makedirs(output_dir, exist_ok=True)

    raw_nii = os.path.join(output_dir, "image_raw.nii.gz")
    if not dicom_to_nifti(dicom_path, raw_nii):
        return

    registered_nii = os.path.join(output_dir, "image_reg.nii.gz")
    try:
        register_to_mni(raw_nii, template_path, registered_nii)
        safe_remove(raw_nii)
    except Exception as e:
        print(f"Lỗi khi đăng ký MNI cho {dicom_path}: {e}")
        safe_remove(raw_nii)
        return

    final_nii = os.path.join(output_dir, "image.nii.gz")
    try:
        clean_background_and_save(registered_nii, final_nii)
        safe_remove(registered_nii)
    except Exception as e:
        print(f"Lỗi khi tách nền cho {dicom_path}: {e}")
        safe_remove(registered_nii)
        return

    print(f"BET hoàn tất: {final_nii}")
    print(f"Hoàn tất: {dicom_path}")

if __name__ == "__main__":
    df = pd.read_csv('data/final_dataset.csv')
    template_path = 'template/MNI152_T1_1mm.nii.gz'

    num_processes = min(4, multiprocessing.cpu_count() - 1)
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.starmap(process_subject, [(row, template_path) for _, row in df.iterrows()])