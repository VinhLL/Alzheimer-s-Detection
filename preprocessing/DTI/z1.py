import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import pandas as pd


def dicom_to_nifti(dicom_path, output_dir):
    """ Chuyển đổi DICOM sang NIfTI sử dụng dcm2niix. """
    try:
        subprocess.run([
            "dcm2niix",
            "-z", "y",                 # nén .nii.gz
            "-f", "dti",               # tên tệp đầu ra
            "-o", output_dir,
            dicom_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(e)
        raise RuntimeError("Failed to convert DICOM to NIfTI using dcm2niix.")

    nii_file = os.path.join(output_dir, "dti.nii.gz")
    bvecs = os.path.join(output_dir, "dti.bvec")
    bvals = os.path.join(output_dir, "dti.bval")
    return nii_file, bvecs, bvals


def extract_b0(nii_file, output_dir):
    """ 
    Trích ảnh b=0 đầu tiên từ NIfTI.

    Ảnh b0 dùng dùng để thực hiện skull stripping.
    """
    b0_file = os.path.join(output_dir, "b0.nii.gz")
    try:
        subprocess.run([
            "fslroi", nii_file, b0_file, "0", "1"
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(e)
        raise RuntimeError("Failed to extract b0 image using fslroi.")
    
    return b0_file


def skull_strip(b0_file, output_dir):
    """ Thực hiện skull stripping trên ảnh b0 sử dụng BET. """
    bet_output = os.path.join(output_dir, "nodif_brain")
    try:
        subprocess.run([
            "bet", b0_file, bet_output, '-f', '0.35', "-m"
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(e)
        raise RuntimeError("Failed to perform skull stripping using BET.")
    return bet_output


# def eddy_correction(nii_file, output_dir):
#     """ Thực hiện eddy current correction sử dụng eddy_correct. """
#     dti_ec = os.path.join(output_dir, "dti_ec.nii.gz")
#     try:
#         subprocess.run([
#             "eddy_correct", nii_file, dti_ec, "0"
#         ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#     except subprocess.CalledProcessError as e:
#         print(e)
#         raise RuntimeError("Failed to perform eddy current correction using eddy_correct.")
#     return dti_ec


def fit_dti(dti_ec, bet_output, bvecs, bvals, output_dir):
    """ Fit DTI model để tạo FA image sử dụng dtifit. """
    try:
        subprocess.run([
            "dtifit",
            "-k", dti_ec,
            "-o", os.path.join(output_dir, "dti"),
            "-m", f"{bet_output}_mask.nii.gz",
            "-r", bvecs,
            "-b", bvals
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(e)
        raise RuntimeError("Failed to fit DTI model using dtifit.")

    return os.path.join(output_dir, "dti_FA.nii.gz")


def register_to_mni(fa_image, output_dir, mni_template):
    """ Đăng ký FA image với template MNI sử dụng FLIRT. """
    output_registered = os.path.join(output_dir, "image.nii.gz")
    matrix_file = os.path.join(output_dir, "fa2mni.mat")
    
    try:
        subprocess.run([
            "flirt",
            "-in", fa_image,
            "-ref", mni_template,
            "-out", output_registered,
            "-omat", matrix_file,
            "-dof", "12" 
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(e)
        raise RuntimeError("Failed to register FA image to MNI template using FLIRT.")

    return output_registered


def remove_intermediate_files(output_dir):
    """ Xoá các tệp trung gian không cần thiết, chỉ giữ lại ảnh đã đăng ký. """
    files_to_keep = ["image.nii.gz"]
    all_files = os.listdir(output_dir)
    files_to_delete = [f for f in all_files if f not in files_to_keep ]

    for file in files_to_delete:
        file_path = os.path.join(output_dir, file)
        if os.path.exists(file_path):
            os.remove(file_path)

    return os.path.join(output_dir, "dti_FA.nii.gz")


def process_dti(dicom_path, output_dir, mni_template):
    """ Xử lý DTI từ DICOM đến FA image đã đăng ký với MNI template."""
    nii_file, bvecs, bvals = dicom_to_nifti(dicom_path, output_dir)
    b0_file = extract_b0(nii_file, output_dir)
    bet_output = skull_strip(b0_file, output_dir)
    # dti_ec = eddy_correction(nii_file, output_dir)
    # fa_image = fit_dti(dti_ec, bet_output, bvecs, bvals, output_dir)
    fa_image = fit_dti(nii_file, bet_output, bvecs, bvals, output_dir)
    register_to_mni(fa_image, output_dir, mni_template)
    


def get_leaf_directories(root_path):
    """ Duyệt thư mục và trả về các thư mục "lá" (không chứa thư mục con). """
    leaf_dirs = []

    with os.scandir(root_path) as entries:
        subdirs = [entry for entry in entries if entry.is_dir()]
        
    if not subdirs:
        return [root_path]

    for subdir in subdirs:
        leaf_dirs.extend(get_leaf_directories(subdir.path))

    return leaf_dirs


def process_single_subject(dicom_path, output_folder, template_path):
    """ Xử lý một folder DICOM và ghi output. """
    try:
        output_dir = os.path.join(*dicom_path.split("/")[7:])  # Tạo relative path
        output_dir = os.path.join(output_folder, output_dir)
        os.makedirs(output_dir, exist_ok=True)

        print(f"Processing {dicom_path}")
        process_dti(dicom_path, output_dir, template_path)
        remove_intermediate_files(output_dir)
        print(f"✅ Done {dicom_path}")
    except Exception as e:
        print(f"❌ Failed {dicom_path}: {e}")
        # ghi dữ liệu lỗi vào file lỗi:
        with open("error_log.txt", "a") as f:
            f.write(f"{dicom_path}\n")


def t1(path):
    s = path.split("/")
    return os.path.join("data", "dti" ,s[7], s[8], s[9][:10], *s[10:]).lower().replace(" ", "_")

def t2(path):
    s = path.split("/")
    t = os.path.join("data", "DTI" ,*s[7:])
    return t

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    output_folder = os.path.join(script_dir, "adni")
    os.makedirs(output_folder, exist_ok=True)

    # data_path = os.path.join(script_dir, "data", "DHCN_C4_DTI_Additional_dataset")
    # data_path = os.path.join(script_dir, "data", "DHCN_C4_DTI_Additional")
    # data_path = os.path.join(script_dir, "data", "DHCN_C4_DTI_Additional1")
    # data_path = os.path.join(script_dir, "data", "DHCN_C4_DTI_Additional2")
    # data_path = os.path.join(script_dir, "data", "DHCN_C4_DTI_Additional3")
    # data_path = os.path.join(script_dir, "data", "DHCN_C4_DTI_Additional4")
    # data_path = os.path.join(script_dir, "data", "DHCN_C4_DTI_Additional5")
    # data_path = os.path.join(script_dir, "data", "DHCN_C4_DTI_Additional6")
    # data_path = os.path.join(script_dir, "data", "DHCN_C4_DTI_Additional7")
    # data_path = os.path.join(script_dir, "data", "DHCN_C4_DTI_Additional8")

    # data_path = os.path.join(script_dir, "data", "DHCN_C4_DTI (1)")
    # data_path = os.path.join(script_dir, "data", "DHCN_C4_DTI1 (1)")
    # data_path = os.path.join(script_dir, "data", "DHCN_C4_DTI2")
    # data_path = os.path.join(script_dir, "data", "DHCN_C4_DTI3")
    # data_path = os.path.join(script_dir, "data", "DHCN_C4_DTI4")
    # data_path = os.path.join(script_dir, "data", "DHCN_C4_DTI5")
    # data_path = os.path.join(script_dir, "data", "DHCN_C4_DTI6")
    # data_path = os.path.join(script_dir, "data", "DHCN_C4_DTI7")
    # data_path = os.path.join(script_dir, "data", "DHCN_C4_DTI8")
    # data_path = os.path.join(script_dir, "data", "DHCN_C4_DTI9")

    # data_path = os.path.join(script_dir, "data", "DHCN_C4_DTI2_dataset")
    data_path = os.path.join(script_dir, "data", "DHCN_C4_DTI2")



    

    template_path = 'templates/FMRIB58_FA_1mm.nii.gz'

    data_paths = get_leaf_directories(data_path)

    df = pd.read_csv("data/final_dataset.csv")
    # dti_links = set([link.lower().replace(" ", "_") for link in df['dti_link']])
    dti_links = set([link for link in df['dti_link']])


    # data_paths = [path for path in data_paths if t1(path) in dti_links]
    data_paths = [path for path in data_paths if t2(path) in dti_links]

    # lưu data_paths ra file.
    with open(f"data/{data_path.split('/')[-1]}.txt", "w") as f:
        for path in data_paths:
            f.write(f"{path}\n")

    workers = 4

    with ProcessPoolExecutor(max_workers=workers) as executor:
        task = partial(process_single_subject, output_folder=output_folder, template_path=template_path)
        executor.map(task, data_paths)





if __name__ == "__main__":
    main()