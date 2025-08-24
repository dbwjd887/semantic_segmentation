import os
import sys
import shutil
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import pydicom
from pydicom.uid import generate_uid
import nibabel as nib
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt

from totalsegmentator.python_api import totalsegmentator
from totalsegmentator.map_to_binary import class_map
from totalsegmentator.map_to_total import map_to_total


IS_WINDOWS = sys.platform.startswith("win")
try:
    import fury  # noqa: F401
    HAS_FURY = True
except Exception:
    HAS_FURY = False
USE_PREVIEW = (HAS_FURY and (not IS_WINDOWS))
if not USE_PREVIEW:
    print("[INFO] TotalSegmentator preview will be deactivated in Windows.")


def scan_series_groups(dicom_dir: str) -> dict:
    groups = defaultdict(list)
    for name in os.listdir(dicom_dir):
        if name.startswith('.'):
            continue
        fp = os.path.join(dicom_dir, name)
        try:
            ds = pydicom.dcmread(fp, stop_before_pixels=False)
            _ = ds.pixel_array
            if not hasattr(ds, "SeriesInstanceUID"):
                continue
            groups[ds.SeriesInstanceUID].append(ds)
        except Exception:
            continue
    if not groups:
        raise RuntimeError("There is no valid DICOM series.")
    return groups


def sort_series_by_normal(slices: List[pydicom.dataset.FileDataset]) -> Tuple[List[pydicom.dataset.FileDataset], np.ndarray]:
    iop = np.array(slices[0].ImageOrientationPatient, dtype=float)
    row_cos, col_cos = iop[:3], iop[3:]
    normal = np.cross(row_cos, col_cos)

    def ipp_proj(ds):
        ipp = np.array(ds.ImagePositionPatient, dtype=float)
        return float(np.dot(ipp, normal))

    slices = [ds for ds in slices if hasattr(ds, "ImagePositionPatient") and hasattr(ds, "PixelSpacing")]
    slices.sort(key=ipp_proj)
    return slices, normal


def select_series(dicom_dir: str, prefer_uid: str = None) -> Tuple[str, List[pydicom.dataset.FileDataset], np.ndarray]:
    groups = scan_series_groups(dicom_dir)
    if prefer_uid and prefer_uid in groups:
        uid = prefer_uid
    else:
        uid = max(groups.items(), key=lambda kv: len(kv[1]))[0] 
    slices, normal = sort_series_by_normal(groups[uid])
    print("[INFO] 선택 SeriesInstanceUID:", uid, "| 슬라이스 수:", len(slices))
    for k, v in groups.items():
        print(f"   - UID {k}: {len(v)} slices")
    return uid, slices, normal


def dicom_slices_to_nifti(slices: List[pydicom.dataset.FileDataset], nifti_path: str, enforce_ras: bool = True):
    iop = np.array(slices[0].ImageOrientationPatient, dtype=float)
    row_cos, col_cos = iop[:3], iop[3:]
    normal = np.cross(row_cos, col_cos)

    ps = np.array(slices[0].PixelSpacing, dtype=float)
    row_spacing, col_spacing = float(ps[0]), float(ps[1])

    if hasattr(slices[0], "SpacingBetweenSlices"):
        try:
            slice_spacing = float(slices[0].SpacingBetweenSlices)
        except Exception:
            slice_spacing = None
    else:
        slice_spacing = None
    if not slice_spacing or slice_spacing <= 0:
        zs = np.array([np.dot(np.array(ds.ImagePositionPatient, dtype=float), normal) for ds in slices], dtype=float)
        diffs = np.diff(zs)
        if diffs.size == 0:
            raise RuntimeError("한 장짜리 시리즈입니다. slice spacing 계산 불가.")
        slice_spacing = float(np.median(np.abs(diffs)))

    vol = []
    for ds in slices:
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        vol.append(arr * slope + intercept)
    vol = np.stack(vol, axis=-1)

    M = np.eye(4, dtype=float)
    M[:3, 0] = col_cos * col_spacing
    M[:3, 1] = row_cos * row_spacing
    M[:3, 2] = normal  * slice_spacing
    M[:3, 3] = np.array(slices[0].ImagePositionPatient, dtype=float)

    if enforce_ras:
        lps_to_ras = np.diag([-1, -1, 1, 1])
        M = lps_to_ras @ M

    img = nib.Nifti1Image(vol, M)
    img.set_qform(M, code=1); img.set_sform(M, code=1)
    nib.save(img, nifti_path)

    axcodes = nib.aff2axcodes(img.affine)
    print(f"[INFO] NIfTI 저장: {nifti_path}  orientation={axcodes}, spacing≈({col_spacing:.3f},{row_spacing:.3f},{slice_spacing:.3f})")


def resample_mask_to_ct_grid(seg_nifti_path: str, series_files_sorted: List[str]) -> Tuple[np.ndarray, sitk.Image]:
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(series_files_sorted)  
    ct = reader.Execute()
    seg = sitk.ReadImage(seg_nifti_path)
    seg_on_ct = sitk.Resample(seg, ct, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    mask_slices = sitk.GetArrayFromImage(seg_on_ct) 
    return mask_slices.astype(np.uint8), ct


def label_name_to_index(task: str, name: str) -> int:
    canonical = map_to_total.get(name, name)
    idx2name = class_map[task]
    name2idx = {v: k for k, v in idx2name.items()}
    if canonical not in name2idx:
        raise KeyError(f"'{name}'(정규화 '{canonical}') 라벨이 task='{task}'에 없습니다.")
    return name2idx[canonical]


def estimate_midsagittal_x_from_ct(ct_image: sitk.Image, hu_threshold: float = -300.0) -> int:
    arr = sitk.GetArrayFromImage(ct_image)  
    body = arr > hu_threshold
    if not np.any(body):
        mid_x = arr.shape[2] // 2
        print("[WARN] Body mask 비어 있음. x 중앙으로 대체:", mid_x)
        return mid_x
    xs = np.where(body)[2]
    mid_x = int(round(0.5 * (xs.min() + xs.max())))
    print(f"[INFO] mid-sagittal x index: {mid_x}  (x_min={xs.min()}, x_max={xs.max()})")
    return mid_x


def save_mask_series_as_dicom(mask_slices: np.ndarray,
                              slices_sorted: List[pydicom.dataset.FileDataset],
                              out_root: str,
                              series_description: str,
                              set_modality_ot: bool = True):
    """
    mask_slices: (z,y,x) = (num_slices, Rows, Cols), 0/1 또는 bool
    slices_sorted: select_series()에서 반환한 '같은 시리즈'의 정렬된 DICOM들
    out_root/series_description/ 아래에 슬라이스별 파일 저장
    """
    os.makedirs(out_root, exist_ok=True)
    out_dir = os.path.join(out_root, series_description)
    os.makedirs(out_dir, exist_ok=True)

    num_slices = len(slices_sorted)
    rows, cols = slices_sorted[0].Rows, slices_sorted[0].Columns
    if mask_slices.shape != (num_slices, rows, cols):
        raise ValueError(f"mask shape {mask_slices.shape} != expected {(num_slices, rows, cols)}")

    mask_slices = (mask_slices > 0).astype(np.uint8)

    new_series_uid = generate_uid()
    for i, (mask, src) in enumerate(zip(mask_slices, slices_sorted), start=1):
        ds = src.copy()
        ds.SOPInstanceUID = generate_uid()
        ds.SeriesInstanceUID = new_series_uid
        ds.SeriesDescription = series_description
        ds.ImageType = ['DERIVED', 'SECONDARY', 'BINARY']
        if set_modality_ot:
            ds.Modality = 'OT'  
   

        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = 'MONOCHROME2'
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
        ds.SmallestImagePixelValue = 0
        ds.LargestImagePixelValue = 1

        if hasattr(ds, "RescaleSlope"): del ds.RescaleSlope
        if hasattr(ds, "RescaleIntercept"): del ds.RescaleIntercept
        if hasattr(ds, "WindowCenter"): del ds.WindowCenter
        if hasattr(ds, "WindowWidth"): del ds.WindowWidth

        ds.Rows, ds.Columns = rows, cols
        ds.PixelData = mask.tobytes()

        out_path = os.path.join(out_dir, f"{i:04d}.dcm")
        ds.save_as(out_path)

    print(f"[INFO] Saved per-slice mask DICOM series: {out_dir}")


def cylinder_in_mask(
        binary_mask: np.ndarray,
        reference_nii: nib.Nifti1Image,
        out_path: str,  
        radius_mm: float = 10.0, 
        height_mm: float | None = None,  
        margin_mm: float = 1.5 
    ) -> np.ndarray:
    
    affine = reference_nii.affine 
    header = reference_nii.header
    vx, vy, vz = reference_nii.header.get_zooms()[:3] 

    edt = distance_transform_edt(binary_mask, sampling = (vx, vy, vz))
    cx, cy, cz = map(int, np.unravel_index(np.argmax(edt), edt.shape)) 

    radius = float(radius_mm)
    radius = min(radius, float(edt[cx, cy, cz]) - float(margin_mm))

    if radius <= 0:
        raise ValueError("실린더 넣을 공간 없음")
    
    if height_mm is not None:
        height = float(height_mm)
    else:
        half_height_up = 0 
        z = cz 
        while z + 1 < binary_mask.shape[2]:
            z += 1
            if (edt[cx, cy, z] < radius + margin_mm):
                break
            half_height_up += vz

        half_height_down = 0
        z = cz
        while z - 1 >= 0:
            z -= 1
            if (edt[cx, cy, z] < radius + margin_mm):
                break
            half_height_down += vz

        half_height_up = max(0, half_height_up - float(margin_mm)) 
        half_height_down = max(0, half_height_down - float(margin_mm))
        height = max(vz, half_height_up + half_height_down) 

        box_x = int(np.ceil(radius / vx)) + 2
        box_y = int(np.ceil(radius / vy)) + 2
        box_z = int(np.ceil(height / vz)) + 2
        
        x0, x1 = max(0, cx - box_x), min(binary_mask.shape[0], cx + box_x + 1) 
        y0, y1 = max(0, cy - box_y), min(binary_mask.shape[1], cy + box_y + 1)
        z0, z1 = max(0, cz - box_z), min(binary_mask.shape[2], cz + box_z + 1)

        X, Y, Z = np.meshgrid(
            np.arange(x0, x1),
            np.arange(y0, y1),
            np.arange(z0, z1),
            indexing="ij"
        )

        dist_xy2 = ((X - cx) * vx) ** 2 + ((Y - cy) * vy) ** 2 
        dist_z = np.abs((Z - cz) * vz) 

        inside_cyl = ((dist_xy2 <= radius ** 2) & (dist_z <= height))
        
        cyl = np.zeros_like(binary_mask, dtype=np.uint8) 
        cyl[x0:x1, y0:y1, z0:z1] = inside_cyl.astype(np.uint8) 
        
        cyl_in = (cyl & (binary_mask > 0).astype(np.uint8)).astype(np.uint8) 

        nib.save(nib.Nifti1Image(cyl_in, affine, header), out_path) 

        return cyl_in
    

# -----------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    dicom_folder  = r"C:\Users\admin\Desktop\PythonCode\TotalSegmentator\dicom"
    temp_nifti    = r"C:\Users\admin\Desktop\PythonCode\TotalSegmentator\tmp\input.nii.gz"
    out_nifti_dir = r"C:\Users\admin\Desktop\PythonCode\TotalSegmentator\seg_nifti"
    out_dicom_dir = r"C:\Users\admin\Desktop\PythonCode\TotalSegmentator\seg_dicom"

    os.makedirs(os.path.dirname(temp_nifti), exist_ok=True)
    os.makedirs(out_nifti_dir, exist_ok=True)
    os.makedirs(out_dicom_dir, exist_ok=True)
    
    ###################################################################################################
    task = "total"
    roi_subset = ['liver', 'aorta', 'common_carotid_artery_left', 'common_carotid_artery_right']
    # roi_subset = []
    fast = True
    ###################################################################################################
  
    selected_uid, slices_sorted, normal = select_series(dicom_folder)
    series_files_sorted = [ds.filename for ds in slices_sorted]

    dicom_slices_to_nifti(slices_sorted, temp_nifti, enforce_ras=True)

    seg_img = totalsegmentator(
        temp_nifti,
        out_nifti_dir,
        task=task,
        roi_subset=roi_subset if len(roi_subset) > 0 else None,
        device='gpu',
        fast=fast,      
        ml=True,
        output_type="nifti",
        robust_crop=True,
        preview=USE_PREVIEW
    )
    seg_path = os.path.join(out_nifti_dir, "seg.nii.gz")
    nib.save(seg_img, seg_path)

    seg_arr = np.rint(seg_img.get_fdata()).astype(np.uint16)

    mask_slices_dummy, ct_img = resample_mask_to_ct_grid(seg_path, series_files_sorted)  
    expected_z, rows, cols = mask_slices_dummy.shape
    print("[INFO] 재배치 기준 CT 격자 shape(z,y,x):", (expected_z, rows, cols))

    mid_x = estimate_midsagittal_x_from_ct(ct_img, hu_threshold=-300.0)

    if not roi_subset:
        roi_subset = ['liver'] 

    for name in roi_subset:
        try:
            idx = label_name_to_index(task, name)
        except Exception as e:
            print(f"[WARN] '{name}' 인덱스 해석 실패: {e}")
            continue

        bin_vol = (seg_arr == idx).astype(np.uint8)
        tmp_bin = os.path.join(out_nifti_dir, f"tmp_{name}_bin.nii.gz")
        bin_nii = nib.Nifti1Image(bin_vol, seg_img.affine, seg_img.header)
        nib.save(bin_nii, tmp_bin)

        tmp_cyl = os.path.join(out_nifti_dir, f"tmp_{name}_cyl.nii.gz")
        cyl_vol = cylinder_in_mask(
            binary_mask = bin_vol,
            reference_nii = bin_nii,
            out_path = tmp_cyl,
            radius_mm = 10.0,
            height_mm = None,
            margin_mm = 1.5
        )

        mask_slices, _ = resample_mask_to_ct_grid(tmp_bin, series_files_sorted)
        
        '''
        # 좌/우 이름 포함된 모델에 대해서 반대편 데이터 0으로 만들기
        lname = name.lower()
        if "left" in lname:
            mask_slices[:, :, mid_x:] = 0
        elif "right" in lname:
            mask_slices[:, :, :mid_x] = 0
        '''

        save_mask_series_as_dicom(mask_slices, slices_sorted, out_dicom_dir, series_description=name)

    print("[DONE] TS → CT-grid align → per-slice mask DICOM export 완료")
