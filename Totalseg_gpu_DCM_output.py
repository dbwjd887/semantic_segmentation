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
from scipy.ndimage import distance_transform_edt # Binary mask에서 각 Voxel 지점이 가장 가까운 경계까지 거리 계산 / sampling(x,y,z) -> mm 단위로 계산

from totalsegmentator.python_api import totalsegmentator
from totalsegmentator.map_to_binary import class_map
from totalsegmentator.map_to_total import map_to_total


# -------------------------
# preview automatic switch
# -------------------------
IS_WINDOWS = sys.platform.startswith("win")
try:
    import fury  # noqa: F401
    HAS_FURY = True
except Exception:
    HAS_FURY = False
USE_PREVIEW = (HAS_FURY and (not IS_WINDOWS))
if not USE_PREVIEW:
    print("[INFO] TotalSegmentator preview will be deactivated in Windows.")


# -------------------------
# SeriesInstanceUID 기준으로 시리즈 선택/정렬 (폴더에 여러 시리즈 섞여 있어도 한 개만 선택)
# -------------------------
def scan_series_groups(dicom_dir: str) -> dict:
    groups = defaultdict(list)
    for name in os.listdir(dicom_dir):
        if name.startswith('.'): #리눅스 숨김파일 대비
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

##image patient position과 image orient position 


def select_series(dicom_dir: str, prefer_uid: str = None) -> Tuple[str, List[pydicom.dataset.FileDataset], np.ndarray]:
    groups = scan_series_groups(dicom_dir)
    if prefer_uid and prefer_uid in groups:
        uid = prefer_uid
    else:
        uid = max(groups.items(), key=lambda kv: len(kv[1]))[0]  # 슬라이스 수 최대
    slices, normal = sort_series_by_normal(groups[uid])
    print("[INFO] 선택 SeriesInstanceUID:", uid, "| 슬라이스 수:", len(slices))
    for k, v in groups.items():
        print(f"   - UID {k}: {len(v)} slices")
    return uid, slices, normal


# -------------------------
# DICOM → NIfTI 
# -------------------------
def dicom_slices_to_nifti(slices: List[pydicom.dataset.FileDataset], nifti_path: str, enforce_ras: bool = True):
    iop = np.array(slices[0].ImageOrientationPatient, dtype=float)
    row_cos, col_cos = iop[:3], iop[3:]
    normal = np.cross(row_cos, col_cos)

    ps = np.array(slices[0].PixelSpacing, dtype=float)
    row_spacing, col_spacing = float(ps[0]), float(ps[1])

    # slice spacing
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

    # HU 적용하여 스택
    vol = []
    for ds in slices:
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        vol.append(arr * slope + intercept)
    vol = np.stack(vol, axis=-1)  # (rows, cols, slices)

    # LPS affine
    M = np.eye(4, dtype=float)
    M[:3, 0] = col_cos * col_spacing
    M[:3, 1] = row_cos * row_spacing
    M[:3, 2] = normal  * slice_spacing
    M[:3, 3] = np.array(slices[0].ImagePositionPatient, dtype=float)

    # RAS 권장 (Slicer와 일치)
    if enforce_ras:
        lps_to_ras = np.diag([-1, -1, 1, 1])
        M = lps_to_ras @ M

    img = nib.Nifti1Image(vol, M)
    img.set_qform(M, code=1); img.set_sform(M, code=1)
    nib.save(img, nifti_path)

    axcodes = nib.aff2axcodes(img.affine)
    print(f"[INFO] NIfTI 저장: {nifti_path}  orientation={axcodes}, spacing≈({col_spacing:.3f},{row_spacing:.3f},{slice_spacing:.3f})")


# -------------------------
# Resampling seg.nii.gz to CT grid 
# -------------------------
def resample_mask_to_ct_grid(seg_nifti_path: str, series_files_sorted: List[str]) -> Tuple[np.ndarray, sitk.Image]:
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(series_files_sorted)  
    ct = reader.Execute()
    seg = sitk.ReadImage(seg_nifti_path)
    seg_on_ct = sitk.Resample(seg, ct, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    mask_slices = sitk.GetArrayFromImage(seg_on_ct)  # [z,y,x]
    return mask_slices.astype(np.uint8), ct


# -------------------------
# 라벨명 → 인덱스
# -------------------------
def label_name_to_index(task: str, name: str) -> int:
    canonical = map_to_total.get(name, name)
    idx2name = class_map[task]
    name2idx = {v: k for k, v in idx2name.items()}
    if canonical not in name2idx:
        raise KeyError(f"'{name}'(정규화 '{canonical}') 라벨이 task='{task}'에 없습니다.")
    return name2idx[canonical]


# -------------------------
# (선택) mid-sagittal x 추정 (좌/우 분리 필요할 때)
# -------------------------
def estimate_midsagittal_x_from_ct(ct_image: sitk.Image, hu_threshold: float = -300.0) -> int:
    arr = sitk.GetArrayFromImage(ct_image)  # [z,y,x]
    body = arr > hu_threshold
    if not np.any(body):
        mid_x = arr.shape[2] // 2
        print("[WARN] Body mask 비어 있음. x 중앙으로 대체:", mid_x)
        return mid_x
    xs = np.where(body)[2]
    mid_x = int(round(0.5 * (xs.min() + xs.max())))
    print(f"[INFO] mid-sagittal x index: {mid_x}  (x_min={xs.min()}, x_max={xs.max()})")
    return mid_x


# -------------------------
# Per-slice 마스크 DICOM 저장
# -------------------------
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

    new_series_uid = generate_uid()  # 파생 시리즈 UID 생성
    for i, (mask, src) in enumerate(zip(mask_slices, slices_sorted), start=1):
        ds = src.copy()
        ds.SOPInstanceUID = generate_uid()
        ds.SeriesInstanceUID = new_series_uid
        ds.SeriesDescription = series_description
        ds.ImageType = ['DERIVED', 'SECONDARY', 'BINARY']
        if set_modality_ot:
            ds.Modality = 'OT'  # 필요시 원래 Modality 유지도 가능

        # 픽셀 관련 속성(마스크용 8-bit)
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = 'MONOCHROME2'
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
        ds.SmallestImagePixelValue = 0
        ds.LargestImagePixelValue = 1

        # 마스크는 HU가 아니므로 Rescale 태그 제거
        if hasattr(ds, "RescaleSlope"): del ds.RescaleSlope
        if hasattr(ds, "RescaleIntercept"): del ds.RescaleIntercept
        if hasattr(ds, "WindowCenter"): del ds.WindowCenter
        if hasattr(ds, "WindowWidth"): del ds.WindowWidth

        # 픽셀 교체
        ds.Rows, ds.Columns = rows, cols
        ds.PixelData = mask.tobytes()

        out_path = os.path.join(out_dir, f"{i:04d}.dcm")
        ds.save_as(out_path)

    print(f"[INFO] Saved per-slice mask DICOM series: {out_dir}")


# -------------------------
# mask 내 cylinder 생성
# -------------------------
def cylinder_in_mask(
        binary_mask: np.ndarray,
        reference_nii: nib.Nifti1Image,
        out_path: str,  # 결과 저장 경로
        radius_mm: float = 10.0, # 실린더 반지름, 경계까지 거리보다 크면 자동 축소
        height_mm: float | None = None,  # 실린더 높이, None이면 z축 방향으로 확장
        margin_mm: float = 1.5 # 경계에 닿지 않게 margin 설정
    ) -> np.ndarray:
    
    affine = reference_nii.affine # voxel 인덱스 (i, j, k) <-> 실제 위치 (x, y, z) mm 변환하는 행렬
    header = reference_nii.header
    vx, vy, vz = reference_nii.header.get_zooms()[:3] # voxel 크기(mm/voxel), zz가 slice thickness / nii.header.get_zooms()랑 같음

    edt = distance_transform_edt(binary_mask, sampling = (vx, vy, vz)) # 마스크 안에서 각 voxel에서 가장 가까운 경계까지 거리(mm), 최소 거리
    # center_index = np.unravel_index(np.argmax(edt), edt.shape) # center = 마스크 경계에서 가장 멀리 떨어진 곳 -> 실린더 중심 index 번호, ex) [10, 30, 50]
    cx, cy, cz = map(int, np.unravel_index(np.argmax(edt), edt.shape)) # 10, 30, 50 

    # 반지름 결정
    radius = float(radius_mm)
    radius = min(radius, float(edt[cx, cy, cz]) - float(margin_mm)) # center에서 경계까지 거리보다 더 큰 반지름 사용 시 경계에 닿음 -> margin 포함해서 축소

    if radius <= 0:
        raise ValueError("실린더 넣을 공간 없음")
    
    # 높이 결정
    if height_mm is not None: # 높이 직접 입력 시
        height = float(height_mm)
    else:
        half_height_up = 0 # 머리쪽 방향
        z = cz # 실린더 중심 z 좌표
        while z + 1 < binary_mask.shape[2]:
            z += 1
            if (edt[cx, cy, z] < radius + margin_mm): # 경계까지의 거리가 반지름 + margin보다 작을 때까지
                break
            half_height_up += vz

        half_height_down = 0
        z = cz
        while z - 1 >= 0:
            z -= 1
            if (edt[cx, cy, z] < radius + margin_mm):
                break
            half_height_down += vz

        half_height_up = max(0, half_height_up - float(margin_mm)) # 경계에 닿지 않게 높이 margin도 포함
        half_height_down = max(0, half_height_down - float(margin_mm))
        height = max(vz, half_height_up + half_height_down) # Voxel 하나 사이즈 vs 확장 시킨 높이 비교

        # 실린더 생성 영역 제한, x,y,z축 Voxel 수 계산 (최대한 크게 잡기)
        box_x = int(np.ceil(radius / vx)) + 2
        box_y = int(np.ceil(radius / vy)) + 2
        box_z = int(np.ceil(height / vz)) + 2
        
        # binary_mask.shape : 배열 크기, (X, Y, Z) = (256, 256, 180)이면 shape[0] = 256 (X축 길이)
        x0, x1 = max(0, cx - box_x), min(binary_mask.shape[0], cx + box_x + 1) # center index에서 왼쪽, 오른쪽으로 반지름 voxel 수만큼 영역 잡기
        y0, y1 = max(0, cy - box_y), min(binary_mask.shape[1], cy + box_y + 1)
        z0, z1 = max(0, cz - box_z), min(binary_mask.shape[2], cz + box_z + 1)

        X, Y, Z = np.meshgrid(
            np.arange(x0, x1), # index x0, x0 + 1, x0 + 2, ...., x1 -1, x1 갖는 배열 생성
            np.arange(y0, y1),
            np.arange(z0, z1),
            indexing="ij"# index 순서 (x, y, z) 맞추기
        )

        dist_xy2 = ((X - cx) * vx) ** 2 + ((Y - cy) * vy) ** 2 # (X-cx)^2 + (Y-cy)^2 <= r^2
        dist_z = np.abs((Z - cz) * vz) # Z축 중심 기준으로 위아래 다 포함

        inside_cyl = ((dist_xy2 <= radius ** 2) & (dist_z <= height)) # 원기둥 형태 실린더 생성
        
        cyl = np.zeros_like(binary_mask, dtype=np.uint8) # binary_mask와 같은 크기의 0 배열 생성
        cyl[x0:x1, y0:y1, z0:z1] = inside_cyl.astype(np.uint8) # inside_cyl 배열값 모두 True/False에서 1/0로 변환
        
        cyl_in = (cyl & (binary_mask > 0).astype(np.uint8)).astype(np.uint8) # AND(교집합), 실제 마스크와 겹치는 부분만 포함하는 실린더 생성

        nib.save(nib.Nifti1Image(cyl_in, affine, header), out_path) # 파일 저장
        
        # center_mm = (affine @ np.array([cx, cy, cz, 1.0]))[:3] # 중심 voxel indexl를 mm로 변환

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
        fast=fast,          # 1.5mm
        ml=True,
        output_type="nifti",
        robust_crop=True,
        preview=USE_PREVIEW
    )
    seg_path = os.path.join(out_nifti_dir, "seg.nii.gz")
    nib.save(seg_img, seg_path)

    # 멀티라벨 → 각 ROI 바이너리
    seg_arr = np.rint(seg_img.get_fdata()).astype(np.uint16)

    # seg → CT 격자 재배치(선택 시리즈만) 후 per-slice DICOM 저장
    #    (좌/우 분할 필요 시 mid-sagittal 이용)
    mask_slices_dummy, ct_img = resample_mask_to_ct_grid(seg_path, series_files_sorted)  # shape 확인용
    expected_z, rows, cols = mask_slices_dummy.shape
    print("[INFO] 재배치 기준 CT 격자 shape(z,y,x):", (expected_z, rows, cols))

    # 좌/우 분할 기준 잡기.
    mid_x = estimate_midsagittal_x_from_ct(ct_img, hu_threshold=-300.0)

    if not roi_subset:
        roi_subset = ['liver']  # 예시

    for name in roi_subset:
        # 1) 라벨 인덱스
        try:
            idx = label_name_to_index(task, name)
        except Exception as e:
            print(f"[WARN] '{name}' 인덱스 해석 실패: {e}")
            continue

        # 바이너리 mask 볼륨으로 저장
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

        # CT grid로 재배치 → [z,y,x]
        mask_slices, _ = resample_mask_to_ct_grid(tmp_bin, series_files_sorted)
        
        '''
        # 좌/우 이름 포함된 모델에 대해서 반대편 데이터 0으로 만들기
        lname = name.lower()
        if "left" in lname:
            mask_slices[:, :, mid_x:] = 0
        elif "right" in lname:
            mask_slices[:, :, :mid_x] = 0
        '''

        # 슬라이스별 마스크 DICOM 저장
        save_mask_series_as_dicom(mask_slices, slices_sorted, out_dicom_dir, series_description=name)

    print("[DONE] TS → CT-grid align → per-slice mask DICOM export 완료")
