# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 15:37:29 2025

@author: mingeon
"""

import os
import shutil
import pydicom
import numpy as np
import nibabel as nib
from totalsegmentator.libs import combine_masks_to_multilabel_file
from totalsegmentator.python_api import totalsegmentator
from scipy.ndimage import distance_transform_edt # Binary mask에서 각 Voxel 지점이 가장 가까운 경계까지 거리 계산 / sampling(x,y,z) -> mm 단위로 계산

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

def dicom_series_to_nifti(dicom_dir, nifti_path):
    # 1) .dcm 파일 수집 & 정렬
    files = [os.path.join(dicom_dir, f)
             for f in os.listdir(dicom_dir)
             if f.lower().endswith('.dcm')]
    if not files:
        raise FileNotFoundError(f"No DICOM files in {dicom_dir}")
    
    datasets = sorted((pydicom.dcmread(f) for f in files),
                      key=lambda ds: int(ds.InstanceNumber))

    # 2) 3D 볼륨 생성 (Z, Y, X)
    volume = np.stack([ds.pixel_array for ds in datasets], axis=0)

    # 3) spacing
    px, py = map(float, datasets[0].PixelSpacing)
    try:
        pz = float(datasets[0].SliceThickness)
    except AttributeError:
        z0 = float(datasets[0].ImagePositionPatient[2])
        z1 = float(datasets[1].ImagePositionPatient[2])
        pz = abs(z1 - z0)

    # 4) 배열 축 재배열 → (X, Y, Z)
    vol_xyz = volume.transpose(2, 1, 0)

    # 5) affine 생성
    affine = np.diag([px, py, pz, 1.0])

    # 6) NIfTI 저장
    nifti_img = nib.Nifti1Image(vol_xyz, affine)
    nib.save(nifti_img, nifti_path)
    print(f"[INFO] Saved NIfTI with correct XYZ ordering: {nifti_path}")

def save_mask_as_dicom(mask, orig_dir, out_dir, mask_description):
    """
    segmentation mask(np.uint8, values 0/1)를
    원본 DICOM 헤더 기반으로 각 slice 별 DICOM으로 저장
    """
    os.makedirs(out_dir, exist_ok=True)
    # 원본 DICOM 읽고 정렬
    files = [os.path.join(orig_dir, f)
             for f in os.listdir(orig_dir)
             if f.lower().endswith('.dcm')]
    datasets = [pydicom.dcmread(f) for f in files]
    datasets.sort(key=lambda ds: int(ds.InstanceNumber))
    # mask shape 확인
    if mask.shape != (len(datasets), datasets[0].Rows, datasets[0].Columns):
        raise ValueError("Mask and DICOM series shape mismatch")
    # 각 슬라이스별로 DICOM 생성
    for i, ds in enumerate(datasets):
        mask_slice = (mask[i] > 0).astype(np.uint8)  # 0/1
        # 복사본 생성
        ds_mask = ds.clone() if hasattr(ds, 'clone') else ds.copy()
        # 픽셀 정보 교체
        ds_mask.PixelData = mask_slice.tobytes()
        ds_mask.Rows, ds_mask.Columns = mask_slice.shape
        # 비트 설정
        ds_mask.BitsAllocated = 8
        ds_mask.BitsStored    = 8
        ds_mask.HighBit       = 7
        ds_mask.PixelRepresentation = 0
        ds_mask.PhotometricInterpretation = 'MONOCHROME2'
        # SeriesDescription 등 메타 바꿔도 좋음
        ds_mask.SeriesDescription = mask_description
        # 파일명: slice 번호 유지
        out_path = os.path.join(out_dir,
                                f"str(mask_description)_{ds.InstanceNumber:03d}.dcm")
        ds_mask.save_as(out_path)
    print(f"[INFO] Saved DICOM series: {out_dir}")
    
def cylinder_in_mask(
        binary_mask: np.ndarray,
        reference_nii: nib.Nifti1Image,
        out_path: str,  # 결과 저장 경로
        radius_mm: float = 10.0, # 실린더 반지름, 경계까지 거리보다 크면 자동 축소
        height_mm: float | None = None,  # 실린더 높이, None이면 z축 방향으로 확장
        margin_mm: float = 1.5 # 경계에 닿지 않게 margin 설정
    ) -> np.darray:
    
    # nii = nib.load(mask_path) # mask NIfTI 파일 읽기, nii 객체
    # binary_mask = (nii.get_fdata() > 0).astype(np.uint8) # mask 데이터를 읽은 뒤 0/1로 반환 (True:1, False:0), uint8로 저장
    # affine = nii.affine # voxel 인덱스 (i, j, k) <-> 실제 위치 (x, y, z) mm 변환하는 행렬
    # vx, vy, vz = nii.header.get_zooms()[:3] # voxel 크기(mm/voxel), zz가 slice thickness / nii.header.get_zooms()랑 같음

    affine = reference_nii.affine
    header = reference_nii.header
    vx, vy, vz = reference_nii.header.get_zooms()[:3]

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
    
        # return {
        #     "center_vox": [cx, cy, cz],
        #     "center_mm": center_mm.tolist(),
        #     "radius_mm": float(radius),
        #     "height_mm": float(height),
        #     "voxels": int(cyl_in.sum()),
        #     }
    

if __name__ == "__main__":
    # 사용자 데이터 위치 설정
    dicom_folder  = "C:\\Users\\admin\\Desktop\\PythonCode\\TotalSegmentator\\dicom"
    temp_nifti    = "C:\\Users\\admin\\Desktop\\PythonCode\\TotalSegmentator\\tmp\\input.nii.gz"
    out_nifti     = "C:\\Users\\admin\\Desktop\\PythonCode\\TotalSegmentator\\seg_nifti\\outmask"
    out_dicom_dir = "C:\\Users\\admin\\Desktop\\PythonCode\\TotalSegmentator\\seg_dicom\\DICOM_out"
    
    
    out_dir = os.path.dirname(out_nifti)
    roi_subset=['hip_left']
    #roi_subset=['hip_left', 'hip_right', 'femur_left', 'femur_right']

# 디렉토리가 없다면 생성
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # 1) DICOM → NIfTI (nnUnet 실행을 위해서..)
    dicom_series_to_nifti(dicom_folder, temp_nifti)
    
    
    

    # 2) hip_left segmentation (GPU 자동 사용, multi label 동작 안함.)
    seg_img = totalsegmentator(
        temp_nifti,
        out_nifti,
        task='total',
        roi_subset=roi_subset,
        roi_subset_robust = roi_subset,
        device='gpu',
        fast=False,
        ml = True,
        output_type="Nifti",
        #robust_crop=True
        
    )
    #roi_subset=['hip_left', 'hip_right', 'femur_left', 'femur_right']
    #task = hip_implant
    seg_data = seg_img.get_fdata().astype(np.uint8)  # (X, Y, Z), 0/1
    cylinder = cylinder_in_mask(seg_data, ) # out_nifti 폴더 내 사용할 segmentation 파일 지정
    
    mask_data = np.transpose(seg_data, (2, 1, 0)) 
    mask_description = roi_subset[0]

    os.makedirs(os.path.dirname(out_nifti), exist_ok=True)
    #nib.save(nib.Nifti1Image(mask_data, seg_img.affine), out_nifti)
    #print(f"[INFO] Saved segmentation NIfTI: {out_nifti}")

    # 5) DICOM 시리즈 저장
    os.makedirs(out_dicom_dir, exist_ok=True)
    save_mask_as_dicom(mask_data, dicom_folder, out_dicom_dir, mask_description)
    
    #roi_subset 별 구동 가능하게 loop 만들기


