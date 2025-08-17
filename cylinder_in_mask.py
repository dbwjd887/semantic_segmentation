import numpy as np # 다차원 배열, 수치계산(거리, 인덱싱)
import nibabel as nib # NIfTI 입출력 / nib.load()로 읽고 nib.Nifti1Image(data, affine, header)로 생성 및 저장
from scipy.ndimage import distance_transform_edt # Binary mask에서 각 Voxel 지점이 가장 가까운 경계까지 거리 계산 / sampling(x,y,z) -> mm 단위로 계산

def cylinder_in_mask(
        mask_path: str, # mask 파일 경로
        out_path: str,  # 결과 저장 경로
        radius_mm: float = 10.0, # 실린더 반지름, 경계까지 거리보다 크면 자동 축소
        height_mm: float | None = None,  # 실린더 높이, None이면 z축 방향으로 확장
        margin_mm: float = 1.5 # 경계에 닿지 않게 margin 설정
    ) -> dict : # 실린더 중심, 반지름, 높이, Voxel 수 등
    
    nii = nib.load(mask_path) # mask NIfTI 파일 읽기, nii 객체
    binary_mask = (nii.get_fdata() > 0).astype(np.uint8) # mask 데이터를 읽은 뒤 0/1로 반환 (True:1, False:0), uint8로 저장
    affine = nii.affine # voxel 인덱스 (i, j, k) <-> 실제 위치 (x, y, z) mm 변환하는 행렬
    vx, vy, vz = nii.header.get_zooms()[:3] # voxel 크기(mm/voxel), zz가 slice thickness / nii.header.get_zooms()랑 같음

    edt = distance_transform_edt(binary_mask, sampling = (vx, vy, vz)) # 마스크 안에서 각 voxel에서 가장 가까운 경계까지 거리(mm), 최소 거리
    center_index = np.unravel_index(np.argmax(edt), edt.shape) # center가 최대인 voxel index = 마스크 경계에서 가장 멀리 떨어진 곳 -> 실린더 중심 index 번호, ex) [10, 30, 50]
    cx, cy, cz = map(int, center_index) # 10, 30, 50 

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
            if (edt[cx, cy, z] < radius + margin_mm): # 경계까지의 거리가 반지름+마진보다 작을 때까지
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
        pad_x = int(np.ceil(radius / vx)) + 2
        pad_y = int(np.ceil(radius / vy)) + 2
        pad_z = int(np.ceil(height / vz)) + 2
        
        # binary_mask.shape : 배열 크기, (X, Y, Z) = (256, 256, 180)이면 shape[0] = 256 (X축 길이)
        x0, x1 = max(0, cx - pad_x), min(binary_mask.shape[0], cx + pad_x + 1) # center index에서 왼쪽, 오른쪽으로 반지름 voxel 수만큼 영역 잡기
        y0, y1 = max(0, cy - pad_y), min(binary_mask.shape[1], cy + pad_y + 1)
        z0, z1 = max(0, cz - pad_z), min(binary_mask.shape[2], cz + pad_z + 1)

        X, Y, Z = np.meshgrid(
            np.arange(x0, x1), # index x0, x0 + 1, x0 + 2, ...., x1 -1, x1 갖는 배열 생성
            np.arange(y0, y1),
            np.arange(z0, z1),
            indexing="ij"# index 순서 (x, y, z) 맞추기
        )

        dist_xy2 = ((X - cx) * vx) ** 2 + ((Y - cy) * vy) ** 2 # (X-cx)^2 + (Y-cy)^2 <= r^2
        dist_z = np.abs((Z - cz) * vz) # Z축 중심 기준으로 위아래 다 포함

        inside_cyl = ((dist_xy2 <= radius_mm ** 2) & (dist_z <= height)) # 원기둥 형태 실린더 생성
        
        cyl = np.zeros_like(binary_mask, dtype=np.uint8) # binary_mask와 같은 크기의 0 배열 생성
        cyl[x0:x1, y0:y1, z0:z1] = inside_cyl.astype(np.uint8) # inside_cyl 배열값 모두 True/False에서 1/0로 변환
        cyl_in = cyl & binary_mask # AND(교집합), 실제 마스크와 겹치는 부분만 포함하는 실린더 생성

        nib.save(nib.Nifti1Image(cyl_in, affine, nii.header), out_path) # 저장
        
        center_mm = (affine @ np.array([cx, cy, cz, 1.0]))[:3] # 중심 voxel indexl를 mm로 변환
        
        return {
            "center_vox": [cx, cy, cz],
            "center_mm": center_mm.tolist(),
            "radius_mm": float(radius),
            "height_mm": float(height),
            "voxels": int(cyl_in.sum()),
            }
    
if __name__ == "__main__":
    
    info_left = cylinder_in_mask("/Users/kimyujung/Desktop/seg_nifti/kidney_left.nii.gz", "/Users/kimyujung/Desktop/seg_nifti/cylinder_kidney_left")
    info_right = cylinder_in_mask("/Users/kimyujung/Desktop/seg_nifti/kidney_right.nii.gz", "/Users/kimyujung/Desktop/seg_nifti/cylinder_kidney_right")


