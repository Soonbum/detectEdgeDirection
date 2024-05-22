import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 엣지 검출
def detect_edges(image_path):
    # 이미지 읽기
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("이미지를 읽을 수 없습니다.")

    # 가우시안 블러 적용 (노이즈 제거)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Canny 엣지 검출
    edges = cv2.Canny(blurred, 50, 150)

    return img, edges

# 2. 엣지 방향 벡터 추출
def extract_edge_directions(edges):
    # Sobel 필터를 사용하여 그래디언트 계산
    grad_x = cv2.Sobel(edges, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(edges, cv2.CV_64F, 0, 1, ksize=3)

    # 그래디언트의 방향 계산
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    direction = np.arctan2(grad_y, grad_x)

    # 방향을 단위 벡터로 변환
    direction_x = np.cos(direction)
    direction_y = np.sin(direction)

    return direction_x, direction_y, magnitude, grad_x, grad_y

# 추가 정보를 출력하는 함수
def print_boundary_intersections(edges):
    height, width = edges.shape

    # 각 변의 교차점을 찾기 위한 마스크 생성
    left_boundary = np.zeros_like(edges)
    right_boundary = np.zeros_like(edges)
    top_boundary = np.zeros_like(edges)
    bottom_boundary = np.zeros_like(edges)

    # 좌측 교차점
    left_boundary[:, 0] = 1
    # 우측 교차점
    right_boundary[:, -1] = 1
    # 상단 교차점
    top_boundary[0, :] = 1
    # 하단 교차점
    bottom_boundary[-1, :] = 1

    # 각 변에 맞닿는 엣지의 개수
    left_count = np.sum(edges & left_boundary)
    right_count = np.sum(edges & right_boundary)
    top_count = np.sum(edges & top_boundary)
    bottom_count = np.sum(edges & bottom_boundary)

    print("좌측에 맞닿는 점의 수:", left_count)
    print("우측에 맞닿는 점의 수:", right_count)
    print("상단에 맞닿는 점의 수:", top_count)
    print("하단에 맞닿는 점의 수:", bottom_count)

# 이미지 경로
image_path = 'test.png'

# 엣지 검출
img, edges = detect_edges(image_path)

# 엣지 방향 벡터 추출
direction_x, direction_y, magnitude, grad_x, grad_y = extract_edge_directions(edges)

# 그래프 범위에 맞닿는 점들의 정보 출력
print_boundary_intersections(edges)

# 결과 시각화
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(img, cmap='gray')

plt.subplot(1, 3, 2)
plt.title('Edge Detection')
plt.imshow(edges, cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Edge Directions')
plt.quiver(direction_x, direction_y, magnitude)
plt.gca().invert_yaxis()  # 이미지의 y축을 반전하여 올바른 방향으로 표시

plt.show()
