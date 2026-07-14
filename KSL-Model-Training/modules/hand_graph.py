"""MediaPipe 손 랜드마크(21점) 골격 그래프 정의.

ST-GCN(Yan et al., 2018, https://arxiv.org/abs/1801.07455)의
"spatial configuration partitioning" 방식을 따른다: 각 관절의 이웃을
중심(손목)으로부터의 거리 기준으로 3개 부분집합으로 나눠 각각 별도의
학습 가능한 가중치를 부여한다.
    subset 0: 자기 자신 + 손목으로부터 같은 거리인 이웃 (예: 손가락 시작 관절끼리의 손바닥 연결)
    subset 1: 손목에 더 가까운 이웃 (centripetal)
    subset 2: 손목에서 더 먼 이웃 (centrifugal)
"""
from collections import deque

import numpy as np

NUM_HAND_JOINTS = 21

# MediaPipe HAND_CONNECTIONS과 동일한 뼈대 연결 (mediapipe.solutions.hands.HAND_CONNECTIONS 기준)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # 엄지
    (0, 5), (5, 6), (6, 7), (7, 8),        # 검지
    (0, 9), (9, 10), (10, 11), (11, 12),   # 중지
    (0, 13), (13, 14), (14, 15), (15, 16), # 약지
    (0, 17), (17, 18), (18, 19), (19, 20), # 소지
    (5, 9), (9, 13), (13, 17),             # 손바닥 (손가락 시작 관절끼리)
]


def _bfs_distances(num_nodes: int, edges: list[tuple[int, int]], center: int) -> np.ndarray:
    adjacency_list = [[] for _ in range(num_nodes)]
    for i, j in edges:
        adjacency_list[i].append(j)
        adjacency_list[j].append(i)

    distance = np.full(num_nodes, -1, dtype=np.int32)
    distance[center] = 0
    queue = deque([center])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency_list[node]:
            if distance[neighbor] == -1:
                distance[neighbor] = distance[node] + 1
                queue.append(neighbor)
    return distance


def _normalize(adjacency: np.ndarray) -> np.ndarray:
    """Λ^-1/2 A Λ^-1/2 (Kipf & Welling 정규화). 고립 노드(차수 0)는 나눗셈에서 제외."""
    degree = adjacency.sum(axis=1)
    inv_sqrt_degree = np.zeros_like(degree)
    nonzero = degree > 0
    inv_sqrt_degree[nonzero] = degree[nonzero] ** -0.5
    D = np.diag(inv_sqrt_degree)
    return D @ adjacency @ D


def build_spatial_partitions(
    num_nodes: int = NUM_HAND_JOINTS,
    edges: list[tuple[int, int]] = HAND_CONNECTIONS,
    center: int = 0,
) -> list[np.ndarray]:
    """ST-GCN 3-분할 정규화 인접행렬 [subset0(self+동일거리), subset1(centripetal), subset2(centrifugal)] 반환."""
    distance = _bfs_distances(num_nodes, edges, center)

    subset0 = np.eye(num_nodes, dtype=np.float32)  # 자기 자신
    subset1 = np.zeros((num_nodes, num_nodes), dtype=np.float32)  # 중심에 더 가까운 이웃
    subset2 = np.zeros((num_nodes, num_nodes), dtype=np.float32)  # 중심에서 더 먼 이웃

    for i, j in edges:
        if distance[j] == distance[i]:
            subset0[i, j] = 1.0
            subset0[j, i] = 1.0
        elif distance[j] < distance[i]:
            subset1[i, j] = 1.0
            subset2[j, i] = 1.0
        else:
            subset2[i, j] = 1.0
            subset1[j, i] = 1.0

    return [_normalize(subset0), _normalize(subset1), _normalize(subset2)]
