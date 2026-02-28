import numpy as np
import cv2
from sklearn.cluster import DBSCAN

class KalmanFilter:
    def __init__(self, dt=0.1, q=1e-2, r=1e-1):
        self.dt = dt
        self.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        self.Q = np.eye(6) * q
        self.R = np.eye(3) * r
        self.P = np.eye(6)
        self.x = None

    def predict(self):
        if self.x is None: return None
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:3]

    def update(self, z):
        if self.x is None:
            self.x = np.zeros(6); self.x[:3] = z; return
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

try:
    from radar_engine_cpp import OctreeCarver
    print("[INFO] Using high-performance C++ OctreeCarver")
except ImportError as e:
    print(f"[WARNING] C++ OctreeCarver not found, falling back to Python. Error: {e}")
    class OctreeCarver:
        def __init__(self, x_limits, y_limits, z_limits, max_depth=5, min_consensus=2):
            self.x_limits = [float(x_limits[0]), float(x_limits[1])]
            self.y_limits = [float(y_limits[0]), float(y_limits[1])]
            self.z_limits = [float(z_limits[0]), float(z_limits[1])]
            self.max_depth = max_depth
            self.min_consensus = min_consensus

        def _get_samples(self, x_range, y_range, z_range):
            xs = np.linspace(x_range[0], x_range[1], 3)
            ys = np.linspace(y_range[0], y_range[1], 3)
            zs = np.linspace(z_range[0], z_range[1], 3)
            X, Y, Z = np.meshgrid(xs, ys, zs)
            pts = np.vstack([X.ravel(), Y.ravel(), Z.ravel(), np.ones(27)])
            return pts

        def _is_voxel_active(self, x_range, y_range, z_range, masks, cameras, depth):
            samples = self._get_samples(x_range, y_range, z_range)
            consensus = 0
            for i, mask in enumerate(masks):
                P = cameras[i]
                uv_hom = P @ samples
                u = (uv_hom[0, :] / uv_hom[2, :]).astype(int)
                v = (uv_hom[1, :] / uv_hom[2, :]).astype(int)
                h, w = mask.shape
                valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
                if np.any(valid):
                    if np.any(mask[v[valid], u[valid]] > 0):
                        consensus += 1
            
            target_consensus = self.min_consensus if depth > 0 else 1
            return consensus >= target_consensus

        def _recursive_carve(self, x_range, y_range, z_range, depth, masks, cameras):
            if not self._is_voxel_active(x_range, y_range, z_range, masks, cameras, depth): return []
            if depth >= self.max_depth:
                return [np.array([(x_range[0]+x_range[1])/2, (y_range[0]+y_range[1])/2, (z_range[0]+z_range[1])/2])]
            x_m, y_m, z_m = (x_range[0]+x_range[1])/2, (y_range[0]+y_range[1])/2, (z_range[0]+z_range[1])/2
            res = []
            for dx in [(x_range[0], x_m), (x_m, x_range[1])]:
                for dy in [(y_range[0], y_m), (y_m, y_range[1])]:
                    for dz in [(z_range[0], z_m), (z_m, z_range[1])]:
                        res.extend(self._recursive_carve(dx, dy, dz, depth+1, masks, cameras))
            return res

        def solve(self, masks, cameras):
            points = self._recursive_carve(self.x_limits, self.y_limits, self.z_limits, 0, masks, cameras)
            return np.array(points) if points else np.empty((0,3))

def cluster_points(points, eps=0.5, min_samples=2):
    if len(points) == 0: return []
    if len(points) < 5: return [np.mean(points, axis=0)]
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    clusters = []
    for l in set(clustering.labels_):
        if l == -1: continue
        clusters.append(np.mean(points[clustering.labels_ == l], axis=0))
    return clusters if clusters else [np.mean(points, axis=0)]
