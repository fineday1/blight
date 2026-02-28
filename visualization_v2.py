import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time
from collections import deque

from radar_engine import KalmanFilter, OctreeCarver, cluster_points

X_LIMITS = [-1, 10]
Y_LIMITS = [-2, 10]
Z_LIMITS = [0, 15]

def load_camera_matrices(data_dir, frame_idx):
    txt_files = sorted(glob.glob(os.path.join(data_dir, f"*frame_{frame_idx:04d}_matrix.txt")))
    matrices = []
    for f in txt_files:
        try:
            P = np.loadtxt(f)
            matrices.append(P)
        except: pass
    return matrices

def get_camera_center(P):
    M = P[:, :3]
    p4 = P[:, 3]
    try:
        C = -np.linalg.inv(M) @ p4
    except:
        C = np.zeros(3)
    return C

def draw_camera_frustum(ax, P, color='cyan', scale=0.5):
    C = get_camera_center(P)
    P_inv = np.linalg.pinv(P)
    w, h = 1000, 1000 
    corners_2d = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]]).T
    corners_3d = []
    for i in range(4):
        X_hom = P_inv @ corners_2d[:, i]
        X = X_hom[:3] / X_hom[3]
        vec = X - C
        vec = vec / np.linalg.norm(vec)
        corners_3d.append(C + vec * scale)
    corners_3d = np.array(corners_3d)
    for i in range(4):
        ax.plot([C[0], corners_3d[i,0]], [C[1], corners_3d[i,1]], [C[2], corners_3d[i,2]], color=color, alpha=0.3, linewidth=1)
    verts = [list(zip(corners_3d[:,0], corners_3d[:,1], corners_3d[:,2]))]
    poly = Poly3DCollection(verts, alpha=0.1, facecolor=color)
    ax.add_collection3d(poly)
    return C

def process_frame(data_dir, frame_idx, cameras, img_buffer, carver, threshold_val=10):
    img_files = sorted(glob.glob(os.path.join(data_dir, f"*frame_{frame_idx:04d}.png")))
    curr_images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in img_files if os.path.exists(f)]
    if not curr_images: return None, None, None

    img_buffer.append(curr_images)
    if len(img_buffer) < 2: return None, None, None

    # Compare current against the oldest in buffer
    prev_images = img_buffer[0]
    masks = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for i, img in enumerate(curr_images):
        diff = cv2.absdiff(img, prev_images[i])
        max_d = np.max(diff)
        _, mask = cv2.threshold(diff, threshold_val, 255, cv2.THRESH_BINARY)
        mask = cv2.dilate(mask, kernel, iterations=2)
        masks.append(mask)
        if frame_idx % 10 == 0:
            print(f"[DEBUG] Frame {frame_idx} Cam {i} MaxDiff: {max_d} MaskNZ: {np.count_nonzero(mask)}")

    occupied_voxels = carver.solve(masks, cameras)
    targets = cluster_points(occupied_voxels, eps=1.2)
    return targets, curr_images, masks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", type=str, default="render1")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--thresh", type=int, default=5)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--consensus", type=int, default=2)
    parser.add_argument("--lookback", type=int, default=2, help="Frames to look back for diff")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    data_dir = os.path.join("blender-render", args.render)
    if not os.path.exists(data_dir): return

    gt_path = os.path.join(data_dir, "ground_truth.csv")
    gt_df = pd.read_csv(gt_path) if os.path.exists(gt_path) else None
    
    global X_LIMITS, Y_LIMITS, Z_LIMITS
    if gt_df is not None:
        X_LIMITS = [float(gt_df['x'].min() - 4), float(gt_df['x'].max() + 4)]
        Y_LIMITS = [float(gt_df['y'].min() - 4), float(gt_df['y'].max() + 4)]
        Z_LIMITS = [float(gt_df['z'].min() - 4), float(gt_df['z'].max() + 4)]


    carver = OctreeCarver(X_LIMITS, Y_LIMITS, Z_LIMITS, max_depth=args.depth, min_consensus=args.consensus)
    trackers, history = {}, {}
    img_buffer = deque(maxlen=args.lookback)
    
    if not args.headless:
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(10, 8), facecolor='black') 
        ax = fig.add_subplot(111, projection='3d')
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    all_pngs = glob.glob(os.path.join(data_dir, "*.png"))
    if not all_pngs: return
    frame_nums = [int(f.split("_frame_")[-1].split(".")[0]) for f in all_pngs if "_frame_" in f]
    start_f, end_f = min(frame_nums), max(frame_nums)
    
    errors = []
    total_detections = 0
    
    for frame_idx in range(start_f, end_f + 1):
        t_start = time.time()
        cameras = load_camera_matrices(data_dir, frame_idx)
        if not cameras: continue
        
        if frame_idx == start_f:
            for idx, P in enumerate(cameras):
                C = get_camera_center(P)
                print(f"[DEBUG] Cam {idx} Center: {C}")
        
        targets, _, masks = process_frame(data_dir, frame_idx, cameras, img_buffer, carver, threshold_val=args.thresh)
        current_gt = gt_df[gt_df['frame_idx'] == frame_idx] if gt_df is not None else None
        
        if current_gt is not None and frame_idx % 10 == 0:
            gt_p = current_gt[['x', 'y', 'z']].values[0]
            gt_hom = np.array([gt_p[0], gt_p[1], gt_p[2], 1])
            for idx, P in enumerate(cameras):
                uv_h = P @ gt_hom
                u, v = int(uv_h[0]/uv_h[2]), int(uv_h[1]/uv_h[2])
                hit = False
                if 0 <= u < 1000 and 0 <= v < 1000:
                    hit = masks[idx][v, u] > 0 if masks is not None else False
                print(f"[DEBUG] Frame {frame_idx} Cam {idx} GT: ({u},{v}) HitMask: {hit}")
        
        if targets:
            total_detections += len(targets)
            for t_pos in targets:
                best_id, min_dist = None, 5.0
                for tid, kf in trackers.items():
                    pred = kf.predict()
                    dist = np.linalg.norm(t_pos - pred)
                    if dist < min_dist: min_dist, best_id = dist, tid
                
                if best_id is None:
                    best_id = f"T{len(trackers)+1}"
                    trackers[best_id], history[best_id] = KalmanFilter(), []
                
                trackers[best_id].update(t_pos)
                history[best_id].append(t_pos)
                
        if current_gt is not None and targets:
            gt_points = current_gt[['x', 'y', 'z']].values
            for gp in gt_points:
                dist = np.linalg.norm(targets - gp, axis=1)
                errors.append(np.min(dist))

        if not args.headless:
            ax.clear(); ax.set_facecolor('black')
            ax.set_xlim(X_LIMITS); ax.set_ylim(Y_LIMITS); ax.set_zlim(Z_LIMITS)
            for P in cameras: draw_camera_frustum(ax, P)
            for tid, pts in history.items():
                if len(pts) > 0:
                    curr = pts[-1]
                    ax.scatter(curr[0], curr[1], curr[2], s=100, alpha=0.5)
                    ax.text(curr[0], curr[1], curr[2], tid, color='white')
                if len(pts) > 1:
                    h_arr = np.array(pts)
                    ax.plot(h_arr[:,0], h_arr[:,1], h_arr[:,2], linewidth=2)
            t_end = time.time()
            fps = 1.0 / (t_end - t_start) if (t_end - t_start) > 0 else 0
            fig.canvas.draw()
            img = cv2.cvtColor(np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (4,)), cv2.COLOR_RGBA2BGR)
            rmse = np.sqrt(np.mean(np.square(errors))) if errors else 0
            cv2.putText(img, f"SCENE: {args.render} | RMSE: {rmse:6.3f}m | DETS: {total_detections}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow("Radar HUD", img)
            if out is None: out = cv2.VideoWriter(f"output_{args.render}.mp4", fourcc, args.fps, (img.shape[1], img.shape[0]))
            out.write(img)
            if cv2.waitKey(1) & 0xFF == 27: break

    if out: out.release()
    if not args.headless: cv2.destroyAllWindows()
    final_rmse = np.sqrt(np.mean(np.square(errors))) if errors else 999.0
    print(f"BENCHMARK_RESULT: {final_rmse:.4f}")

if __name__ == "__main__":
    main()
