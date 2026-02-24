import numpy as np
import cv2
import glob
import os
import shutil
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Configuration
DATA_DIR = "blender-render"
START_FRAME = 1
END_FRAME = 40

# Region of interest
X_LIMITS = [-1, 4]   
Y_LIMITS = [-2, 2]   
Z_LIMITS = [7, 12]   

VOXEL_GRID_SIZE = 80 
DEBUG_MASKS = False # Used for debugging only

def load_all_cameras(start, end):
    print(f"[INFO] Loading frames {start} to {end}...")
    data = {}
    for frame_idx in range(start, end + 1):
        img_files = sorted(glob.glob(os.path.join(DATA_DIR, f"*frame_{frame_idx:04d}.png")))
        frame_cams = []
        for img_path in img_files:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            txt_path = img_path.replace(".png", "_matrix.txt")
            if not os.path.exists(txt_path): continue
            P = np.loadtxt(txt_path)
            frame_cams.append({"img": img, "P": P, "name": os.path.basename(img_path)})
        data[frame_idx] = frame_cams
    print(f"[INFO] Loaded {len(data)} frames.")
    return data

def process_motion_masks(frames_data):
    print("[INFO] Processing Motion Masks (Frame Diff Method)...")
    processed_data = {}
    
    # Create debug folder
    if DEBUG_MASKS:
        debug_dir = "debug_masks"
        if os.path.exists(debug_dir): shutil.rmtree(debug_dir)
        os.makedirs(debug_dir)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) # Thicker kernel
    
    num_cams = len(frames_data[START_FRAME])
    
    # Initialize "Previous frame" buffer with the first frame
    prev_frames = [frames_data[START_FRAME][i]['img'] for i in range(num_cams)]

    for frame_idx in range(START_FRAME, END_FRAME + 1):
        processed_data[frame_idx] = []
        
        # Determine which frame to compare against
        # Compare current against previous to find any pixel change
        if frame_idx == START_FRAME:
            pass 
        
        for cam_idx in range(num_cams):
            if cam_idx >= len(frames_data[frame_idx]): continue

            cam_data = frames_data[frame_idx][cam_idx]
            curr_img = cam_data['img']
            P = cam_data['P']
            prev_img = prev_frames[cam_idx]
            
            # Find absolute difference between this frame and the last one
            diff = cv2.absdiff(curr_img, prev_img)
            
            # Any pixel changing by more than 5 (out of 255) is motion
            _, mask = cv2.threshold(diff, 5, 255, cv2.THRESH_BINARY)
            
            # Inflate the dots to form a solid blob
            mask = cv2.dilate(mask, kernel, iterations=4)
            
            # Save debug image
            if DEBUG_MASKS and cam_idx == 0:
                cv2.imwrite(f"debug_masks/mask_frame_{frame_idx:04d}.png", mask)
            
            processed_data[frame_idx].append({"img": mask, "P": P})
            
            # Update history buffer
            prev_frames[cam_idx] = curr_img

    print(f"[INFO] Masks generated. Check 'debug_masks' folder if tracking fails.")
    return processed_data

def reconstruct_and_track(processed_data):
    trajectory = []
    
    # Pre-calculate grid
    x = np.linspace(X_LIMITS[0], X_LIMITS[1], VOXEL_GRID_SIZE)
    y = np.linspace(Y_LIMITS[0], Y_LIMITS[1], VOXEL_GRID_SIZE)
    z = np.linspace(Z_LIMITS[0], Z_LIMITS[1], VOXEL_GRID_SIZE)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    voxel_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    voxel_points_hom = np.hstack([voxel_points, np.ones((voxel_points.shape[0], 1))])
    
    print(f"[INFO] Tracking Target in ROI: X{X_LIMITS} Y{Y_LIMITS} Z{Z_LIMITS}...")
    
    # Start loop
    for frame_idx in range(START_FRAME + 1, END_FRAME + 1):
        if frame_idx not in processed_data: continue

        cameras = processed_data[frame_idx]
        voxel_scores = np.zeros(voxel_points.shape[0])
        
        for cam in cameras:
            mask = cam['img']
            P = cam['P']
            
            uv_hom = (P @ voxel_points_hom.T).T
            with np.errstate(divide='ignore', invalid='ignore'):
                u = uv_hom[:, 0] / uv_hom[:, 2]
                v = uv_hom[:, 1] / uv_hom[:, 2]
            
            h, w = mask.shape
            valid_mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            u_valid = u[valid_mask].astype(int)
            v_valid = v[valid_mask].astype(int)
            voxel_scores[valid_mask] += (mask[v_valid, u_valid] > 0).astype(float)
            
        # Relaxed voting
        threshold = max(2, len(cameras) - 1)
        
        occupied_indices = voxel_scores >= threshold
        points = voxel_points[occupied_indices]
        
        if len(points) > 0:
            center = np.mean(points, axis=0)
            trajectory.append(center)
            print(f"Frame {frame_idx}: Target at {center.round(3)}")
        else:
            print(f"Frame {frame_idx}: Lost Target.")
            
    return np.array(trajectory)

# Main
if __name__ == "__main__":
    raw_data = load_all_cameras(START_FRAME, END_FRAME)
    smart_data = process_motion_masks(raw_data)
    path = reconstruct_and_track(smart_data)
    
    if len(path) > 0:
        print(f"[SUCCESS] Tracked {len(path)} points.")
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(path[:,0], path[:,1], path[:,2], c='b', linewidth=2, label='Flight Path')
        ax.scatter(path[0,0], path[0,1], path[0,2], c='g', s=50, label='Start')
        ax.scatter(path[-1,0], path[-1,1], path[-1,2], c='r', s=50, label='End')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(X_LIMITS)
        ax.set_ylim(Y_LIMITS)
        ax.set_zlim(Z_LIMITS)
        ax.legend()
        plt.savefig("trajectory.png")
        print("Saved trajectory.png")
    else:
        print("[FAIL] No trajectory found.")