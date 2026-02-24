import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import tkinter as tk 

# Configuration
DATA_DIR = "blender-render"
START_FRAME = 1
END_FRAME = 40
OUTPUT_VIDEO = "visualization.mp4"

X_LIMITS = [-1, 4]
Y_LIMITS = [-2, 2]
Z_LIMITS = [7, 12]

FRUSTUM_SCALE = 0.5
WINDOW_NAME = "Visualization"

def get_screen_center():
    """Gets the (x, y) coordinates to center the window."""
    try:
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()
        return screen_width, screen_height
    except:
        return 1920, 1080 

def load_camera_matrices(frame_idx):
    txt_files = sorted(glob.glob(os.path.join(DATA_DIR, f"*frame_{frame_idx:04d}_matrix.txt")))
    matrices = []
    for f in txt_files:
        P = np.loadtxt(f)
        matrices.append(P)
    return matrices

def get_camera_center(P):
    M = P[:, :3]
    p4 = P[:, 3]
    C = -np.linalg.inv(M) @ p4
    return C

def draw_camera_frustum(ax, P, color='cyan'):
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
        corners_3d.append(C + vec * FRUSTUM_SCALE)
        
    corners_3d = np.array(corners_3d)
    
    for i in range(4):
        ax.plot([C[0], corners_3d[i,0]], [C[1], corners_3d[i,1]], [C[2], corners_3d[i,2]], color=color, alpha=0.5, linewidth=1)
        
    verts = [list(zip(corners_3d[:,0], corners_3d[:,1], corners_3d[:,2]))]
    poly = Poly3DCollection(verts, alpha=0.1, facecolor=color)
    ax.add_collection3d(poly)
    
    return C

def get_track_point(frame_idx, cameras):
    images = []
    for i in range(len(cameras)):
        img_files = sorted(glob.glob(os.path.join(DATA_DIR, f"*frame_{frame_idx:04d}.png")))
        if i < len(img_files):
            images.append(cv2.imread(img_files[i], cv2.IMREAD_GRAYSCALE))
    
    if not images: return None
    
    x = np.linspace(X_LIMITS[0], X_LIMITS[1], 60)
    y = np.linspace(Y_LIMITS[0], Y_LIMITS[1], 60)
    z = np.linspace(Z_LIMITS[0], Z_LIMITS[1], 60)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    scores = np.zeros(points.shape[0])
    
    for i, img in enumerate(images):
        _, mask = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
        P = cameras[i]
        uv_hom = (P @ np.hstack([points, np.ones((points.shape[0], 1))]).T).T
        with np.errstate(divide='ignore', invalid='ignore'):
            u = uv_hom[:, 0] / uv_hom[:, 2]
            v = uv_hom[:, 1] / uv_hom[:, 2]
        h, w = mask.shape
        valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u_v = u[valid].astype(int)
        v_v = v[valid].astype(int)
        scores[valid] += (mask[v_v, u_v] > 0).astype(float)
        
    threshold = max(2, len(cameras) - 1)
    valid_indices = scores >= threshold
    if np.any(valid_indices):
        return np.mean(points[valid_indices], axis=0)
    return None

# Main render loop
if __name__ == "__main__":
    print("[INFO] Starting visualization...")
    
    screen_w, screen_h = get_screen_center()
    window_w, window_h = 800, 600
    center_x = (screen_w - window_w) // 2
    center_y = (screen_h - window_h) // 2

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(8, 6), facecolor='black') 
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black') 
    
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    history = []
    window_initialized = False

    for i in range(START_FRAME, END_FRAME + 1):
        ax.clear()
        
        # Style grid and axis
        ax.grid(True, color='dimgray', linestyle='--', linewidth=0.5)
        
        ax.tick_params(axis='x', colors='white', labelsize=8)
        ax.tick_params(axis='y', colors='white', labelsize=8)
        ax.tick_params(axis='z', colors='white', labelsize=8)
        
        ax.set_xlabel('X', color='white')
        ax.set_ylabel('Y', color='white')
        ax.set_zlabel('Z', color='white')
        
        ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))

        ax.set_xlim(X_LIMITS)
        ax.set_ylim(Y_LIMITS)
        ax.set_zlim(Z_LIMITS)

        cameras_P = load_camera_matrices(i)
        cam_centers = []
        
        for P in cameras_P:
            C = draw_camera_frustum(ax, P, color='cyan')
            cam_centers.append(C)
            
        target = get_track_point(i, cameras_P)
        
        # Draw 3D Elements
        if target is not None:
            history.append(target)
            ax.scatter(target[0], target[1], target[2], c='white', s=20, marker='o')
            ax.scatter(target[0], target[1], target[2], c='red', s=150, marker='o', alpha=0.3)
            
            for C in cam_centers:
                ax.plot([C[0], target[0]], [C[1], target[1]], [C[2], target[2]], c='yellow', alpha=0.2, linewidth=0.5)
        
        if len(history) > 1:
            h_arr = np.array(history)
            ax.plot(h_arr[:,0], h_arr[:,1], h_arr[:,2], c='lime', linewidth=1.5, alpha=0.9)

        ax.view_init(elev=15, azim=i * 0.8 + 45)
        
        # Render to image
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        
        # Coordinates
        if target is not None:
            coord_text = f"POS: [{target[0]:6.2f}, {target[1]:6.2f}, {target[2]:6.2f}] m"
            
            # Text Shadow
            cv2.putText(img, coord_text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            # Text Body
            cv2.putText(img, coord_text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 1)
        else:
            # Signal Lost
            cv2.putText(img, "SIGNAL LOST", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        if not window_initialized:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, window_w, window_h)
            cv2.moveWindow(WINDOW_NAME, center_x, center_y)
            window_initialized = True
        
        cv2.imshow(WINDOW_NAME, img)
        
        if out is None:
            h, w = img.shape[:2]
            out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 10.0, (w, h))
        out.write(img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27: break

    out.release()
    cv2.destroyAllWindows()
    print(f"[SUCCESS] Video saved as {OUTPUT_VIDEO}")