import numpy as np
import cv2

class OctreeCarver:
    """
    Prototypes the recursive Octree subdivision for 3D reconstruction.
    This replaces the O(N^3) brute-force voxel carving with a sparse search.
    """
    def __init__(self, x_limits, y_limits, z_limits, max_depth=6, min_consensus=3):
        self.x_limits = x_limits
        self.y_limits = y_limits
        self.z_limits = z_limits
        self.max_depth = max_depth
        self.min_consensus = min_consensus

    def _get_corners(self, x_range, y_range, z_range):
        """Returns the 8 corners of a voxel."""
        return np.array([
            [x_range[0], y_range[0], z_range[0], 1],
            [x_range[0], y_range[0], z_range[1], 1],
            [x_range[0], y_range[1], z_range[0], 1],
            [x_range[0], y_range[1], z_range[1], 1],
            [x_range[1], y_range[0], z_range[0], 1],
            [x_range[1], y_range[0], z_range[1], 1],
            [x_range[1], y_range[1], z_range[0], 1],
            [x_range[1], y_range[1], z_range[1], 1],
            # Also include center for better coverage in coarse levels
            [(x_range[0]+x_range[1])/2, (y_range[0]+y_range[1])/2, (z_range[0]+z_range[1])/2, 1]
        ]).T

    def _is_voxel_active(self, x_range, y_range, z_range, masks, cameras):
        """Checks if a voxel's projection overlaps with motion in masks."""
        corners = self._get_corners(x_range, y_range, z_range)
        
        consensus = 0
        for i, mask in enumerate(masks):
            P = cameras[i]
            # Project all 9 points at once
            uv_hom = P @ corners
            with np.errstate(divide='ignore', invalid='ignore'):
                u = (uv_hom[0, :] / uv_hom[2, :]).astype(int)
                v = (uv_hom[1, :] / uv_hom[2, :]).astype(int)
            
            h, w = mask.shape
            valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            
            if np.any(valid):
                # Check if any of the projected points hit a motion pixel
                if np.any(mask[v[valid], u[valid]] > 0):
                    consensus += 1
            
        return consensus >= self.min_consensus

    def _recursive_carve(self, x_range, y_range, z_range, depth, masks, cameras):
        """Recursively subdivides voxels that project to motion."""
        if not self._is_voxel_active(x_range, y_range, z_range, masks, cameras):
            return []

        if depth >= self.max_depth:
            # Return the center of the leaf voxel
            return [np.array([
                (x_range[0] + x_range[1]) / 2,
                (y_range[0] + y_range[1]) / 2,
                (z_range[0] + z_range[1]) / 2
            ])]

        # Subdivide into 8 octants
        x_mid = (x_range[0] + x_range[1]) / 2
        y_mid = (y_range[0] + y_range[1]) / 2
        z_mid = (z_range[0] + z_range[1]) / 2
        
        found_points = []
        for dx in [(x_range[0], x_mid), (x_mid, x_range[1])]:
            for dy in [(y_range[0], y_mid), (y_mid, y_range[1])]:
                for dz in [(z_range[0], z_mid), (z_mid, z_range[1])]:
                    found_points.extend(
                        self._recursive_carve(dx, dy, dz, depth + 1, masks, cameras)
                    )
        
        return found_points

    def solve(self, masks, cameras):
        """Entry point for the octree carver."""
        points = self._recursive_carve(
            self.x_limits, self.y_limits, self.z_limits, 0, masks, cameras
        )
        if not points:
            return None
        # Return the centroid of all detected leaf voxels
        return np.mean(points, axis=0)

if __name__ == "__main__":
    # Example usage / mock test
    print("[INFO] Octree Carver Prototype initialized.")
    # This would be integrated into the main loop similar to get_track_point
