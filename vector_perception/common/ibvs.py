import numpy as np

class PersonDistanceEstimator:
    def __init__(self, K, camera_pitch, camera_height):
        """
        Initialize the distance estimator using ground plane constraint.
        
        Args:
            K: 3x3 Camera intrinsic matrix in OpenCV format
               (Assumed to be already for an undistorted image)
            camera_pitch: Upward pitch of the camera (in radians)
                         Positive means looking up, negative means looking down
            camera_height: Height of the camera above the ground (in meters)
        """
        self.K = K
        self.camera_height = camera_height
        
        # Precompute the inverse intrinsic matrix
        self.K_inv = np.linalg.inv(K)
        
        # Transform from camera to robot frame (z-forward to x-forward)
        self.T = np.array([[0, 0, 1],
                          [-1, 0, 0],
                          [0, -1, 0]])
        
        # Pitch rotation matrix (positive is upward)
        theta = -camera_pitch  # Negative since positive pitch is negative rotation about robot Y
        self.R_pitch = np.array([
            [ np.cos(theta), 0, np.sin(theta)],
            [            0, 1,            0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
        
        # Combined transform from camera to robot frame
        self.A = self.R_pitch @ self.T
        
        # Store focal length and principal point for angle calculation
        self.fx = K[0, 0]
        self.cx = K[0, 2]

    def estimate_distance_angle(self, bbox):
        """
        Estimate distance and angle to person using ground plane constraint.
        
        Args:
            bbox: tuple (x_min, y_min, x_max, y_max)
                 where y_max represents the feet position
        
        Returns:
            depth: distance to person along camera's z-axis (meters)
            angle: horizontal angle in camera frame (radians, positive right)
        """
        x_min, _, x_max, y_max = bbox
        
        # Get center point of feet
        u_c = (x_min + x_max) / 2.0
        v_feet = y_max
        
        # Create homogeneous feet point and get ray direction
        p_feet = np.array([u_c, v_feet, 1.0])
        d_feet_cam = self.K_inv @ p_feet
        
        # Convert ray to robot frame
        d_feet_robot = self.A @ d_feet_cam
        
        # Ground plane intersection (z=0)
        # camera_height + t * d_feet_robot[2] = 0
        if abs(d_feet_robot[2]) < 1e-6:
            raise ValueError("Feet ray is parallel to ground plane")
            
        # Solve for scaling factor t
        t = -self.camera_height / d_feet_robot[2]
        
        # Get 3D feet position in robot frame
        p_feet_robot = t * d_feet_robot
        
        # Convert back to camera frame
        p_feet_cam = self.A.T @ p_feet_robot
        
        # Extract depth (z-coordinate in camera frame)
        depth = p_feet_cam[2]
        
        # Calculate horizontal angle from image center
        angle = np.arctan((u_c - self.cx) / self.fx)
        
        return depth, angle


# Example usage:
if __name__ == "__main__":
    # Example camera calibration
    K = np.array([[600,   0, 320],
                  [  0, 600, 240],
                  [  0,   0,   1]], dtype=np.float32)
    
    # Camera mounted 1.2m high, pitched down 10 degrees
    camera_pitch = np.deg2rad(-10)  # negative for downward pitch
    camera_height = 1.2  # meters
    
    estimator = PersonDistanceEstimator(K, camera_pitch, camera_height)
    
    # Example detection
    bbox = (300, 100, 380, 400)  # x1, y1, x2, y2
    
    depth, angle = estimator.estimate_distance_angle(bbox)
    print(f"Estimated depth: {depth:.2f} m")
    print(f"Estimated angle: {np.rad2deg(angle):.1f}°")