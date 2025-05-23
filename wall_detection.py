#!/usr/bin/env python3

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import argparse
import os
from typing import List, Tuple, Dict
import json
from scipy.spatial import ConvexHull

class WallDetector:
    """
    A class for detecting vertical walls in 3D point clouds using geometric analysis.
    """
    
    def __init__(self):
        """
        Initialize the wall detector with fixed parameters.
        """
        # RANSAC parameters
        self.distance_threshold = 0.02
        self.ransac_n = 3
        self.num_iterations = 1000
        
        # Wall detection criteria
        self.min_wall_height = 1.0
        self.max_wall_angle = np.radians(15.0)  # 15 degrees from vertical
        self.min_points_per_wall = 100
        self.min_wall_width = 0.3  # Minimum width for a wall (meters)
        self.min_wall_area = 0.5   # Minimum area for a wall (square meters)
        
        # Storage for detected walls and results
        self.walls = []
        self.wall_point_clouds = []
        self.original_cloud = None
        self.processed_cloud = None
        
    def load_point_cloud(self, file_path: str) -> o3d.geometry.PointCloud:
        """
        Load a point cloud from file.
        
        Args:
            file_path: Path to the .pcd file
            
        Returns:
            Open3D point cloud object
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Point cloud file not found: {file_path}")
            
        print(f"Loading point cloud from: {file_path}")
        cloud = o3d.io.read_point_cloud(file_path)
        
        if len(cloud.points) == 0:
            raise ValueError("Empty point cloud loaded")
            
        print(f"Loaded {len(cloud.points)} points")
        self.original_cloud = cloud
        return cloud
    
    def preprocess_point_cloud(self, cloud: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Preprocess the point cloud: remove outliers, downsample if needed.
        
        Args:
            cloud: Input point cloud
            
        Returns:
            Preprocessed point cloud
        """
        print("Preprocessing point cloud...")
        
        # Remove statistical outliers
        cloud_filtered, _ = cloud.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=2.0)
        
        # Optional: Downsample if point cloud is very dense
        if len(cloud_filtered.points) > 50000:
            print("Downsampling dense point cloud...")
            cloud_filtered = cloud_filtered.voxel_down_sample(voxel_size=0.01)
            
        print(f"After preprocessing: {len(cloud_filtered.points)} points")
        self.processed_cloud = cloud_filtered
        return cloud_filtered
    
    def is_vertical_plane(self, plane_model: np.ndarray) -> bool:
        """
        Check if a plane is approximately vertical (i.e., could be a wall).
        
        Args:
            plane_model: Plane equation coefficients [a, b, c, d] where ax + by + cz + d = 0
            
        Returns:
            True if the plane is vertical within the specified tolerance
        """
        # Extract normal vector (a, b, c)
        normal = plane_model[:3]
        normal = normal / np.linalg.norm(normal)  # Normalize
        
        # Vertical direction (up vector)
        up_vector = np.array([0, 0, 1])
        
        # Calculate angle between normal and horizontal plane (xy-plane)
        # A vertical wall should have a normal that's perpendicular to the up vector
        angle_with_up = np.arccos(np.abs(np.dot(normal, up_vector)))
        
        # Check if the angle is close to 90 degrees (pi/2)
        angle_from_vertical = np.abs(angle_with_up - np.pi/2)
        
        return angle_from_vertical <= self.max_wall_angle
    
    def get_plane_height(self, points: np.ndarray) -> float:
        """
        Calculate the height (z-range) of points in a plane.
        
        Args:
            points: Array of 3D points
            
        Returns:
            Height of the plane (max_z - min_z)
        """
        if len(points) == 0:
            return 0.0
        z_coords = points[:, 2]
        return np.max(z_coords) - np.min(z_coords)
    
    def calculate_plane_dimensions(self, points: np.ndarray, plane_model: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate the width, height, and area of a plane.
        
        Args:
            points: Array of 3D points belonging to the plane
            plane_model: Plane equation coefficients [a, b, c, d]
            
        Returns:
            Tuple of (width, height, area)
        """
        if len(points) < 3:
            return 0.0, 0.0, 0.0
        
        # Get height (z-range)
        height = self.get_plane_height(points)
        
        # Project points onto the plane and calculate 2D dimensions
        try:
            # Get the normal vector
            normal = plane_model[:3]
            normal = normal / np.linalg.norm(normal)
            
            # Create a coordinate system on the plane
            # Find two orthogonal vectors in the plane
            if abs(normal[2]) < 0.9:
                # If normal is not too close to z-axis, use z-axis for cross product
                v1 = np.cross(normal, [0, 0, 1])
            else:
                # Use x-axis for cross product
                v1 = np.cross(normal, [1, 0, 0])
            
            v1 = v1 / np.linalg.norm(v1)
            v2 = np.cross(normal, v1)
            v2 = v2 / np.linalg.norm(v2)
            
            # Project all points onto the 2D plane coordinate system
            projected_points = []
            for point in points:
                # Translate point relative to first point
                relative_point = point - points[0]
                # Project onto the two plane vectors
                u = np.dot(relative_point, v1)
                v = np.dot(relative_point, v2)
                projected_points.append([u, v])
            
            projected_points = np.array(projected_points)
            
            # Calculate bounding box dimensions
            min_coords = np.min(projected_points, axis=0)
            max_coords = np.max(projected_points, axis=0)
            width = np.max(max_coords - min_coords)
            
            # Calculate approximate area using convex hull
            try:
                if len(projected_points) >= 3:
                    hull = ConvexHull(projected_points)
                    area = hull.volume  # In 2D, volume is actually area
                else:
                    area = 0.0
            except:
                # Fallback: use bounding box area
                area = (max_coords[0] - min_coords[0]) * (max_coords[1] - min_coords[1])
            
        except Exception as e:
            print(f"Error calculating plane dimensions: {e}")
            # Fallback: use simple 3D bounding box
            min_coords = np.min(points, axis=0)
            max_coords = np.max(points, axis=0)
            width = np.max([max_coords[0] - min_coords[0], max_coords[1] - min_coords[1]])
            area = width * height
        
        return width, height, area
    
    def is_valid_wall(self, points: np.ndarray, plane_model: np.ndarray) -> Tuple[bool, Dict]:
        """
        Check if a detected plane is a valid wall based on multiple criteria.
        
        Args:
            points: Array of 3D points belonging to the plane
            plane_model: Plane equation coefficients
            
        Returns:
            Tuple of (is_valid, criteria_dict)
        """
        # Basic checks
        is_vertical = self.is_vertical_plane(plane_model)
        width, height, area = self.calculate_plane_dimensions(points, plane_model)
        
        # Wall criteria
        criteria = {
            'is_vertical': is_vertical,
            'height': height,
            'width': width,
            'area': area,
            'num_points': len(points),
            'height_ok': height >= self.min_wall_height,
            'width_ok': width >= self.min_wall_width,
            'area_ok': area >= self.min_wall_area,
            'points_ok': len(points) >= self.min_points_per_wall
        }
        
        # A valid wall must satisfy ALL criteria
        is_valid = (criteria['is_vertical'] and 
                   criteria['height_ok'] and 
                   criteria['width_ok'] and 
                   criteria['area_ok'] and 
                   criteria['points_ok'])
        
        return is_valid, criteria
    
    def segment_planes(self, cloud: o3d.geometry.PointCloud) -> List[Dict]:
        """
        Segment planes from the point cloud using RANSAC.
        
        Args:
            cloud: Input point cloud
            
        Returns:
            List of dictionaries containing plane information
        """
        print("Starting plane segmentation...")
        
        planes = []
        remaining_cloud = cloud
        iteration = 0
        max_planes = 20  # Prevent infinite loops
        
        while len(remaining_cloud.points) > self.min_points_per_wall and iteration < max_planes:
            print(f"Segmenting plane {iteration + 1}...")
            
            # Apply RANSAC plane segmentation
            plane_model, inliers = remaining_cloud.segment_plane(
                distance_threshold=self.distance_threshold,
                ransac_n=self.ransac_n,
                num_iterations=self.num_iterations
            )
            
            if len(inliers) < self.min_points_per_wall:
                print(f"Plane {iteration + 1}: Too few points ({len(inliers)}), stopping")
                break
                
            # Extract plane points
            plane_cloud = remaining_cloud.select_by_index(inliers)
            plane_points = np.asarray(plane_cloud.points)
            
            # Check if this is a valid wall
            is_wall, criteria = self.is_valid_wall(plane_points, plane_model)
            
            plane_info = {
                'model': plane_model,
                'points': plane_points,
                'point_cloud': plane_cloud,
                'inlier_count': len(inliers),
                'is_wall': is_wall,
                'criteria': criteria,
                'normal': plane_model[:3] / np.linalg.norm(plane_model[:3])
            }
            
            planes.append(plane_info)
            
            print(f"Plane {iteration + 1}: {len(inliers)} points, "
                  f"height: {criteria['height']:.2f}m, "
                  f"width: {criteria['width']:.2f}m, "
                  f"area: {criteria['area']:.2f}m², "
                  f"vertical: {criteria['is_vertical']}, "
                  f"wall: {is_wall}")
            
            # Remove the segmented plane from the remaining cloud
            remaining_cloud = remaining_cloud.select_by_index(inliers, invert=True)
            iteration += 1
            
        print(f"Segmented {len(planes)} planes total")
        return planes
    
    def detect_walls(self, file_path: str) -> List[Dict]:
        """
        Main method to detect walls in a point cloud.
        
        Args:
            file_path: Path to the point cloud file
            
        Returns:
            List of detected wall information
        """
        # Load and preprocess point cloud
        cloud = self.load_point_cloud(file_path)
        cloud = self.preprocess_point_cloud(cloud)
        
        # Segment planes
        planes = self.segment_planes(cloud)
        
        # Filter walls
        self.walls = [plane for plane in planes if plane['is_wall']]
        self.wall_point_clouds = [wall['point_cloud'] for wall in self.walls]
        
        print(f"\nDetected {len(self.walls)} walls out of {len(planes)} planes")
        
        return self.walls
    
    def visualize_results(self):
        """
        Visualize the wall detection results interactively.
        """
        if not self.walls:
            print("No walls detected to visualize")
            return
            
        print("Creating visualization...")
        
        # Create visualization geometries
        geometries = []
        
        # Add original point cloud in gray
        if self.processed_cloud:
            original_vis = o3d.geometry.PointCloud()
            original_vis.points = self.processed_cloud.points
            original_vis.paint_uniform_color([0.7, 0.7, 0.7])  # Gray
            geometries.append(original_vis)
        
        # Color palette for walls
        colors = [
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green
            [0.0, 0.0, 1.0],  # Blue
            [1.0, 1.0, 0.0],  # Yellow
            [1.0, 0.0, 1.0],  # Magenta
            [0.0, 1.0, 1.0],  # Cyan
            [1.0, 0.5, 0.0],  # Orange
            [0.5, 0.0, 1.0],  # Purple
        ]
        
        # Add detected walls with different colors
        for i, wall in enumerate(self.walls):
            wall_cloud = o3d.geometry.PointCloud()
            wall_cloud.points = o3d.utility.Vector3dVector(wall['points'])
            color = colors[i % len(colors)]
            wall_cloud.paint_uniform_color(color)
            geometries.append(wall_cloud)
        
        # Interactive visualization
        print("Showing interactive visualization...")
        o3d.visualization.draw_geometries(
            geometries,
            window_name="Wall Detection Results",
            width=1200,
            height=800
        )
    
    def save_results(self):
        """
        Save the detection results to files in wall_detection_results folder.
        """
        if not self.walls:
            print("No walls to save")
            return
        
        # Create results directory
        results_dir = "wall_detection_results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        print(f"Saving results to {results_dir}/...")
        
        # Save individual wall point clouds
        for i, wall in enumerate(self.walls):
            filename = os.path.join(results_dir, f"wall_{i+1}.pcd")
            wall_cloud = o3d.geometry.PointCloud()
            wall_cloud.points = o3d.utility.Vector3dVector(wall['points'])
            o3d.io.write_point_cloud(filename, wall_cloud)
            print(f"Saved wall {i+1} to: {filename}")
        
        # Save combined walls point cloud
        if len(self.wall_point_clouds) > 0:
            combined_walls = o3d.geometry.PointCloud()
            all_wall_points = []
            
            for wall in self.walls:
                all_wall_points.extend(wall['points'])
                
            combined_walls.points = o3d.utility.Vector3dVector(all_wall_points)
            combined_filename = os.path.join(results_dir, "all_walls_combined.pcd")
            o3d.io.write_point_cloud(combined_filename, combined_walls)
            print(f"Saved combined walls to: {combined_filename}")
        
        # Save wall information to JSON
        wall_info = []
        for i, wall in enumerate(self.walls):
            info = {
                'wall_id': i + 1,
                'num_points': len(wall['points']),
                'height': float(wall['criteria']['height']),
                'width': float(wall['criteria']['width']),
                'area': float(wall['criteria']['area']),
                'normal_vector': wall['normal'].tolist(),
                'plane_equation': wall['model'].tolist()
            }
            wall_info.append(info)
            
        json_filename = os.path.join(results_dir, "wall_detection_info.json")
        with open(json_filename, 'w') as f:
            json.dump(wall_info, f, indent=2)
        print(f"Saved wall information to: {json_filename}")
    
    def print_summary(self):
        """Print a summary of the wall detection results."""
        if not self.walls:
            print("No walls detected.")
            return
            
        print(f"\n{'='*60}")
        print("WALL DETECTION SUMMARY")
        print(f"{'='*60}")
        print(f"Total walls detected: {len(self.walls)}")
        print()
        
        for i, wall in enumerate(self.walls):
            criteria = wall['criteria']
            print(f"Wall {i+1}:")
            print(f"  Points: {len(wall['points'])}")
            print(f"  Height: {criteria['height']:.2f} m")
            print(f"  Width: {criteria['width']:.2f} m")
            print(f"  Area: {criteria['area']:.2f} m²")
            print(f"  Normal: [{wall['normal'][0]:.3f}, {wall['normal'][1]:.3f}, {wall['normal'][2]:.3f}]")
            print(f"  Plane equation: {wall['model'][0]:.3f}x + {wall['model'][1]:.3f}y + {wall['model'][2]:.3f}z + {wall['model'][3]:.3f} = 0")
            print()


def main():
    """Main function to run wall detection."""
    parser = argparse.ArgumentParser(description='Detect walls in 3D point cloud data')
    parser.add_argument('input_file', help='Path to input .pcd file')
    
    args = parser.parse_args()
    
    # Create wall detector with fixed parameters
    detector = WallDetector()
    
    try:
        # Detect walls
        walls = detector.detect_walls(args.input_file)
        
        # Print summary
        detector.print_summary()
        
        # Visualize results
        detector.visualize_results()
        
        # Save results
        detector.save_results()
        
        print(f"\nWall detection completed successfully!")
        print(f"Found {len(walls)} walls in the point cloud.")
        print("Results saved in 'wall_detection_results' folder.")
        
    except Exception as e:
        print(f"Error during wall detection: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main())