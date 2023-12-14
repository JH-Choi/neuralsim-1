import argparse
import numpy as np
import os
import pickle
import shutil
from pyquaternion import Quaternion


def read_cameras_file(filename):
    cameras = {}
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('#') or len(line.strip()) == 0:
                continue  # Skip comments and empty lines
            parts = line.strip().split()
            camera_id = int(parts[0])
            model = parts[1]
            if model == "PINHOLE":
                focal_length_x, focal_length_y = float(parts[4]), float(parts[5])
                cx, cy = float(parts[6]), float(parts[7])
                intr_matrix = np.array([
                    [focal_length_x, 0, cx],
                    [0, focal_length_y, cy],
                    [0, 0, 1]
                ])
                cameras[camera_id] = {
                    'hw': [int(parts[3]), int(parts[2])],
                    'intr': intr_matrix,
                    'distortion': np.zeros(5)  # No distortion parameters for PINHOLE model
                }
    return cameras



def read_images_file(filename):
    images = {}
    global_frame_ind = -1
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('#') or len(line.strip()) == 0:
                continue  # Skip comments and empty lines
            parts = line.strip().split()
            image_id = int(parts[0])
            qw, qx, qy, qz = map(float, parts[1:5])  # Quaternion
            tx, ty, tz = map(float, parts[5:8])  # Translation
            camera_id = int(parts[8])
            image_name = os.path.basename(parts[9])
            global_frame_ind = global_frame_ind + 1;

            # Convert quaternion and translation into a 4x4 matrix
            quaternion = Quaternion(qw, qx, qy, qz)
            rotation_matrix = quaternion.rotation_matrix
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = rotation_matrix
            transformation_matrix[:3, 3] = [tx, ty, tz]
            images[image_id] = {
                'image_name': image_name,
                'camera_id': camera_id,
                'timestamp': 0,
                'global_frame_ind': global_frame_ind,
                'transformation_matrix': transformation_matrix
            }
            print(image_name, ":", global_frame_ind)
    return images

def create_symbolic_links(src_dir, dst_dir, images, extension):

    src_dir = os.path.realpath(src_dir)
    dst_dir = os.path.realpath(dst_dir)
    
    if not os.path.exists(src_dir):
        return  # Skip if source directory does not exist

    os.makedirs(dst_dir, exist_ok=True)

    for img_id, img_data in images.items():
        global_frame_ind = f"{images[img_id]['global_frame_ind']:08d}"  # Pad with zeros up to 8 digits
        src_file = os.path.join(src_dir, os.path.splitext(img_data['image_name'])[0] + extension)
        dst_file = os.path.join(dst_dir, f"{global_frame_ind}" + extension)

        if os.path.exists(src_file):
            if os.path.exists(dst_file):
                os.remove(dst_file)
            os.symlink(src_file, dst_file)

def main():
    parser = argparse.ArgumentParser(description='Process and save COLMAP data.')
    parser.add_argument('--scene_id', type=str, required=True, help='An ID for the scene')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory containing COLMAP model files')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the processed data')
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory for additional data')

    args = parser.parse_args()

    cameras_file = os.path.join(args.model_dir, 'cameras.txt')
    images_file = os.path.join(args.model_dir, 'images.txt')

    # Read cameras and images from COLMAP files
    cameras = read_cameras_file(cameras_file)
    images = read_images_file(images_file)


    n_frames = len(images)

    # Initialize 'hw', 'intr', and 'distortion' arrays
    hw_array = np.array([cameras[images[i]['camera_id']]['hw'] for i in images], dtype=np.int32)
    intr_array = np.array([cameras[images[i]['camera_id']]['intr'] for i in images])
    distortion_array = np.zeros((len(images), 5))  # Array of zeros for distortion

    timestamp_array = np.array([images[i]['timestamp'] for i in images])
    global_frame_ind_array = np.array([images[i]['global_frame_ind'] for i in images])

    T_cw = np.array([images[i]['transformation_matrix'] for i in images])
    T_wc = np.linalg.inv(T_cw)  # Inverse of T_cw

    # Transformation matrices
    opencv_to_waymo = np.eye(4)
    opencv_to_waymo[:3, :3] = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    
    c2v_0 = np.eye(4)  # Assuming T_vc is identity
    c2v = c2v_0 @ opencv_to_waymo
    sensor_v2w = T_wc  # Assuming T_wv = T_wc
    c2w = T_wc @ c2v_0 @ opencv_to_waymo  # T_wv @ T_vc @ opencv_to_waymo

    # Replicate transformations for each frame
    c2v_0_array = np.tile(c2v_0, (n_frames, 1, 1))
    c2v_array = np.tile(c2v, (n_frames, 1, 1))
    sensor_v2w_array = sensor_v2w
    c2w_array = c2w
    v2w_array = sensor_v2w_array  # v2w is the same as sensor_v2w

    # Initialize data structure with transformations
    data = {
        "scene_id": args.scene_id,
        "metas": {
            "n_frames": n_frames,
            "dynamic_stats": None,
            "Vehicle": None,
            "Pedestrian": None,
            "Sign": None
        },
        "objects": {},
        "observers": {
            "camera_FRONT": {
                "class_name": "Camera",
                "n_frames": n_frames,
                "data": {
                    'hw': hw_array,
                    'intr': intr_array,
                    'distortion': distortion_array,
                    'c2v_0': c2v_0_array,
                    'c2v': c2v_array,
                    'sensor_v2w': sensor_v2w_array,
                    'c2w': c2w_array,
                    'timestamp': timestamp_array,
                    'global_frame_ind': global_frame_ind_array
                }
            },
            "ego_car": {
                "class_name": "EgoVehicle",
                "n_frames": len(images),
                "data": {
                    "v2w": v2w_array,  # Same as sensor_v2w in camera_FRONT
                    'timestamp': timestamp_array,
                    "global_frame_ind": global_frame_ind_array
                }
            }
        }
    }

    # Save the data to 'scenario.pt' in the specified save directory
    save_path = os.path.join(args.save_dir, 'scenario.pt')
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)

    create_symbolic_links(os.path.join(args.root_dir, "images"), os.path.join(args.save_dir, "images/camera_FRONT"), images, ".png")
    create_symbolic_links(os.path.join(args.root_dir, "depths"), os.path.join(args.save_dir, "depths/camera_FRONT"), images, ".npz")
    create_symbolic_links(os.path.join(args.root_dir, "normals"), os.path.join(args.save_dir, "normals/camera_FRONT"), images, ".jpg")
    create_symbolic_links(os.path.join(args.root_dir, "masks"), os.path.join(args.save_dir, "masks/camera_FRONT"), images, ".npz")


    print(f"Data saved to {save_path}")

if __name__ == "__main__":
    main()

