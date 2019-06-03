import logging
import argparse
import yaml
import trimesh
import numpy as np
import os
import pathlib
import cv2


#from visualization import Visualizer3D as vis3d, Visualizer2D as vis
from autolab_core import RigidTransform, YamlConfig
from meshrender import Scene, SceneObject, VirtualCamera, DirectionalLight, AmbientLight, MaterialProperties, SceneViewer, UniformPlanarWorksurfaceImageRandomVariable
from perception import CameraIntrinsics, RenderMode, Image

#PLANE_MESH = '/home/vsatish/Workspace/dev/dex-net/data/objects/plane/plane.obj'
#PLANE_POSE = '/home/vsatish/Workspace/dev/dex-net/data/objects/plane/pose.tf'

parser = argparse.ArgumentParser()
parser.add_argument('mesh', type=str)
parser.add_argument('cfg', type=str)
parser.add_argument('num_images', type=int)
parser.add_argument('output_folder', type=str)
# parser.add_argument('--color', type=str, default="rgb") # rgb: save rgb image. depth: save depth as grayscale
parser.add_argument('--stable_pose', type=int, default=0)
parser.add_argument('--render_scene', action='store_true')
args = parser.parse_args()

# Create output folder if it doesn't already exist
pathlib.Path(args.output_folder + "/processed/images/").mkdir(parents=True, exist_ok=True)
pathlib.Path(args.output_folder + "/processed/images_rgb/").mkdir(parents=True, exist_ok=True)


mesh_path = args.mesh
stable_pose = args.stable_pose
render_scene = args.render_scene

print(mesh_path)


logging.getLogger().setLevel(logging.INFO)

logging.info('Loading mesh...')
mesh = trimesh.load(mesh_path)
# copy mesh to pdc location
trimesh.io.export.export_mesh(mesh, file_obj=args.output_folder + "/processed/fusion_mesh.ply", file_type="ply")


logging.info('Computing stable poses...')
# transforms, probs = trimesh.poses.compute_stable_poses(mesh)

scene = Scene()

# wrap the object to be rendered as a SceneObject
# rotation, translation = RigidTransform.rotation_and_translation_from_matrix(transforms[stable_pose])
# T_obj_table = RigidTransform(rotation=rotation, translation=translation)


# default pose
default_pose = RigidTransform(
        rotation=np.eye(3),
        translation=np.array([0.0, 0.0, 0.0]),
        from_frame='obj',
        to_frame='world'
)

obj_material_properties = MaterialProperties(
                            color = np.array([66, 134, 244])/255.,
                            # color = 5.0*np.array([0.1, 0.1, 0.1]),
                            k_a = 0.3,
                            k_d = 0.5,
                            k_s = 0.2,
                            alpha = 10.0,
                            smooth=False,
                            wireframe=False
                        )

obj = SceneObject(mesh, default_pose, obj_material_properties)
scene.add_object('to_render', obj)


# table_obj_properties = MaterialProperties(
#                       color = np.array([0, 0, 0]),
#                       )


# wrap the table as a SceneObject
# table_mesh = trimesh.load(PLANE_MESH)
# T_table_world = RigidTransform.load(PLANE_POSE)
# table = SceneObject(table_mesh, T_table_world, table_obj_properties)
# scene.add_object('table', table)

# add light
ambient = AmbientLight(
    color=np.array([1.0, 1.0, 1.0]),
    strength=1.0
)

scene.ambient_light = ambient

dl = DirectionalLight(
    direction=np.array([0, 0, -1.0]),
    color=np.array([1.0, 1.0, 1.0]),
    strength=1.0
)
scene.add_light('direc', dl)


#====================================
# Add a camera to the scene
#====================================

# Set up camera intrinsics
ci = CameraIntrinsics(
    frame = 'camera',
    fx = 525.0,
    fy = 525.0,
    cx = 320.0,
    cy = 240.0,
    skew=0.0,
    height=480,
    width=640
)

# Set up the camera pose (z axis faces away from scene, x to right, y up)
cp = RigidTransform(
    rotation = np.array([
        [0.0, 0.0, 1.0],
        [0.0, -1.0,  0.0],
        [1.0, 0.0,  0.0]
    ]),
    translation = np.array([0.0, 0.0, 0.0]),
    from_frame='camera',
    to_frame='world'
)

# Create a VirtualCamera
camera = VirtualCamera(ci, cp)

# Add the camera to the scene
scene.camera = camera

cfg = YamlConfig(args.cfg)

# cfg = {
#   'focal_length': {
#       'min' : 520,
#       'max' : 520,
#   },
#   'delta_optical_center': {
#       'min' : 0.0,
#       'max' : 0.0,
#   },
#   'radius': {
#       'min' : 0.1,
#       'max' : 0.15,
#   },
#   'azimuth': {
#       'min' : 0.0,
#       'max' : 360.0,
#   },
#   'elevation': {
#       'min' : 0.0,
#       'max' : 360.0,
#   },
#   'roll': {
#       'min' : 0.0,
#       'max' : 360.0,
#   },
#   'x': {
#       'min' : -0.01,
#       'max' : -0.01,
#   },
#   'y': {
#       'min' : -0.01,
#       'max' : 0.01,
#   },
#   'im_width': 640,
#   'im_height': 480
# }


# num_images: 874 for test dataset, 3730 for train dataset

urv = UniformPlanarWorksurfaceImageRandomVariable('to_render', scene, [RenderMode.COLOR, RenderMode.DEPTH], 'camera', cfg)
renders = urv.sample(args.num_images, front_and_back=True)

# v = SceneViewer(scene, raymond_lighting=True)

camera_intr = None
camera_poses = {}

for i, render in enumerate(renders):
    color = render.renders[RenderMode.COLOR]
    depth = render.renders[RenderMode.DEPTH]
    # print(np.max(depth.data))
    # print(np.min(depth.data))
    # print(1/0)
    # depth_scaled = depth.data/0.2 * 255.
    # depth_to_grayscale = np.stack([depth_scaled]*3, axis=2)
    # depth_grayscale_image = Image.from_array(depth_to_grayscale)

    # vis.figure()
    # vis.imshow(color)
    # vis.show()

    color_filename = "{0:06d}_rgb.png".format(i)
    depth_filename = "{0:06d}_depth.png".format(i)

    camera_intr = render.camera.camera_intr


    quaternion = render.camera.T_camera_world.quaternion
    translation = render.camera.T_camera_world.translation # !! it is possible that the order may be w.r.t theirs. is this xyz?

    camera_poses[i] = {
        'camera_to_world': {
            'quaternion': {
                'w': float(quaternion[0]),
                'x': float(quaternion[1]),
                'y': float(quaternion[2]),
                'z': float(quaternion[3]),
            },
            'translation': {
                'x': float(translation[0]),
                'y': float(translation[1]),
                'z': float(translation[2]),
            }
        },
        'depth_image_filename': depth_filename,
        'rgb_image_filename': color_filename,
    }

    # if args.color == "rgb":
    #   color.save(args.output_folder + "/processed/images/{}".format(color_filename))
    # else:
    #   assert args.color == "depth"
    #   depth_grayscale_image.save(args.output_folder + "/processed/images/{}".format(color_filename))

    # color.save(args.output_folder + "/processed/images_rgb/{}".format(color_filename))
    color.save(args.output_folder + "/processed/images/{}".format(color_filename))
    # depth.save(args.output_folder + "/processed/images_rgb/{}".format(depth_filename))

    # depth_grayscale_image.save(args.output_folder + "/processed/images/{}".format(color_filename))

    if True:
        depth.save(args.output_folder + "/processed/images/{}".format(depth_filename))
    else: # saving depth images in the same way as manuelli et al.
        depth_mm = 1000.0 * depth.data
        print("using opencv depth save")
        cv2.imwrite(args.output_folder + "/processed/images/{}".format(depth_filename), depth_mm.astype(np.uint16))

    # depth_grayscale_image.save(args.output_folder + "/processed/images_rgb/{}".format(color_filename))


camera_info = {
    'camera_matrix': {
        'data': [float(camera_intr.fx), 0, float(camera_intr.cx), 0, float(camera_intr.fy), float(camera_intr.cy), 0, 0, 1]
    },
    'image_width': camera_intr.width,
    'image_height': camera_intr.height
}

with open(args.output_folder + "/processed/images/camera_info.yaml", 'w') as outfile:
    yaml.dump(camera_info, outfile, default_flow_style=False)

with open(args.output_folder + "/processed/images/pose_data.yaml", 'w') as outfile:
    yaml.dump(camera_poses, outfile, default_flow_style=False)

# v = SceneViewer(scene, raymond_lighting=True)
