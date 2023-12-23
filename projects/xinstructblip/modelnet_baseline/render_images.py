
 # Copyright (c) 2023, salesforce.com, inc.
 # All rights reserved.
 # SPDX-License-Identifier: BSD-3-Clause
 # For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

from tqdm import tqdm
import pickle
import open3d as o3d
import numpy as np
from PIL import Image

data, labels = pickle.load(open('modelnet40_test_1024pts.dat', 'rb'))
for i,points in tqdm(enumerate(data)):
	# Convert the numpy array to Open3D PointCloud
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(points[:, 0:3])

	# Create a visualization window and set up the view
	vis = o3d.visualization.Visualizer()
	vis.create_window(width=800, height=600, visible=False)  # Set to off-screen mode
	vis.add_geometry(pcd)

	# Set viewpoint parameters if desired (this is optional)
	# For example, set the camera to view from a specific position:
	# vis.get_view_control().set_lookat([x, y, z])
	# vis.get_view_control().set_front([x, y, z])
	# vis.get_view_control().set_up([x, y, z])
	# vis.get_view_control().set_zoom(zoom_factor)

	# Capture the image to a numpy array
	img_array = np.asarray(vis.capture_screen_float_buffer(do_render=True))
	# Convert the numpy array to a PIL Image
	img_pil = Image.fromarray((img_array*255.).astype(np.uint8))
	# Close the visualization
	vis.destroy_window()

	# Display the image (if desired)
	# img_pil.show()
	# from pdb import set_trace; set_trace()
	img_pil.save(f'images_test40_rgb/{i}.jpeg')
