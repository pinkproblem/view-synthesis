# Image-based Network

This network learns to generate stylized views of color, depth an normal renders of point clouds. After training on a large dataset, it can be applied independently.
Note that due to size constraints the dataset only contains a few example images.

The following (non-default) python packages are needed:
* tensorflow 1.11.0
* imageio (optional)
* pyrr (optional, for reconstruction)
* plyfile (optional, for reconstruction)

## Dataset

The model uses different representations for every view in the training, e.g. depth and normal pass. The separate images should be stored in the *data* folder in the corresponding folders. Images belonging to the same view are identified by their name, so the different images for one view must have the same name to be recognized. 

Training and test sets are defined by placing the color pass images in the *train* and *test* folders. So for example, if there is an image named "view01" in the *train* folder, it will be used as color pass in the training dataset, and the corresponding depth and normal passes will be taken from the folders in *data* by searching for other images called "view01". Images in the *test* folder will be used as test dataset.

Refer to the given example images for more insight.

## Train

The training can then be started by running `image_network.py`.
Logs and intermediate results are placed in *./log*, and can be visualized using tensorboard.

## Evaluate

A trained model can be applied by feeding in a stack of images consisting of a color, depth and normal pass of a view of a point cloud.
Find an example in *eval.py*.
*eval.py* can also be used to run a default evaluation on the images in the *test* folder. Don't forget to set the correct model path in the script. Results will be placed in the specified folder.
Note that pretrained models are not included due to size restrictions. 

## Variant: Smoothing Network

This network tries to achieve frame-to-frame consistency when generating views. 

Training and test images must therefore be provided in a consecutive way when sorted by name, like the frames of a video.
The *turntable_* scripts in *../blender_scripts* provide a suitable rendering process for Blender.
Additionally, the training requires flow information for consecutive frames in .exr format.

The training can be started with `smooth_network.py` and evaluated with `eval_smooth`. Don't forget to set the correct model folder in the latter.

## Variant: Photo Hinting

This network can receive additional hints about the generated view, by passing it the nearest known view together with the corresponding flow information.

For every training image, the nearest known view must be provided in the *data/photo* folder. Also, the visual flow from the photo view to the generated view must be provided in *data/photo_flow*, as well as a corresponding validity mask in *data/photo_mask*, which can be generated using `mask.py`.

The training can be started with `photo_network.py` and evaluated with `eval_photo`. Don't forget to set the correct model folder in the latter.

## Reconstruction

A set of line drawings plus depth images created by `eval.py` can be reprojected into a point cloud using `reconstruction.py`. Just set the correct folder for lines and depth images in the script and run it; also make sure that corresponding images have the same name.

## Sources

[1] P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros, "Image-to-image translation with conditional
adversarial networks," arxiv, 2016.
