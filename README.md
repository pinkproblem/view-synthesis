# Improving Stylized View Synthesis of Image-based Reconstructions using Neural Networks

This project contains two variants for generating e.g. line drawings from imperfect point clouds.
The variants are:

* **Image-based Network**: Is trained to convert rendered images of the point cloud based on color, depth and normal passes. Needs to train on a large dataset, but is generically applicable afterwards.
* **Voxel-based Network**: Is trained to generate views from a voxelized version of the point cloud. Is trained and applied to precisely one model per training.

Details can be found in the respective folders.