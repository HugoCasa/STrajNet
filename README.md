# STrajNet with trajectory prediction
This repo is an extension of the STrajNet model to include trajectory prediction.

The goal was to add multi-modal trajectory prediction and encourage agreement between the grid and trajectory predictions to improve the overall performance.
This required adding a seperate head for the trajectory prediction with its own loss as well as a consistency loss.
The trajectory are predicted in a local frame of reference.

To toggle on and off the trajectory prediction, simply change to `True` or `False` the `traj_pred` variable in `train.py`.

## Description of changes

- Update preprocessing to include trajectory ground truths in local frame of reference.
- Added an additional cross-attention module (9 modules instead of 8) to pass to the trajectory prediction module
- Implemented a modified ViT for trajectory prediction
  - use the trajectories embeddings as queries
  - use the scene embedding as keys and values.
  - use 6 latent vectors for multimodality
- Implemented a trajectory loss
  - Average displacement error of the predicted trajectory closest to the ground truth
  - Classification error
- Consistency loss
  - Generate occupancy and flow grids from predicted trajectories
  - Compute cross-entropy and flow loss between generated and predicted grids

## Original STrajNet paper

**STrajNet: Multi-Model Hierarchical Transformer for Occupancy Flow Field Prediction in Autonomous Driving**
<br> [Haochen Liu](https://scholar.google.com/citations?user=iizqKUsAAAAJ&hl=en), [Zhiyu Huang](https://mczhi.github.io/), [Chen Lv](https://scholar.google.com/citations?user=UKVs2CEAAAAJ&hl=en) 
<br> [AutoMan Research Lab, Nanyang Technological University](https://lvchen.wixsite.com/automan)
<br> **[[arXiv]](http://arxiv.org/abs/2208.00394)**&nbsp;

