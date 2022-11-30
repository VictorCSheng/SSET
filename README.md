# Serial Section ET
Here we report an intelligent workflow for sub-nanoscale 3D reconstruction of intact synapses from serial section electron tomography. 
Our workflow contains rapid ROI locating, automatic alignment, volume reconstruction, and semi-automatic synapse reconstruction. 
Multiple intact synapses in wild-type rat were reconstructed at a resolution of 0.664 nm/voxel to demonstrate the effectiveness of our workflow.


## Table of Contents
- [Dependencies](#Dependencies)
- [Instructions for Use](#Instructions-for-Use)
- [Examples and Comparison Results](#Examples-and-Comparison-Results)
- [Contributing](#Contributing)

## Dependencies
Our workflow is mainly based on python and matlab.
The required libraries are as follows. 

python3.8, numpy, scipy, tqdm, tifffile, skimage, matplotlib, opencv-python, opencv-contrib-python, torch, torchvision, dgl, connected-components-3d, networkx, Pillow, connectomics

If you don't have some of these libraries, you can install them using pip or another package manager.

For the installation of connectomics, see https://github.com/zudi-lin/pytorch_connectomics.

In addition, we also use the pre training network of Super-SloMo. For details of Super SloMo, see https://github.com/avinashpaliwal/Super-SloMo.

## Instructions for Use
If you want to use our workflow to reconstruct your target structure from serial section electron tomography, you only need to execute the files in order.

For a detailed description of our workflow, please refer to the paper "An intelligent workflow for sub-nanoscale 3D reconstruction of intact synapses from serial section electron tomography".

## Examples and Comparison Results
Here is a example of intact synapse reconstructed by our workflow at a resolution of 0.664 nm/voxel.
And we compare it with the results of IMOD, TrackEM and Irtool.

![Synapse reconstruction](https://github.com/VictorCSheng/SSET/raw/main/paper_image/fig8.png)

![Results comparison](https://github.com/VictorCSheng/SSET/raw/main/paper_image/fig9.png)

## Contributing
Please refer to the paper "An intelligent workflow for sub-nanoscale 3D reconstruction of intact synapses from serial section electron tomography".