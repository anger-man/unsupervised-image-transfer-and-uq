# unsupervised-depth-estimation
Unsupervised single-shot depth estimation with perceptual reconstruction loss

Implementation of a framework for fully unsupervised single-view depth estimation as proposed in:

Preprint version: #####

## Installation

```
#from github
git clone https://github.com/anger-man/unsupervised-depth-estimation
cd unsupervised-depth-estimation
conda env create --name tf_2.2.0 --file=environment.yml
conda activate tf_2.2.0
```
## Usage

Model architectures and training parameters can be set in **config_file.ini**.
Then:
```
python train.py --direc path_to_project_folder
```
## Data preparation

RGB images should be given in .jpg-format with dimension **d**x**d**x**3**, where the possible values for spatial resolution **d** should be powers of 2 (at least 128). Depth images should be given in .npy-format with dimension **d**x**d**x**1**. The evaluation folder should contain some paired samples, where an RGB image is linked by an unique index with its depth counterpart.

```
./path_to_project_folder/
|--input
|----random_filename1.jpg
|----random_filename2.jpg
|--target
|----random_filename1.npy
|----random_filename2.npy
|--evaluation
|----image_index1.jpg
|----depth_index1.npy
|----image_index2.jpg
|----depth_index2.npy
```
The directory *face_depth* gives an example of the needed structure of the project folder. It contains pre-processed samples taken from the **Texas 3D Face Recognition** database (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5483908).



