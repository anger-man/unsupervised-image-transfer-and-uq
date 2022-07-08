# unsupervised-depth-estimation-and-uq
Unsupervised joint image transfer and uncertainty quantification using patch invariant networks

Implementation of a light-weight framework for completely unpaired domain mapping and simultaneous uncertainty quantification.
Preprint version: #####

## Installation

```
#from github
git clone https://github.com/anger-man/unsupervised-image-transfer-and-uq
cd unsupervised-image-transfer-and-uq
conda env create --name unsupervised-transfer-uq --file=environment_tf22.yml
conda activate unsupervised-transfer-uq
```
## Usage

Model architectures and training parameters can be set in **config_file.ini**.
Then:
```
python train.py --direc path_to_project_folder
```
## Data preparation

Input and target images should be given in .jpg-format with spatial dimension **d**x**d**, where the possible values for spatial resolution **d** should be powers of 2 (at least 128). The evaluation folder should contain some paired samples, where an image from the input domain is linked by an unique index with its target counterpart.

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
The directory *IXI* gives an example of the needed structure of the project folder. It contains pre-processed samples taken from the **IXI Dataset** (https://brain-development.org/ixi-dataset/).



