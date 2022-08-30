# unsupervised-depth-estimation-and-uq
Unsupervised joint image transfer and uncertainty quantification using patch invariant networks

Implementation of a light-weight framework for completely unpaired domain mapping and simultaneous uncertainty quantification.

Preprint version: https://arxiv.org/abs/2207.04325

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
python train_uapi.py --direc path_to_project_folder
```
For example:
```
python train_uapi.py --direc 'ixi/'
```
## Data preparation

Input and target images should be given in .jpg-format with spatial dimension **d**x**d**, where the possible values for spatial resolution **d** should be powers of 2 (at least 128). The evaluation folder should contain some paired samples, where an image from the input domain is linked by an unique index with its target counterpart.

```
./path_to_project_folder/
|--input
|----random_filename1.jpg
|----random_filename2.jpg
|--target
|----random_filename1.jpg
|----random_filename2.jpg
|--evaluation
|----index1_input.jpg
|----index1_target.jpg
|----index2_input.jpg
|----index2_target.jpg
```
The directory *IXI* gives an example of the needed structure of the project folder. It contains pre-processed samples taken from the **IXI Dataset** (https://brain-development.org/ixi-dataset/).



