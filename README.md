# InterNav-CaMP
## [CaMP: Causal Multi-policy Planning for Interactive Navigation in Multi-room Scenes](https://proceedings.neurips.cc/paper_files/paper/2023/file/333581887bf483296118a97773cab0c1-Paper-Conference.pdf) (Neurips 2023)

![](fig/episode_demo.gif)
#### Dataset Demonstration
![](fig/dataset_demo.gif)
#### Object Interaction Demonstration
![](fig/interaction_demo.gif)

### Installation
0. Clone this repository

   ```
   git clone git@github.com:polkalian/InterNav.git
   ```

1. Install `xorg` if the machine does not have it

   **Note**: This codebase should be executed on GPU. Thus, we need `xserver`for GPU redering.

   ```
   # Need sudo permission to install xserver
   sudo apt-get install xorg
   ```

   Then, do the xserver refiguration for GPU

   ```
   sudo python startx.py
   ```
   
3. Create a conda environment and install the required packages
   
   **Note**: The `python` version needs to be above `3.10`, since `python 2.x` may have issues with some required packages.

   ```
   conda create -n camp python=3.10
   pip install requirement.txt
   ```

### Environment/Dataset

   Our work is developed based on the physics-enabled, visually rich [AI2-THOR](http://ai2thor.allenai.org/) environment and [AllenAct](https://www.allenact.org/) framework.

   Download our dataset [here](https://drive.google.com/drive/folders/12i_Rfw558tPkqac_fgciYMDe-Ld9cV9Q?usp=sharing) and unzip it into the datasets folder.

### Train a new model from scratch with AllenAct

   Before running training or inference you'll first have to add the `InterNav` directory to your `PYTHONPATH` (so that `python` and `AllenAct` knows where to for various modules). To do this you can run the following:
    
   ```
   cd YOUR/PATH/TO/InterNav
   export PYTHONPATH=$PYTHONPATH:$PWD
   ```
    
   If you want to train a `CaMP` model, this can be done by running the command
    
   ```
   allenact -s 23456 -o out -b . configs/proc10k_ObsNav/obstacles_nav_rgbd_proc.py
   ```

   The `PPO+intent` model mentioned in the paper are also available in ivn_proc/models_baseline.py (corresponding to the tasks.py).

### Inference your model with AllenAct

   ```
   allenact -s 23456 -b . configs/proc10k_ObsNav/obstacles_nav_rgbd_proc.py -c PATH/TO/YOUR/MODEL --eval
   ```

### Citation

   If you find this project useful in your research, please consider citing our paper:

   ```
   @inproceedings{wang2023CaMP,
 author = {Wang, Xiaohan and Liu, Yuehu and Song, Xinhang and Wang, Beibei and Jiang, Shuqiang},
 booktitle = {Neurips},
 title = {CaMP: Causal Multi-policy Planning for Interactive Navigation in  Multi-room Scenes},
 url = {https://proceedings.neurips.cc/paper_files/paper/2023/file/333581887bf483296118a97773cab0c1-Paper-Conference.pdf},
 year = {2023}
}
   ```
