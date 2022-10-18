This repository contains the official implementation for SA-INR introduced in the following paper:

# Spatial Attention-based Implicit Neural Representation for Arbitrary Reduction of MRI Slice Spacing
**Authors:**   
> Xin Wang[1], Sheng Wang[1], Honglin Xiong[2], Kai Xuan[1], Zixu Zhuang[1], Mengjun Liu[1], Zhenrong Shen[1], Xiangyu Zhao[1], Lichi Zhang[1], Qian Wang[2]
> 
**Institution:**
> [1] School of Biomedical Engineering, Shanghai Jiao Tong University, Shanghai, China
> 
> [2] School of Biomedical Engineering, ShanghaiTech University, Shanghai, China

The project page with video is at https://yinboc.github.io/liif/.
## Usage
## Quick Start

1. We provide a pre-trained model `checkpoing/model.pth` for reducing slice-spacing of knee MRI.

2. Use the following command for reducing slice-spacing of a single test case.

```python
python single_test.py --add_res --gpu [GPU] --save_dir [set a dir to save your images] --model_path [model_path] --nii_path [set the path to your test case] --slice_spacing [set your desired slice spacing] 
```
We also provide a knee MRI `test/knee.nii.gz` for testing.

## Reproducing Experiments
### Data Preparation
- Split your data into training set and test set. 

- Write the paths of data to `[your_dataset_name].json` as in the following example.

```
{'train': [case1.nii.gz, case2.nii.gz...], 'test': [case3.nii.gz, case4.nii.gz...]}
```

- Edit `train_SA_INR.yaml `, change `data: clinical_knee.json` to `data: [your_dataset_name].json` 

### Training
```python
python train.py --add_res --gpu [GPU] --save_path [set a dir to save your checkpoints] --config [train_SA_INR.yaml]
```
In default, the local-aware spatial attention (LASA) is applied to each query coordinate. One can use `--add_branch` to learn a gating mask for conditionally applying LASA.

### Testing
```python
python test.py --add_res --gpu [GPU] --save_dir [set a dir to save your images] --model_path [model_path]  --slice_spacing [set your desired slice spacing]
```
In the same way, one can use `--add_branch` for conditionally applying LASA.
