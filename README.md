# Surgical_Instruments_Segmentation
This repository is the implementation of the paper [Real-Time Instrument Segmentation in Robotic Surgery Using Auxiliary Supervised Deep Adversarial Learning](https://ieeexplore.ieee.org/abstract/document/8648150).


[MICCAI Surgical Instrument Segmentation Challenge 2017](https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org/)
dataset is used to conduct all the experiments in this paper. The dataset is split into train and validation set as:

Train Set: 1,2,3,4,5,6

Validation Set: 7,8 

[Trained model in type wise segmentation](https://drive.google.com/file/d/10s1NQhbJsEUDsrax7MvQWQUwOwSucsDi/view?usp=sharing) 

### Architectures:
<img align='center' style="border-color:gray;border-width:2px;border-style:dashed"   src='figures/Proposed_Architecture.png' padding='5px' height="500px"></img>


### Results: Binary Prediction
<img align='center' style="border-color:gray;border-width:2px;border-style:dashed"   src='figures/results.png' padding='5px' height="600px"></img>

## Acknowledgement
The adversarial learning part is adopted from this [repository](https://github.com/hfslyc/AdvSemiSeg)

## Citation
If you use this code for your research, please cite our paper.

```
@article{islam2019real,
  title={Real-time instrument segmentation in robotic surgery using auxiliary supervised deep adversarial learning},
  author={Islam, Mobarakol and Atputharuban, Daniel Anojan and Ramesh, Ravikiran and Ren, Hongliang},
  journal={IEEE Robotics and Automation Letters},
  volume={4},
  number={2},
  pages={2188--2195},
  year={2019},
  publisher={IEEE}
}
```
