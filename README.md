# Star Shape Prior in Fully Convolutional Networks for Skin Lesion Segmentation
Encoding a differentiable form of the star shape prior in the loss function

# Abstract
Semantic segmentation is an important preliminary step towards automatic medical image interpretation. Recently deep convolutional neural networks have become the first choice for the task of pixelwise class prediction. While incorporating prior knowledge about the structure of target objects has proven effective in traditional energybased segmentation approaches, there has not been a clear way for encoding prior knowledge into deep learning frameworks. In this work, we propose a new loss term that encodes the star shape prior into the loss function of an end-to-end trainable fully convolutional network (FCN) framework. We penalize non-star shape segments in FCN prediction maps to guarantee a global structure in segmentation results. Our experiments indicated that leveraging the prior knowledge in fully convolutional networks yield convergence to an improved output space.

<p align="center">
   <img width='450' src="https://github.com/zmirikha/Star_Shape_Loss/blob/main/regional_map.png" alt="[regional_map]"/>
</p>

## Keywords
Semantic Segmentation, Fully Convolutional Networks, Star Shape Prior, Skin Lesion

## Cite
Zahra Mirikharaji, Ghassan Hamarneh, "[Star shape prior in fully convolutional networks for skin lesion segmentation](https://www.cs.sfu.ca/~hamarneh/ecopy/miccai2018a.pdf)", International Conference on Medical Image Computing and Computer-Assisted Intervention, 2018.

The corresponding bibtex entry is:

```
@inproceedings{mirikharaji2018star,
  title={Star shape prior in fully convolutional networks for skin lesion segmentation},
  author={Mirikharaji, Zahra and Hamarneh, Ghassan},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={737--745},
  year={2018},
  organization={Springer}
}
```
## Usage
An example usage is shown in `demo.ipynb` Star shape loss is calculated for a sample skin lesion image, its ground truth, and a predicted probability map, included in `images` directory.
