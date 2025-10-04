# DissaMM: Explainable Multi-Modal AI for Disaster Response

DissaMM is a lightweight, explainable, multi-modal neural architecture for classifying disaster-related social media posts. It uses a modified ResNet-50 for image feature extraction and BERT for textual encoding, with their outputs fused through a compact MLP head. This design enables real-time, interpretable disaster monitoring to support humanitarian operations, while keeping computational cost low. Figure 1 illustrates the architectural design.

<p align="center">
  <img src="images/2disaster.png" alt="DissaMM Architecture" width="90%">
  <br>
  <em>Figure: Architecture of the DissaMM model</em>
</p>

## Features

- ⚡ **Lightweight & real-time inference** – optimized for speed in resource-constrained environments using TL.  
- 🔗 **Multi-modal (text + image)** – processes both visual and textual information for richer context, unlike DisasterNet which relies only on images. Since social media posts are often noisy, we add a textual encoder to provide additional context.  
- 🔍 **Explainability** – integrates LIME and SHAP to generate transparent, human-interpretable decision-making visuals.  
- 🌍 **Low-resource optimization** – tailored for deployment in emerging regions with limited computational resources.  


## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yunusa2k2/DissaMM.git
cd DissaMM
pip install -r requirements.txt
```

## Dataset

For training and evaluation, we used the **DisasterNet dataset** introduced by Johnson et al. (2020).  

- **Source**: Collected via Twitter’s *Spritzer* stream (1% random sample of tweets) during **Hurricane Harvey (Aug 17 – Sep 17, 2017)**.  
- **Filtering**: Tweets were retrieved using disaster-related keywords such as  
  `hurricane`, `harvey`, `hurricaneharvey`, `harveyhouston`.  
- **Media**: Linked images were downloaded, deduplicated via MD5 checksums, and cleaned.  
- **Final dataset size**: ~17,483 unique images.  
- **Annotations**: A human-coded subset of 1,128 images was labeled by categories:  
  - **Time period**: pre-storm, landfall, aftermath/cleanup  
  - **Urgency**: 0 (spam/unrelated) → 4 (highly urgent)  
  - **Motifs**: ad, animals, damage, drink, food, gear, macro, outside, people, relief, other  

> ⚠️ **Note**: This dataset is **not included** in this repository. Please refer to the original publication for details on access:  
> Johnson, M., Murthy, D., Robertson, B., Smith, R., & Stephens, K. (2020).  
> *DisasterNet: Evaluating the performance of transfer learning to classify hurricane-related images posted on Twitter.* Proceedings of the International Conference on Web and Social Media (ICWSM).
