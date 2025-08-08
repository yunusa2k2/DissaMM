# DissaMM
DissaMM is a multi-modal neural architecture for classifying disaster-related social media posts by jointly processing images and text. It combines a modified RegNet visual encoder with BERT for text encoding, and fuses their outputs for final classification. Unlike DisasterNet (Johnson et al., 2020), which relies only on visual cues, DissaMM integrates textual context to improve accuracy while keeping computational cost low by freezing the textual encoder during training. It also integrates explainable AI (xAI) methods, SHAP and LIME, to explain predictions and highlight the contribution of visual and textual features.

Reference:
Johnson, M., Murthy, D., Robertson, B., Smith, R., & Stephens, K. (2020). DisasterNet: Evaluating the performance of transfer learning to classify hurricane-related images posted on Twitter. Proceedings of the International ISCRAM Conference on Information Systems for Crisis Response and Management, 17(1), 699–713.
