# MultimodalGEC
The implementation of the paper titled `Improving Grammatical Error Correction with Multimodal Feature Integration`, which was accepted in the Findings of ACL 2023.

## Introduction
 

<div align="center">
    <img src="/images/overall-framework.png" width="70%" title="Overall framework of multimodal GEC model."</img>
    <p class="image-caption">Overall framework of multimodal GEC model. </p>
</div>

Grammatical error correction (GEC) is a promising task aimed at correcting errors in a text. Many methods have been proposed to facilitate this task with remarkable results. However, most of them only focus on enhancing textual feature extraction without exploring the usage of other modalities' information (e.g., speech), which can also provide valuable knowledge to help the model detect grammatical errors. To shore up this deficiency, we propose a novel framework that integrates both speech and text features to enhance GEC. In detail, we create new multimodal GEC datasets for English and German by generating audio from text using the advanced text-to-speech models. Subsequently, we extract acoustic and textual representations by a multimodal encoder that consists of a speech and a text encoder. A mixture-of-experts (MoE) layer is employed to selectively align representations from the two modalities, and then a dot attention mechanism is used to fuse them as final multimodal representations. 

## Multimodal Data Creation

Due to the large size of the English Clang8 data and the significant memory requirements for speech data, we are sharing the detailed methodology for generating English multimodal speech data.

- Step 1:  Requirements and Installation

    This implementation is based on [fairseq](https://github.com/facebookresearch/fairseq)
    - [PyTorch](https://pytorch.org/) version >= 1.10.0
    - Python version >= 3.8

     ```
   # to install the latest stable release (0.10.x)
   pip install fairseq

