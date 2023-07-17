# MultimodalGEC
The multimodal GEC data creation for the paper titled `Improving Grammatical Error Correction with Multimodal Feature Integration`, which was accepted in the Findings of ACL 2023.

### Citation
```bibtex
@inproceedings{fang-etal-2023-improving,
    title = "Improving Grammatical Error Correction with Multimodal Feature Integration",
    author = "Fang, Tao  and
      Hu, Jinpeng  and
      Wong, Derek F.  and
      Wan, Xiang  and
      Chao, Lidia S.  and
      Chang, Tsung-Hui",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.594",
    pages = "9328--9344",
}
```

## Introduction
 

<div align="center">
    <img src="/images/overall-framework.png" width="70%" title="Overall framework of multimodal GEC model."</img>
    <p class="image-caption">Overall framework of multimodal GEC model. </p>
</div>

Grammatical error correction (GEC) is a promising task aimed at correcting errors in a text. Many methods have been proposed to facilitate this task with remarkable results. However, most of them only focus on enhancing textual feature extraction without exploring the usage of other modalities' information (e.g., speech), which can also provide valuable knowledge to help the model detect grammatical errors. To shore up this deficiency, we propose a novel framework that integrates both speech and text features to enhance GEC. In detail, we create new multimodal GEC datasets for English and German by generating audio from text using the advanced text-to-speech models. Subsequently, we extract acoustic and textual representations by a multimodal encoder that consists of a speech and a text encoder. A mixture-of-experts (MoE) layer is employed to selectively align representations from the two modalities, and then a dot attention mechanism is used to fuse them as final multimodal representations. 

## Multimodal Data Creation

### English GEC Multimodal Data

Due to the large size of the English Clang8 data and the significant memory requirements for speech data, we are sharing the detailed methodology for generating English multimodal speech data.

- Step 1:  Requirements and Installation

    This implementation is based on [fairseq](https://github.com/facebookresearch/fairseq)
    - [PyTorch](https://pytorch.org/) version >= 1.10.0
    - Python version >= 3.8
    

     ```
   # to install the latest stable release (0.10.x)
   pip install fairseq
   ```
   
- Step 2:  Download English Train/Dev/Test Data

  You can download the original data that was used to generate the speech data from the [Link](https://drive.google.com/drive/folders/1WwZzhI7VvUV1qJOaaGP0R3UMoFdE7Z2i?usp=share_link).
  
  
- Step 3:  Generate the speech data from the GEC data

     ```
   # To accommodate the large size of the English Clang8 data, you can split it into multiple smaller files for easier handling.
   sh generate_english_speech.sh
   ```
   
   Please refer to the [`examples`](https://github.com/NLP2CT/MultimodalGEC/tree/main/examples) folder where we have showcased some speech data generated using the CoNLL14 test set.

### German GEC Multimodal Data

Since the amount of German data is not substantial, we have released all the speech data directly, and you can download it from the provided [link](https://drive.google.com/drive/folders/1Cq2arkGpWQ7IQjbOjFC95F-DKTeD--IL?usp=share_link).


## Train and Evaluation

  ```
   sh train_and_eval.sh
   ```
Note: If you are training in German, change the.mv format in the [`dataset.py`](https://github.com/NLP2CT/MultimodalGEC/blob/main/gec_speech_moe_mse/model/dataset.py) file to read MP3 format
