# LOPS : Learning Order Inspired Pseudo-Label Selection for Weakly Supervised Text Classification

- [Model](#model)
- [Training](#training)
	- [Required Inputs](#required-inputs)
	- [Commands](#commands)
	- [Requirements](#requirements)
- [Citation](#citation)

## Model


## Training

### Required inputs
Each Dataset should contain following files:
- **DataFrame pickle file**
  - Example: ```data/nyt-fine/df.pkl```
    - This dataset should contain two columns named ```text```, ```label```
    - ```text``` contains text and ```label``` contains its corresponding label.
    - Must be named as ```df.pkl```
- **Seed Words Json file**
  - Example: ```data/nyt-fine/seedwords.json```
    - This json file contains seed words list for each label.
    - Must be named as ```seedwords.json```

### Commands

The ```main.py``` requires five arguments: 
- ```dataset_path```, which is a path to dataset containing  required DataFrame and seedwords
- ```dataset``` is the name of the dataset. Ex: ```nyt-fine```
- ```clf``` is the text classifier, currently only accepts ```gpt2``` for GPT-2, ```bert``` for BERT, ```roberta``` for RoBERTa, and ```xlnet``` for XL-Net.
- ```gpu_id``` refers to the id of the gpu. If not mentioned, the process runs on cpu.
- ```lops``` is the flag if set to 1 enables LOPS and if set to 0 disables LOPS

Example command to run:
```shell script
$ python main.py --dataset_path dataset_path --dataset dataset --clf bert --gpu_id 3 --lops 1
```

### Requirements

This project is based on ```python==3.7```. The dependencies are as follow:
```
keras-contrib==2.0.8
scikit-learn==0.21.3
torch==1.9.1
argparse
transformers==4.3.3
nltk
scipy=1.3.1
numpy==1.17.2
```

## Citation

```
@inproceedings{mekala-etal-2022-lops,
    title = "{LOPS}: Learning Order Inspired Pseudo-Label Selection for Weakly Supervised Text Classification",
    author = "Mekala, Dheeraj  and
      Dong, Chengyu  and
      Shang, Jingbo",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-emnlp.360",
    pages = "4894--4908",
    abstract = "Weakly supervised text classification methods typically train a deep neural classifier based on pseudo-labels. The quality of pseudo-labels is crucial to final performance but they are inevitably noisy due to their heuristic nature, so selecting the correct ones has a huge potential for performance boost. One straightforward solution is to select samples based on the softmax probability scores in the neural classifier corresponding to their pseudo-labels. However, we show through our experiments that such solutions are ineffective and unstable due to the erroneously high-confidence predictions from poorly calibrated models. Recent studies on the memorization effects of deep neural models suggest that these models first memorize training samples with clean labels and then those with noisy labels. Inspired by this observation, we propose a novel pseudo-label selection method LOPS that takes learning order of samples into consideration. We hypothesize that the learning order reflects the probability of wrong annotation in terms of ranking, and therefore, propose to select the samples that are learnt earlier. LOPS can be viewed as a strong performance-boost plug-in to most existing weakly-supervised text classification methods, as confirmed in extensive experiments on four real-world datasets.",
}
```