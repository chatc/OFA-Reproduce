# CSE597 Vision and Language - Fall 2022

In this project we reproduce the OFA model which is a well known a unified multimodal pretrained model based on transformers.


# Model Card
We list the parameters and pretrained checkpoints of OFAs below. For finetuned checkpoints, please refer to [checkpoints.md](checkpoints.md). 

<table border="1" width="100%">
    <tr align="center">
        <th>Model</th><th>Ckpt</th><th>Params</th><th>Backbone</th><th>Hidden size</th><th>Intermediate size</th><th>Num. of heads</th><th>Enc layers</th><th>Dec layers</th>
    </tr>
    <tr align="center">
        <td>OFA<sub>Tiny</sub></td><td><a href="https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/ofa_tiny.pt">Download</a></td><td>33M</td><td>ResNet50</td><td>256</td><td>1024</td><td>4</td><td>4</td><td>4</td>
    </tr>
    <tr align="center">
        <td>OFA<sub>Medium</sub></td><td><a href="https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/ofa_medium.pt">Download</a></td><td>93M</td><td>ResNet101</td><td>512</td></td><td>2048</td><td>8</td><td>4</td><td>4</td>
    </tr>
    <tr align="center">
        <td>OFA<sub>Base</sub></td><td><a href="https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/ofa_base.pt">Download</a></td><td>180M</td><td>ResNet101</td><td>768</td></td><td>3072</td><td>12</td><td>6</td><td>6</td>
    </tr>
    <tr align="center">
        <td>OFA<sub>Large</sub></td><td><a href="https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/ofa_large.pt">Download</a></td><td>470M</td><td>ResNet152</td><td>1024</td></td><td>4096</td><td>16</td><td>12</td><td>12</td>
    </tr>
</table>
<br></br>


# Requirements
* python 3.7.4
* pytorch 1.8.1
* torchvision 0.9.1
* JAVA 1.8 (for COCO evaluation)
<br></br>

# Installation
```bash
git clone https://github.com/OFA-Sys/OFA
pip install -r requirements.txt
```
<br></br>

# Datasets and Checkpoints
See [datasets.md](datasets.md) and [checkpoints.md](checkpoints.md).
<br></br>

# Finetuning & Inference
Below we provide methods for finetuning and inference on different downstream tasks. We provide both pretrained OFA-Large and OFA-Base in [checkpoints.md](checkpoints.md). The scripts mentioned in this section are prepared for OFA-Large. For reproducing the downstreaming results of OFA-Base, we have also provided the corresponding finetuning and inference scripts for OFA-Base in the `run_scripts/` folder.

We recommend that your workspace directory should be organized like this: 
```
OFA/
├── checkpoints/
│   ├── ofa_base.pt
│   ├── ofa_large.pt
│   ├── caption_large_best_clean.pt
│   └── ...
├── criterions/
├── data/
├── dataset/
│   ├── caption_data/
│   ├── gigaword_data/
│   └── ...
├── fairseq/
├── models/
├── run_scripts/
├── tasks/
├── train.py
├── trainer.py
└── utils/
```


## Visual Question Answering
Here we provide the finetuning and inference codes to reproduce the VQAv2 result. We believe much improvement on accuracy can still be achieved based on this codebase :)
<details>
    <summary><b>1. Prepare the Dataset & Checkpoints</b></summary>
    <p>
        Download data (see <a href="datasets.md">datasets.md</a>) and models (see <a href="checkpoints.md">checkpoints.md</a>) and put them in the correct directory. The dataset zipfile <code>vqa_data.zip</code> is around 100G and the decompressed data costs around 135G disk storage, which contains the training, validation and testing samples together with other necessary data resources. (Since <code>vqa_data.zip</code> is large in size, we have also provided chunked parts of the dataset files for more convenient and stable downloading. Please refer to <a href="https://github.com/OFA-Sys/OFA/issues/68#issuecomment-1096837349">issue #68</a>.) Following common practice, VG-QA samples are also included in the training data. To adapt to the seq2seq paradigm of OFA, we transform original VQA training questions with multiple golden answers into multiple training samples. For the original VQA validation set, we keep around 10k samples for our validation and utilize the other samples for training. Each line of the dataset represents a VQA sample with the following format. The information of question-id, image-id, question, answer (with confidence), predicted object labels (taken from <a href="https://github.com/pzzhang/VinVL">VinVL</a>, slightly brings around +0.1 accuracy improvement), image base64 string are separated by tabs. 
    </p>
<pre>
79459   79459   is this person wearing shorts?  0.6|!+no    house&&short&&...&&sky  /9j/4AAQS...tigZ/9k=
</pre>
    <p>
        For fine-tuning on customed VQA-formulated tasks, please refer to issue <a href="https://github.com/OFA-Sys/OFA/issues/76">#76</a> and <a href="https://github.com/OFA-Sys/OFA/issues/73">#73</a> for more information.
    </p>
</details>
<details>
    <summary><b>2. Shuffle the Training Data</b></summary>
    <p>
        (Optional, but achieves better finetuning accuracy): If the disk storage is sufficient, we recommend to prepare the shuffled training data for each epoch in advance. In our experiments, we use shuffling which brings around <b>+0.3</b> improvement on VQA accuracy.
    </p>
<pre>
cd dataset/vqa_data
ln vqa_train.tsv vqa_train_1.tsv
for idx in `seq 1 9`;do shuf vqa_train_${idx}.tsv > vqa_train_$[${idx}+1].tsv;done # each file is used for an epoch
</pre>
</details>
<details>
    <summary><b>3. Finetuning</b></summary>
    <p>
        In our experiments, the VQA finetuning is performed on 4 RTX8000 GPUs. Here provides the finetuning script <code>train_vqa_distributed.sh</code>, which supports multi-server distributed training (as well as single-server training). Please refer to the comments in the beginning of the script and set the configs correctly according to your distribution environment. If you have shuffled the training data in the previous step, please correctly specify the training data path following the guide in the script comments. <b>The command should be run on each worker.</b> 
    </p>
<pre>
# run on each worker after the distributed and data configs have been correctly set following the guide in train_vqa_distributed.sh 
cd run_scripts/vqa
bash train_vqa_distributed.sh 
</pre>
    <p>
        In our experiments, the finetuning costs around 36 hours (for 12 epochs). After each epoch, an evaluation on validation set is performed. The log is saved in <code>${log_dir}</code>.
    </p>
    <p>
        <i>(Update on validation time-cost)</i> As will be mentioned in the <i>4. Inference</i> section, we prepare 2 types of inference: beam-search and all-candidate inference. By default, all-candidate inference is used for validation during fine-tuning, which achieves better accuracy but costs much time. Now we have added a new option in the training scripts called <code>--val-inference-type</code> to switch the validation inference type during fine-tuning. If you feel the validation takes too long, you can refer to <a href="https://github.com/OFA-Sys/OFA/pull/79">PR #79</a> to activate beam-search validation, which significantly takes much less time, with around 0.5-0.6 validation score degradation compared with all-candidate validation.
    </p>
</details>
<details>
    <summary><b>4. Inference</b></summary>
    <p>
        We provide 2 types of inference, <b>beam-search</b> (much faster but gets sub-optimal accuracy) and <b>all-candidate evaluation</b> (slower but best accuracy). <br></br>
        For beam-search inference, use the script <code>evaluate_vqa_beam.sh</code>. Refer to the command below. The inference on test set costs around 16 GPU hours. After inference on test set, the result JSON file will be dumped in the <code>${result_path}</code> defined in the shell script. You can submit the result <code>test_predict.json</code> to <a href="https://eval.ai/web/challenges/challenge-page/830/overview">EvalAI</a>. Using our released finetuned checkpoint.
    </p>
<pre>
cd run_scripts/vqa
bash evaluate_vqa_beam.sh val # specify 'val' or 'test'
</pre>
    <p>
        For all-candidate evaluation, we recommend to use the distributed script <code>evaluate_vqa_allcand_distributed.sh</code>. Please refer to the guide in the script to set the distributed configs before running. The result JSON file will be dumped in the <code>${result_path}</code> defined in the shell script of rank-0 server. All-candidate evaluation computes scores on all the candidate answers in the VQA dataset, reproducing our reported results in the paper.
    </p>
<pre>
# run on each worker after the distributed configs have been correctly set following the guide in evaluate_vqa_allcand_distributed.sh
cd run_scripts/vqa
bash evaluate_vqa_allcand_distributed.sh val # specify 'val' or 'test'
</pre>
</details>

