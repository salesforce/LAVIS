## Plug-and-Play VQA: Zero-shot VQA by Conjoining Large Pretrained Models with Zero Training

<img src="pnp_vqa.png" width="700">

This is the code for <a href="https://arxiv.org/abs/2210.08773">PNP-VQA paper</a>.

### Demo
We include an interactive demo [Colab notebook](https://colab.research.google.com/github/salesforce/LAVIS/blob/main/projects/pnp-vqa/pnp_vqa.ipynb)
to show PNP-VQA inference workflow:
1. Image-question matching: compute the relevancy score of the image patches wrt the question.
2. Image captioning: generate question-guided captions based on the relevancy score.
3. Question answering: answer the question by using the captions.

### Visual Question Answering
Evaluate PNP-VQA<sub>base</sub> as following:

#### VQAv2
<pre> bash run_scripts/pnp-vqa/eval/eval_vqav2.sh </pre>

#### OK-VQA
<pre> bash run_scripts/pnp-vqa/eval/eval_okvqa.sh </pre>

#### GQA
<pre> bash run_scripts/pnp-vqa/eval/eval_gqa.sh </pre>

For PNP-VQA<sub>large</sub> and PNP-VQA<sub>3B</sub>, run the respective scripts denoted with ```large``` and ```3b```. 
We reduce the number of captions for PNP-VQA<sub>3B</sub> to prevent OOM using NVIDIA A100 40GB.

### Citation
If you find this code to be useful for your research, please consider citing.
<pre>
@misc{tiong2022pnpvqa,
      title={Plug-and-Play VQA: Zero-shot VQA by Conjoining Large Pretrained Models with Zero Training}, 
      author={Anthony Meng Huat Tiong, Junnan Li, Boyang Li, Silvio Savarese, and Steven C.H. Hoi},
      year={2022},
      eprint={2210.08773},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}</pre>
