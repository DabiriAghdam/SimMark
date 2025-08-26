# <p align="center">_**SimMark**_: A Robust Sentence-Level Similarity-Based Watermarking Algorithm for Large Language Models</p>

<p align="center">
  <br>
  <a href="https://arxiv.org/abs/2502.02787"><img alt="Paper" src="https://img.shields.io/badge/📃-Paper-808080"></a>
  <a href="https://simmark-llm.github.io/"><img alt="Website" src="https://img.shields.io/badge/%F0%9F%8C%90-Website-008080"></a>
</p>

This is the official repo for  <a href="https://arxiv.org/abs/2502.02787">_SimMark_:A Robust Sentence-Level Similarity-Based Watermarking Algorithm for Large Language Models</a> (accepted to EMNLP 2025).

## Abstract
The widespread adoption of large language models (LLMs) necessitates reliable methods to detect LLM-generated text. We introduce **_SimMark_**, a robust sentence-level watermarking algorithm that makes LLMs' outputs traceable without requiring access to model internals, making it compatible with both open and API-based LLMs. By leveraging the similarity of semantic sentence embeddings combined with rejection sampling to embed detectable statistical patterns imperceptible to humans, and employing a *soft* counting mechanism, *SimMark* achieves robustness against paraphrasing attacks. Experimental results demonstrate that *SimMark* sets a new benchmark for robust watermarking of LLM-generated content, surpassing prior sentence-level watermarking techniques in robustness, sampling efficiency, and applicability across diverse domains, all while maintaining the text quality and fluency.

<div align="center">
  <img src="https://github.com/user-attachments/assets/1550a812-fd1b-49a4-bb38-8a63b207773b" alt="A high-level overview of SimMark detection algorithm" width="72%">
</div>

A high-level overview of *SimMark* detection algorithm. The input text is divided into individual sentences $`X_1`$ to $`X_N`$, which are embedded using a semantic embedding model. The similarity between consecutive sentence embeddings is computed. Sentences with similarities within a predefined interval $`[a, b]`$ are considered $`\color{ForestGreen}\textbf{valid}`$, while those outside are $`\textcolor{BrickRed}{\textbf{invalid}}`$.  A statistical test is performed using the count of $`\textcolor{ForestGreen}{\textbf{valid}}`$ sentences to determine whether the text is watermarked.

## Overview of SimMark: A Similarity-Based Watermark
<div align="center">
  <img src="https://github.com/user-attachments/assets/f7d78bc0-dd7e-4dbf-a0ae-f6b79313fe80" alt="Overview of SimMark: A Similarity-Based Watermark">
</div>

***Top: Generation***. For each newly generated sentence ($`X_{i+1}`$), its embedding ($`e_{i+1}`$) is computed using a semantic text embedding model, optionally applying PCA for dimensionality reduction. The cosine similarity (or Euclidean distance) between $`e_{i+1}`$ and the embedding of the previous sentence ($`e_i`$), denoted as $`s_{i+1}`$, is calculated. If $`s_{i+1}`$ lies within the predefined interval $`[a, b]`$, the sentence is marked $`\color{ForestGreen}\textbf{valid}`$ and accepted. Otherwise, rejection sampling generates a new candidate sentence until validity is achieved or the iteration limit is reached. Once a sentence is accepted, the process repeats for subsequent sentences.  

***Bottom: Detection (+ Paraphrase attack)***. Paraphrased versions of watermarked sentences are generated ($`Y_{i}`$), and their embeddings ($`e'_{i}`$) are computed. The similarity between consecutive sentences in the paraphrased text is evaluated. If paraphrasing causes the similarity ($`s'_{i+1}`$) to fall outside $`[a, b]`$, it is mismarked as $`\textcolor{BrickRed}{\textbf{invalid}}`$. A *soft counting* mechanism (via function $`c(s_{i+1})`$ instead of a regular counting with a step function in the interval $`[a, b]`$) quantifies partial validity based on proximity to the interval bounds, enabling detection of watermarked text via the ***soft***-$`z`$-test even under paraphrase attacks. It should be noted that soft counting is always applied, as we cannot assume prior knowledge of paraphrasing.

## Detection Performance
<div align="center">
  <img src="https://github.com/user-attachments/assets/5f914fd6-888e-4c27-a73e-59890d712d1e" alt="Performance of different algorithms across datasets and paraphrasers">
</div>

Performance of different algorithms across datasets and paraphrasers, evaluated using <b>ROC-AUC ↑</b> / <b>TP@FP=1% ↑</b> / <b>TP@FP=5% ↑</b>, respectively (↑: higher is better), reported from left to right. In each column, <b>bold</b> values indicate the best performance for a given dataset and metric, while <u>underlined</u> values denote the second-best. **<i>SimMark</i> consistently outperforms or is on par with other state-of-the-art methods across datasets and paraphrasers, and it is the best on average.**

<div align="center">
  <img src="https://github.com/user-attachments/assets/fda28ed4-b1d5-460b-898a-6a090991a6ba"" alt="Averaged detection performance of different watermarking methods under various paraphrasing attacks", width="66%">
</div>

Detection performance of different watermarking methods under various paraphrasing attacks, measured by TP@FP=1\% ↑ and averaged across all three datasets (RealNews, BookSum, Reddit-TIFU). Each axis corresponds to a specific paraphrasing attack method (e.g., Pegasus-Bigram), and higher values are better. Our methods, $`\color{Orange}{cosine-SimMark}`$ and $`\color{Emerald}{Euclidean-SimMark}`$, consistently outperform or match baselines across most paraphrasers, especially under more challenging conditions such as bigram-level paraphrasing.

## How To Run?
### Installation
1. Clone this repository:
```
git clone https://github.com/DabiriAghdam/SimMark.git
```
2. Create a virtual environment (python3.10 recommended), and then install dependencies: 
```
    pip3 install -r requirements.txt
```
3. Then run following commands to download the necessary data:
```
    python3 load_punkt_tab.py
    python3 load_c4.py
    python3 load_booksum.py
    python3 load_tifu.py
```
### Reproducing our results (requires GPU)
#### For RealNews dataset as an example you can run the following:
1. **Without paraphrase**:
  - _Cosine-SimMark_:  
  ```
  python3 detection.py watermarked/c4/c4-cosine  --human_text human/c4  --mode cosine --a 0.68 --b 0.76 
  ```
  - _Euclidean-SimMark_:  
  ```
  python3 detection.py watermarked/c4/c4-euclidean-pca  --human_text human/c4  --mode euclidean --a 0.28 --b 0.36 --use_pca
  ```
2. **With Pegasus paraphraser**:
  - _Cosine-SimMark_:   
  ```
  python3 detection.py watermarked/c4/c4-cosine-pegasus --human_text human/c4  --mode cosine --a 0.68 --b 0.76 
  ```
  - _Euclidean-SimMark_:  
  ```
  python3 detection.py watermarked/c4/c4-euclidean-pca-pegasus  --human_text human/c4  --mode euclidean --a 0.28 --b 0.36 --use_pca    
  ```  
3. Additional paraphrasers data can be used similarly by changing the dataset path.

  #### For other datasets, just replace the dataset paths and make sure you set the correct parameters for "human_text", "mode", "a", "b", and if necessary add "--use_pca" flag.
4. For example for the BookSum dataset with Pegasus paraphraser:
  - _Cosine-SimMark_:   
  ```
  python3 detection.py watermarked/booksum/booksum-cosine-pegasus --human_text human/booksum  --mode cosine --a 0.68 --b 0.76
  ```
  - _Euclidean-SimMark_:  
  ```
  python3 detection.py watermarked/booksum/booksum-euclidean-pegasus --human_text human/booksum  --mode euclidean --a 0.4 --b 0.55
  ```
### Hyperparameters
The key parameters that can be used with detection.py are summarized as follows (for more details, see the code itself):
- Data path: See the watermarked folder.
- Mode (--mode): 'cosine' or 'euclidean'
- Predefined Intervals ([a, b]):
  - RealNews dataset (human_text should be human/c4):
    - Cosine Similarity: [0.68, 0.76]
    - Euclidean Distance: [0.28, 0.36] (must add --use_pca flag)
  - BookSum dataset (human_text should be human/booksum):
    - Cosine Similarity: [0.68, 0.76]
    - Euclidean Distance: [0.4, 0.55] (**DO NOT** add --use_pca flag for this dataset)
  - Reddit-TIFU dataset (human_text should be human/tifu):
      - Cosine Similarity: [0.68, 0.76]
      - Euclidean Distance: [0.28, 0.36] (must add --use_pca flag)
- Soft Count Decay Factor (--K): 250 is the default.
- LLM (--model_path): 'AbeHou/opt-1.3b-semstamp' is the default.
- Embedding Model (--embedder): 'hkunlp/instructor-large' is the default.

For further details on hyperparameter selection, refer to the paper.

### Text Quality Evaluation
For text quality evaluation, you can run the following command for Euclidean-SimMark on the BookSum dataset for instance:
```
python3 eval_quality.py --dataset_name watermarked/booksum/booksum-euclidean --human_ref_name human/booksum
```
### Generating New Watermarked Text (requires GPU)
  1. (Optional) If you want you can re-train a PCA model by running the following (first ensure the necessary data is loaded using ```load_c4.py```):
  ```
  python3 train_PCA.py --num_components 16 
  ```
  2. Generate a smaller subset of RealNews dataset (for example a subset of size n = 1000):
  ```
  python3 build_subset.py data/c4-val --n 1000
  ```
  3. Then to generate watermarked data run the following, with the appropriate parameters for ```human_text```, ```mode```, ```a```, ```b```, ```output```:   
        - _Cosine-SimMark_: 
        ```
        python3 sampling.py data/c4-val-1000  --output c4-cosine-new  --a 0.68 --b 0.76 --mode cosine 
        ```
        - _Euclidean-SimMark_:
        ```
        python3 sampling.py data/c4-val-1000  --output c4-euclidean-new  --a 0.28 --b 0.36 --mode euclidean --use_pca
        ```
  4.(Optional) If you want to apply paraphrasing you can run this for Pegasus-bigram paraphraser, for example with 25 beams:
  ```
  python3 paraphrase_gen.py c4-cosine-new --paraphraser pegasus-bigram --num_beams 25
  ```
  _You can replace ```pegasus-bigram``` with other available paraphrasers._
  
  5. You can then detect watermarks (before and after paraphrasing):
  ```
  python3 detection.py c4-cosine-new --human_text human/c4  --mode cosine --a 0.68 --b 0.76
  python3 detection.py c4-cosine-new-pegasus-bigram=True-num_beams=25-threshold=0.1  --human_text human/c4  --mode cosine --a 0.68 --b 0.76
  ```


#### Acknowledgement: 
Some of the codes were partially adapted from these work (original github link: https://github.com/bohanhou14/semstamp):

  - "SemStamp: A Semantic Watermark with Paraphrastic Robustness for Text Generation"  (https://arxiv.org/abs/2310.03991)

  - "k-SemStamp: A Clustering-Based Semantic Watermark for Detection of Machine-Generated Text"  (https://arxiv.org/abs/2402.11399)
  
## Citation
If you found this repository helpful, please don't forget to cite our paper:
```BibTeX
@misc{dabiriaghdam2025simmarkrobustsentencelevelsimilaritybased,
      title={SimMark: A Robust Sentence-Level Similarity-Based Watermarking Algorithm for Large Language Models}, 
      author={Amirhossein Dabiriaghdam and Lele Wang},
      year={2025},
      eprint={2502.02787},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.02787}, 
}
```
If you have any questions, please feel free to contact  [amirhossein@ece.ubc.ca](mailto:amirhossein@ece.ubc.ca).
