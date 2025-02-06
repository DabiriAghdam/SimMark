'''partially adapted from SemStamp'''
# By the courtesy of the authors of 
# "SemStamp: A Semantic Watermark with Paraphrastic Robustness for Text Generation"
# https://arxiv.org/abs/2310.03991
# "k-SemStamp: A Clustering-Based Semantic Watermark for Detection of Machine-Generated Text"
# https://arxiv.org/abs/2402.11399
# original github link: https://github.com/bohanhou14/semstamp

import matplotlib
matplotlib.use('Agg')

from datasets import load_from_disk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np
from functools import partial
from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)
import argparse
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import roc_curve, auc
import pickle

class DummyPCA:
    def transform(self, x):
        return x

def in_interval(intervals, dist, margin=0):
    for interval in intervals:
        if (interval[0] + margin / 2 <= dist <= interval[1] - margin / 2): return True
    return False

def get_roc_metrics(labels, preds):
    fpr, tpr, _ = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), float(roc_auc)

def get_roc_metrics_from_zscores(mp, h):
    mp = np.nan_to_num(mp)
    h = np.nan_to_num(h)
    len_z = len(mp)
    mp_fpr, _, mp_area = get_roc_metrics(
        [1] * len_z + [0] * len_z, np.concatenate((mp, h[:len_z])))
    return mp_area, mp_fpr

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', help='hf dataset containing text and para_text columns')
    parser.add_argument('--embedder', type=str, help='sentence embedder model', default='hkunlp/instructor-large')
    parser.add_argument('--human_text', type=str, help='hf dataset containing text column')
    parser.add_argument('--a', type=float, default=0.5)
    parser.add_argument('--b', type=float, default=1)
    parser.add_argument('--mode', type=str, choices=['cosine', 'euclidean'])
    parser.add_argument('--use_pca', action='store_true', default=False, help='When True, use PCA for dimensionality reduction')
    parser.add_argument('--K', type=int, default=250, help='Scaling factor for distance to interval')
    parser.add_argument('--model_path', type=str, default='AbeHou/opt-1.3b-semstamp')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()
    return args

def distance_to_intervals(intervals, x):
    min_dist = float('inf')
    for interval in intervals: # for cases where the interval is split into more than one
        start, end = interval[0], interval[1]
        if (start <= x <= end):
            return 0 
        dist_start = abs(x - start)
        dist_end = abs(x - end)
        min_dist = min(min_dist, dist_start, dist_end)
    
    return min_dist

def calculate_embeddings(all_sents, embedder, instruction, fix_space=True):
    aggr_sents = []
    sent_boundaries = [0]
    for text in all_sents:
        sents = text if type(text) == list else sent_tokenize(text)
        aggr_sents.extend(sents)
        sent_boundaries.append(sent_boundaries[-1] + len(sents))

    preprocessed_sents = list()
    for i in range(len(aggr_sents)):
        sent = aggr_sents[i]
        is_first_in_group = i in sent_boundaries[:-1]
        if (fix_space and not is_first_in_group): preprocessed_sents.append(" " + sent)
        else: preprocessed_sents.append(sent)

    all_embeddings = embedder.encode(preprocessed_sents, prompt=instruction, batch_size=256, show_progress_bar=True, convert_to_tensor=True, normalize_embeddings=True)

    return [all_embeddings[start:end] for start, end in zip(sent_boundaries[:-1], sent_boundaries[1:])]

def calc_dist(sents, mode, embeddings, pca_model, skip_count=1):
    dists = list()
    for j in range(skip_count, len(sents)):
        index1 = j - skip_count
        index2 = j
        embedding1 = embeddings[index1].reshape(1, -1).cpu().detach().numpy()
        embedding2 = embeddings[index2].reshape(1, -1).cpu().detach().numpy()
        if ('cosine' in mode):
            dist = cosine_similarity(pca_model.transform(embedding1), pca_model.transform(embedding2))[0, 0] 
        elif ('euclidean' in mode):
            dist = euclidean_distances(pca_model.transform(embedding1), pca_model.transform(embedding2))[0, 0] 
        else:
            raise NotImplementedError
        dists.append(dist)

    return dists

def detect(sents, mode, embeddings, pca_model, gamma, skip_count=1, alt_sent=False, fix_space=True, whole_interval=(0.5, 1), verbose=False, K=250):
    n_sent = len(sents)
    n_watermark = 0
    for j in range(skip_count, len(sents)):
        index1 = j - skip_count
        sent1 = sents[index1]
        if (j > skip_count and fix_space): sent1 = " " + sent1
        index2 = j
        if (fix_space): sent2 = " " + sents[index2]
        else: sent2 = sents[index2]

        if (alt_sent and index1 >= 0 and index2 % 2 == 1):
            n_sent -= 1
            if (verbose):
                print(sent1, "---", sent2, None, None)
            continue

        embedding1 = embeddings[index1].reshape(1, -1).cpu().detach().numpy()
        embedding2 = embeddings[index2].reshape(1, -1).cpu().detach().numpy()
        interval = [(whole_interval[0], whole_interval[1])]
        if ('cosine' in mode):
            dist = cosine_similarity(pca_model.transform(embedding1), pca_model.transform(embedding2))[0, 0] 
        elif ('euclidean' in mode):
            dist = euclidean_distances(pca_model.transform(embedding1), pca_model.transform(embedding2))[0, 0] 
        else:
            raise NotImplementedError

        dist_to_interval = distance_to_intervals(interval, dist)
        n_watermark += np.exp(-K*dist_to_interval)
        if (verbose): 
            print(sent1, "---", sent2, in_interval(interval, dist), np.exp(-K*dist_to_interval))
            
    n_test_sent = n_sent - skip_count  # exclude the prompt
    num = n_watermark - gamma * (n_test_sent)
    denom = np.sqrt((n_test_sent) * gamma * (1-gamma) + 1e-12) # 1e-12 is for numerical stability
    if (verbose): 
        print(f'n_watermark: {n_watermark}, n_test_sent: {n_test_sent}', num / denom)
    return num / denom

def evaluate_z_scores(hz):
    hz = np.array(hz)
    fpr_5_threshold = 0
    fpr_1_threshold = 0
    for z_threshold in np.arange(10, -10, -0.001): 
        fp = len(hz[hz > z_threshold]) / len(hz)
        if (fp >= 0.0095 and fp <= 0.0104):
            fpr_1_threshold = z_threshold
        elif (fp >= 0.045 and fp <= 0.054):
            fpr_5_threshold = z_threshold
    if fpr_1_threshold == 0:
        fpr_1_threshold = 2.33 # according to standard z-score table
    print(fpr_1_threshold, ', ', fpr_5_threshold, sep='')
    return fpr_1_threshold, fpr_5_threshold

if __name__ == '__main__':
    args = parse_args()
    embedder = SentenceTransformer(args.embedder)
    if (args.use_pca):
        pca_model = pickle.load(open('pca_model_16.pkl', 'rb'))
    else:
        pca_model = DummyPCA() 
    whole_interval=(args.a, args.b)
    fpr_1_threshold, fpr_5_threshold = 2.33, 2.33
    if ('cosine' in args.mode):
        instruction = "Represent the sentence for cosine similarity:"
    elif ('euclidean' in args.mode):
        instruction = "Represent the sentence for euclidean distance:"

    if (args.use_pca):  
        instruction = "Represent the sentence for PCA:"
    

    human_texts = load_from_disk(args.human_text)['text']
    embeddings = calculate_embeddings(human_texts, embedder, instruction=instruction)

    # Calculate area under curve
    data = list()
    for i in range(len(human_texts)):
        sents = sent_tokenize(human_texts[i])
        dist = calc_dist(sents, args.mode, embeddings[i], pca_model)
        data.extend(dist)
    data = np.array(data)
    counts, bins = np.histogram(data, bins=1000)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    mask = (bin_centers >= args.a) & (bin_centers <= args.b)
    gamma = np.sum(counts[mask]) / len(data)  
    print(f"Area under curve of human data in range ({args.a}, {args.b}): {gamma:.3f}")

    hz_scores = list()
    for i in tqdm(range(len(human_texts))):
        sents = sent_tokenize(human_texts[i])
        z_score = detect(sents, args.mode, embeddings[i], pca_model, gamma=gamma, whole_interval=whole_interval, verbose=False, K=args.K)
        hz_scores.append(z_score)
    hz = np.array(hz_scores)

    fpr_1_threshold, fpr_5_threshold = evaluate_z_scores(hz)
    print('FP:', 100 * len(hz[(hz) > fpr_1_threshold]) / len(hz))
    
    watermark_texts = load_from_disk(args.dataset_path)
    if 'para_text' in watermark_texts.column_names:
        watermark_texts = watermark_texts['para_text']
    else:
        watermark_texts = watermark_texts['text']
    embeddings = calculate_embeddings(watermark_texts, embedder, instruction=instruction)
    z_scores = list()
    for i in tqdm(range(len(watermark_texts))):
        sents = watermark_texts[i] if type(watermark_texts[i]) == list else sent_tokenize(watermark_texts[i])
        z_score = detect(sents, args.mode, embeddings[i], pca_model, gamma=gamma, whole_interval=whole_interval, verbose=args.verbose, K=args.K)
        z_scores.append(z_score)

    z = np.array(z_scores)
    
    print('TP@FP=1:', 100 * len(z[(z) > fpr_1_threshold]) / len(z))
    print('TP@FP=5:', 100 * len(z[(z) > fpr_5_threshold]) / len(z))

    mp_area, mp_fpr = get_roc_metrics_from_zscores(z, hz)
    print('AUC:', mp_area)