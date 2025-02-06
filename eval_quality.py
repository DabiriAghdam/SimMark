'''partially adapted from SemStamp'''
# By the courtesy of the authors of 
# "SemStamp: A Semantic Watermark with Paraphrastic Robustness for Text Generation "
# https://arxiv.org/abs/2310.03991
# "k-SemStamp: A Clustering-Based Semantic Watermark for Detection of Machine-Generated Text"
# https://arxiv.org/abs/2402.11399
# original github link: https://github.com/bohanhou14/semstamp

import argparse
from datasets import load_from_disk
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from eval_clm import eval_perplexity, text_entropy, rep_ngram
from collections import Counter
import os
import faiss
import pickle
import torch
from functools import partial
from tqdm import trange as std_trange
trange = partial(std_trange, dynamic_ncols=True)

def tokenize_untokenize(text, tokenizer):
    input_ids = tokenizer(text, return_tensors="pt").input_ids[0]
    tokens = []
    for id in input_ids:
        tokens.append(tokenizer.decode(id, skip_special_tokens=True))
    return tokens

def run_entropy(gens, tokenizer):
    gen_sen_lis = [tokenize_untokenize(g, tokenizer) for g in gens]
    gen_entros = []
    for k in [2, 3]:
        gen_entro = text_entropy(gen_sen_lis, k=k)
        gen_entros.append(gen_entro)
    return gen_entros

def run_ngrams(gens, tokenizer):
    gen_sen_lis = [tokenize_untokenize(g, tokenizer) for g in gens]
    rep_scores = []
    for k in [2,3,4]:
        rep_score = rep_ngram(gen_sen_lis, k)
        rep_scores.append(rep_score)
    return rep_scores

def run_sem_ent(model, gens, tokenizer, ref_texts, args):
    # Step 0: parse args
    load_kmeans_path = None 
    save_kmeans_path = args.human_ref_name
    load_testgen_path = None 
    save_testgen_path = args.dataset_name
    max_new_token = args.max_new_token
    mode = args.sem_ent_mode 
    cluster_size = args.cluster_size

    # Step 1: Perform clustering (if no centroids provided)
    def batch_tokenize(texts):
        input_ids = tokenizer(texts, return_tensors="pt", max_length = max_new_token, truncation=True, padding="longest").input_ids.to("cuda")
        return input_ids.unsqueeze(1) # shape: num of text X longest X embed dim 

    def embed_gen(texts, desc="Featurizing", mode='last_token', cluster_size = 50):
        input_ids = batch_tokenize(texts)
        sent_len = [len(tokenizer(text, return_tensors="pt", truncation= True, max_length = max_new_token).input_ids[0]) for text in texts]
        out_embeds = []
        if mode == 'last_token':
            out_embeds = [model(input_ids[i], output_hidden_states=True).hidden_states[-1][0][sent_len[i] - 1].cpu().detach().squeeze() for i in trange(len(input_ids), desc = desc)]
        elif mode == 'last_mean_pooling':
            for i in trange(len(input_ids), desc = desc):
                out_embeds.append(torch.mean([model(input_ids[i], output_hidden_states=True).hidden_states[-1][0][j].cpu().detach().squeeze()]) for j in range(sent_len))
        elif mode == 'all_mean_pooling':
            for i in trange(len(input_ids), desc = desc):
                hidden_states = model(input_ids[i], output_hidden_states=True).hidden_states
                hidden_states_len = len(hidden_states) 
                hidden_states_list = [hidden_states[k][0][sent_len[i]-1].cpu().detach().squeeze() for k in range(hidden_states_len)]
                pooled_states = torch.mean(torch.stack(hidden_states_list), dim=0)
                out_embeds.append(pooled_states)
        input_tensor = torch.stack(out_embeds).numpy()
        return input_tensor

    # build from scratch if no stored index and centroids
    
    if load_kmeans_path == None:
        ncentroids = cluster_size
        niter = 100
        verbose = True
        input_tensor = embed_gen(ref_texts, desc="Featurizing train generations", mode=mode)
        
        d = input_tensor.shape[1]
        kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
        kmeans.train(input_tensor)
        centroids = kmeans.centroids
        index = kmeans.index
        # save
        cpath = os.path.join(save_kmeans_path, f"mode={mode}-cluster_size={cluster_size}-centroids.npy")
        ipath = os.path.join(save_kmeans_path, f"mode={mode}-cluster_size={cluster_size}-index.pkl")
        with open(cpath, "wb") as f:
            np.save(f, centroids)
            f.close()
        with open(ipath, "wb") as f:
            pickle.dump(index, f)
            f.close()
    # Otherwise, load saved index and centroids
    else:
        cpath = os.path.join(load_kmeans_path, f"mode={mode}-cluster_size={cluster_size}-centroids.npy")
        ipath = os.path.join(load_kmeans_path, f"mode={mode}-cluster_size={cluster_size}-index.pkl")
        centroids = np.load(cpath)
        with open(ipath, "rb") as f:
            index = pickle.load(f)
            f.close()
                
    # Step 2: Generate and assign clusters
    if load_testgen_path == None:
        gen_embed = embed_gen(gens, "Featurizing test generations", mode=mode)
        # save
        gen_embed_path = os.path.join(save_testgen_path, "test-gen-embed.pkl")
        with open(gen_embed_path, "wb") as f:
            pickle.dump(gen_embed, f)
            f.close()
    else:
        gen_embed_path = os.path.join(load_testgen_path, "test-gen-embed.pkl")
        with open(gen_embed_path, "rb") as f:
            gen_embed = pickle.load(f)
            f.close()
    # find 1-closest cluster
    _, I = index.search(gen_embed, 1)
    
    # Step 3: Calculate entropy
    freq_dict = Counter(list(I.squeeze()))
    freqs = list(freq_dict.values())
    total_freqs = np.sum(freqs)
    approx_p = [f / total_freqs for f in freqs]
    sem_ent = np.sum([-1 * p * np.log(p) for p in approx_p])
    with open(os.path.join(save_testgen_path, f'mode={mode}-cluster_size={cluster_size}-distribution.txt'), 'wb') as f:
        print(f"Semantic Ent: {sem_ent}")
        f.close()
    return sem_ent
    
def eval_quality(model, gens, ref_texts, tokenizer):
    
    if (type(gens[0]) == list):
        gens_text = [" ".join(g) for g in gens]
    else:
        gens_text = gens
    print("Evaluating perplexity...")
    gen_ppl = eval_perplexity(model, tokenizer, gens_text)

    print("Evaluating semantic entropy")
    sem_ent = run_sem_ent(model, gens, tokenizer, ref_texts)
    
    print("Evaluating n-gram repetition...")
    rep_scores = run_ngrams(gens_text, tokenizer)

    print("Evaluating entropy...")
    gen_entros= run_entropy(gens_text, tokenizer)

    return gen_ppl, gen_entros[0], gen_entros[1], rep_scores[0], rep_scores[1], rep_scores[2], sem_ent

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--human_ref_name', type=str)
    parser.add_argument('--max_new_token', type=int, default=205)
    parser.add_argument('--model_path', type=str, default='facebook/opt-2.7b')
    parser.add_argument('--cluster_size', type=int, default=50)
    parser.add_argument('--sem_ent_mode', type=str, choices = ['last_token', 'last_mean_pooling', 'all_mean_pooling'], default='last_token')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    gens = load_from_disk(args.dataset_name)['text']
    if args.human_ref_name != None:
        gen_ref_texts = load_from_disk(args.human_ref_name)['text']
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    pad_id = tokenizer.encode(tokenizer.eos_token)[0]
    model = AutoModelForCausalLM.from_pretrained(args.model_path, return_dict=True, pad_token_id=pad_id).to('cuda')
    model.eval()
    if (type(gens[0]) == list):
        gens_text = [" ".join(g) for g in gens]
    else:
        gens_text = gens

    print("Evaluating perplexity...")
    gen_ppl = eval_perplexity(model, tokenizer, gens_text)

    print("Evaluating semantic entropy")
    sem_ent = run_sem_ent(model, gens_text, tokenizer, gen_ref_texts, args)
    
    print("Evaluating entropy...")
    gen_entros = run_entropy(gens_text, tokenizer)

    print(f"gen_ppl:{gen_ppl:.2f}\ttri_entro:{gen_entros[1]:.2f}\tsem_ent:{sem_ent:.2f}")
        