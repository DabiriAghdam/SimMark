'''Partially adapted from SemStamp'''
# By the courtesy of the authors of 
# "SemStamp: A Semantic Watermark with Paraphrastic Robustness for Text Generation "
# https://arxiv.org/abs/2310.03991
# "k-SemStamp: A Clustering-Based Semantic Watermark for Detection of Machine-Generated Text"
# https://arxiv.org/abs/2402.11399
# original github link: https://github.com/bohanhou14/semstamp

import argparse
from datasets import load_from_disk, Dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', help='hf dataset containing text column')
    parser.add_argument('--n', help='number of texts to keep', default=1000, type=int)
    args = parser.parse_args()
    dataset = load_from_disk(args.dataset_path)
    texts = dataset['text'][:args.n] # Change to 'documents' if using tifu dataset or 'summary_text' if using BookSum dataset!
    Dataset.from_dict({'text': texts}).save_to_disk( args.dataset_path + f'-{args.n}')

