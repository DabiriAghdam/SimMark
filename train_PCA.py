from sklearn.decomposition import PCA
from datasets import load_from_disk
from nltk.tokenize import sent_tokenize
import pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
import argparse

parser = argparse.ArgumentParser(description='Train PCA on sentence embeddings.')
parser.add_argument('--num_components', type=int, default=16, help='Number of PCA components')
parser.add_argument('--train_texts_path', type=str, default='data/c4-train', help='Path to training texts')
parser.add_argument('--embedder', type=str, default='hkunlp/instructor-large', help='Sentence transformer model name')
parser.add_argument('--instruction', type=str, default='Represent the sentence for PCA:', help='Instruction for sentence transformer model')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size for generating embeddings')
args = parser.parse_args()

train_texts = load_from_disk(args.train_texts_path)['text']
model = SentenceTransformer(args.embedder)
instruction = args.instruction
inst_sents = list()
for text in tqdm(train_texts):
    sents = sent_tokenize(text)
    for sent in sents:
        inst_sents.append(sent)

print("Generating embeddings...")
embeddings = model.encode(inst_sents, prompt=instruction, batch_size=args.batch_size, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

# Normalize embeddings
scaler = StandardScaler()
embeddings = scaler.fit_transform(embeddings)

num_components = args.num_components
pca = PCA(n_components=num_components)
pca.fit(embeddings)

pca_model_filename = f'pca_model_{num_components}.pkl'
with open(pca_model_filename, 'wb') as f:
    pickle.dump(pca, f)
print(f"Saved PCA model to {pca_model_filename}")