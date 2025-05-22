'''partially adapted from SemStamp'''
# By the courtesy of the authors of 
# "SemStamp: A Semantic Watermark with Paraphrastic Robustness for Text Generation "
# https://arxiv.org/abs/2310.03991
# "k-SemStamp: A Clustering-Based Semantic Watermark for Detection of Machine-Generated Text"
# https://arxiv.org/abs/2402.11399
# original github link: https://github.com/bohanhou14/semstamp

import torch
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import StoppingCriteriaList, GenerationConfig, StoppingCriteria
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from string import punctuation
from itertools import groupby

MAX_TRIALS = 0
if torch.cuda.is_available():
    rng = torch.Generator("cuda")
else: 
    rng = torch.Generator("cpu")
PUNCTS = '!.?'
device = "cuda" if torch.cuda.is_available() else "cpu"


class DummyPCA:
    def transform(self, x):
        return x

class SentenceEndCriteria(StoppingCriteria):
    """
    ONLY WORK WITH BATCH SIZE 1

    Stop generation whenever the generated string is **more than one** sentence (i.e. one full sentence + one extra token). this is determined by nltk sent_tokenize.
    Only stop if ALL sentences in the batch is at least two sentences

    Args:
        tokenizer (PreTrainedTokenizer):
            The exact tokenizer used for generation. MUST BE THE SAME!
    """

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.current_num_sentences = 0

    def update(self, current_text):
        self.current_num_sentences = len(sent_tokenize(current_text))

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        assert input_ids.size(0) == 1
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return len(sent_tokenize(text)) > self.current_num_sentences + 1


def discard_final_token_in_outputs(outputs):
    outputs.sequences = outputs.sequences[:, :-1]  # (bz, seqlen)
    return outputs

def gen_sent(model, tokenizer, text_ids, gen_config, stopping_criteria):
    outputs = model.generate(
            # input_ids,
            text_ids,
            gen_config,
            stopping_criteria=stopping_criteria,
        )
    outputs = discard_final_token_in_outputs(outputs)
    new_text_ids = outputs.sequences
    new_text = tokenizer.decode(
        new_text_ids[0, text_ids.size(1):], skip_special_tokens=True)
    return new_text, new_text_ids


def well_formed_sentence(sent, end_sent=False):
    sent = first_upper(sent)
    sent = sent.replace('  ', ' ')
    sent = sent.replace(' i ', " I ")
    if end_sent and len(sent) > 0 and sent[-1] not in PUNCTS:
        sent += "."
    return clean_text(sent)

def clean_text(s):
    punc = set(punctuation) - set('.')
    punc.add("\n")
    newtext = []
    for k, g in groupby(s):
        if k in punc:
            newtext.append(k)
        else:
            newtext.extend(g)
    return ''.join(newtext)

def first_upper(s):
    if len(s) == 0:
        return s
    else:
        return s[0].upper() + s[1:]

def cosine_reject_completion(
        prompt: str,
        model: PreTrainedModel, tokenizer: PreTrainedTokenizer, gen_config: GenerationConfig,  # gen args
        embedder: SentenceTransformer,
        whole_interval: tuple,
        pca_model,
        skip_count: int = 1,
        alt_sent: bool = False,
        verbose: bool =False,
        device: str ='cuda',
        **kwargs):

    if isinstance(pca_model, DummyPCA):
        instruction = "Represent the sentence for cosine similarity:"
    else:
        instruction = "Represent the sentence for PCA:"

    sent_end_criteria = SentenceEndCriteria(tokenizer)
    text = prompt
    new_text = prompt
    text_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    prompt_length = len(text_ids[0])
    sent_end_criteria.update(new_text)

    total_trials = 0
    success_trials = 0  # also include trials that maxed out MAX_TRIALS
    current_trials = 0
    maxedout_trials = 0
    old_text = None #prompt
    all_texts = list()
    all_texts.append(prompt)
    old_interval = None
    i = 1
    if (i - skip_count >= 0): old_text = all_texts[i - skip_count]
    while (True):
        stopping_criteria = StoppingCriteriaList([sent_end_criteria])
        new_text, new_text_ids = gen_sent(model = model, 
                tokenizer = tokenizer, 
                text_ids = text_ids,
                gen_config = gen_config,
                stopping_criteria = stopping_criteria
            )
        if new_text == '':
            print('WARNING: stopped generation because generated nothing (after discarding last generated token)', flush=True)
            break

        total_trials += 1
        current_trials += 1

        if (alt_sent and i - skip_count >= 0 and i % 2 == 1):
            accepted_text, interval, cosine_sim = cosine_reject_overlap(prev_sent=None, sent=new_text, embedder=embedder, old_interval=old_interval, whole_interval=whole_interval, pca_model=pca_model, instruction=instruction) 
        else:
            accepted_text, interval, cosine_sim = cosine_reject_overlap(prev_sent=old_text, sent=new_text, embedder=embedder, old_interval=old_interval, whole_interval=whole_interval, pca_model=pca_model, instruction=instruction) 
        
        if (accepted_text == None and current_trials < MAX_TRIALS):
            old_interval = interval
            continue
        else:
            new_text = accepted_text if accepted_text != None else new_text
            if (verbose):
                print(old_text, "---", new_text, interval, cosine_sim)
            text += new_text
            text_ids = new_text_ids
            all_texts.append(new_text)
            i += 1
            old_text = None if i < skip_count else all_texts[i - skip_count] #new_text
            sent_end_criteria.update(text)
            old_interval = None
            if (current_trials >= MAX_TRIALS):
                print(
                    f'WARNING: desired semantic similarity can\'t be sampled after max_trials {MAX_TRIALS}', flush=True)
                maxedout_trials += 1
            
            success_trials += 1
            current_trials = 0

        if (len(text_ids[0]) - prompt_length) >= gen_config.max_new_tokens - 1:
            break

    if (verbose): print(total_trials, maxedout_trials)
    return text


def euclidean_reject_completion(
        prompt: str,
        model: PreTrainedModel, tokenizer: PreTrainedTokenizer, gen_config: GenerationConfig,  # gen args
        embedder: SentenceTransformer,
        whole_interval: tuple,
        pca_model,
        skip_count: int = 1,
        alt_sent: bool = False,
        verbose: bool = False,
        device: str ='cuda',
        **kwargs):

    if isinstance(pca_model, DummyPCA):
        instruction = "Represent the sentence for euclidean distance:"
    else:
        instruction = "Represent the sentence for PCA:"

    sent_end_criteria = SentenceEndCriteria(tokenizer)
    text = prompt
    new_text = prompt
    text_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    prompt_length = len(text_ids[0])
    sent_end_criteria.update(new_text)

    total_trials = 0
    success_trials = 0  # also include trials that maxed out sampling_utils.MAX_TRIALS
    current_trials = 0
    maxedout_trials = 0
    old_text = None #prompt
    all_texts = list()
    all_texts.append(prompt)
    old_interval = None
    i = 1
    if (i - skip_count >= 0): old_text = all_texts[i - skip_count]
    while True:
        stopping_criteria = StoppingCriteriaList([sent_end_criteria])
        new_text, new_text_ids = gen_sent(model = model, 
                tokenizer = tokenizer, 
                text_ids = text_ids,
                gen_config = gen_config,
                stopping_criteria = stopping_criteria
            )
        if new_text == '':
            print('WARNING: stopped generation because generated nothing (after discarding last generated token)', flush=True)
            break

        total_trials += 1
        current_trials += 1

        if (alt_sent and i - skip_count >= 0 and i % 2 == 1):
            accepted_text, interval, dist = euclidean_reject_overlap(prev_sent=None, sent=new_text, embedder=embedder, old_interval=old_interval, whole_interval=whole_interval, pca_model=pca_model, instruction=instruction)
        else:
            accepted_text, interval, dist = euclidean_reject_overlap(prev_sent=old_text, sent=new_text, embedder=embedder, old_interval=old_interval, whole_interval=whole_interval, pca_model=pca_model, instruction=instruction) 
      
        if (accepted_text == None and current_trials < MAX_TRIALS):
            old_interval = interval
            continue
        else:
            new_text = accepted_text if accepted_text != None else new_text
            if (verbose):
                print(old_text, "---", new_text, interval, dist)
            text += new_text
            text_ids = new_text_ids
            all_texts.append(new_text)
            i += 1
            old_text = None if i < skip_count else all_texts[i - skip_count] #new_text
            sent_end_criteria.update(text)
            old_interval = None
            if current_trials >= MAX_TRIALS:
                print(
                    f'WARNING: desired semantic distance can\'t be sampled after max_trials {MAX_TRIALS}', flush=True)
                maxedout_trials += 1
            
            success_trials += 1
            current_trials = 0

        if (len(text_ids[0]) - prompt_length) >= gen_config.max_new_tokens - 1:
            break

    if (verbose): print(total_trials, maxedout_trials)
    return text

def in_interval(intervals, dist, margin=0):
    for interval in intervals:
        if (interval[0] + margin / 2 <= dist <= interval[1] - margin / 2): return True
    return False

def euclidean_reject_overlap(prev_sent, sent, embedder, old_interval, whole_interval, pca_model, instruction):
    if (prev_sent is not None):
        sent_embed = embedder.encode(sent, prompt=instruction, convert_to_tensor=True, normalize_embeddings = True).reshape(1, -1)
        prev_sent_embed =  embedder.encode(prev_sent, prompt=instruction, convert_to_tensor=True, normalize_embeddings = True).reshape(1, -1)
        dist = euclidean_distances(pca_model.transform(prev_sent_embed.cpu().detach().numpy()), pca_model.transform(sent_embed.cpu().detach().numpy()))[0, 0]
        if (old_interval is None):
            interval = [(whole_interval[0], whole_interval[1])]
        else:
            interval = old_interval
        if (in_interval(interval, dist, margin=0)):
            return sent, interval, dist
        else:
            return None, interval, dist
    else:
        return sent, None, None

def cosine_reject_overlap(prev_sent, sent, embedder, old_interval, whole_interval, pca_model, instruction):
    if (prev_sent is not None):
        sent_embed = embedder.encode(sent, prompt=instruction, convert_to_tensor=True, normalize_embeddings = True).reshape(1, -1)
        prev_sent_embed =  embedder.encode(prev_sent, prompt=instruction, convert_to_tensor=True, normalize_embeddings = True).reshape(1, -1)
        cosine_sim = cosine_similarity(pca_model.transform(prev_sent_embed.cpu().detach().numpy()), pca_model.transform(sent_embed.cpu().detach().numpy()))[0, 0]

        if (old_interval is None):
            interval =  [(whole_interval[0], whole_interval[1])]
        else:
            interval = old_interval
        if (in_interval(interval, cosine_sim, margin=0)):
            return sent, interval, cosine_sim
        else:
            return None, interval, cosine_sim
    else:
        return sent, None, None