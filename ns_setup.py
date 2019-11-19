"""Download and pre-process SQuAD with BERT.

Pre-processing code adapted from:
    > https://github.com/chrischute/squad

Author:
    Pranav Modi
"""

import numpy as np
import os
import spacy
import ujson as json
import urllib.request

from ns_args import get_setup_args
from codecs import open
from collections import Counter
from subprocess import run
from tqdm import tqdm
from zipfile import ZipFile
from transformers import BertTokenizer


def save(filename, obj, message=None):
    if message is not None:
        print(f"Saving {message}...")
        with open(filename, "w") as fh:
            json.dump(obj, fh)


def download_url(url, output_path, show_progress=True):
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    if show_progress:
        # Download with a progress bar
        with DownloadProgressBar(unit='B', unit_scale=True,
                                 miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url,
                                       filename=output_path,
                                       reporthook=t.update_to)
    else:
        # Simple download with no progress bar
        urllib.request.urlretrieve(url, output_path)


def url_to_data_path(url):
    return os.path.join('./data/', url.split('/')[-1])


def download(args):
    downloads = [
        # Can add other downloads here (e.g., other word vectors)
        ('GloVe word vectors', args.glove_url),
    ]

    for name, url in downloads:
        output_path = url_to_data_path(url)
        if not os.path.exists(output_path):
            print(f'Downloading {name}...')
            download_url(url, output_path)

        if os.path.exists(output_path) and output_path.endswith('.zip'):
            extracted_path = output_path.replace('.zip', '')
            if not os.path.exists(extracted_path):
                print(f'Unzipping {name}...')
                with ZipFile(output_path, 'r') as zip_fh:
                    zip_fh.extractall(extracted_path)

    print('Downloading spacy language model...')
    run(['python', '-m', 'spacy', 'download', 'en'])

def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]


def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print(f"Token {token} cannot be found")
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def is_answerable(example):
    return len(example['y2s']) > 0 and len(example['y1s']) > 0


def save(filename, obj, message=None):
    if message is not None:
        print(f"Saving {message}...")
        with open(filename, "w") as fh:
            json.dump(obj, fh)



def batchify_indices(indices):
    """ Return padded indices and a attention mask
    """
    lengths = [len(i) for i in indices]
    att_mask = []
    mlen = max(lengths)
    for sent in indices:
        att_s = [1]*len(sent) + [0]*(mlen - len(sent))
        att_mask.append(att_s)
        sent += [0]*(mlen - len(sent))

    batch = torch.tensor(indices)
    att_mask = torch.tensor(att_mask)
    return batch, att_mask


def map_tokens_to_context(context, tokenizer):
    context_tokens = []
    tokens_to_context_map = {}
    ctoks = context.split(' ')
    current = 0
    tnum = 0
    till = 0
    for ct in ctoks:
        if ct == '':
            current += 1
            continue
        till = current + len(ct)
        wtoks = tokenizer.tokenize(ct)
        context_tokens.extend(wtoks)
        for i, w in enumerate(wtoks):
            tokens_to_context_map[tnum + i] = (current, till)
        tnum += len(wtoks)
        current = till + 1

    assert len(context_tokens) == len(tokens_to_context_map.keys())
    return context_tokens, tokens_to_context_map



def process_file(filename, tokenizer, output_file, batch_size=64, cls='[CLS]', para_limit=384,
                 pad_token=0, text_limit=512, sep='[SEP]', output_dir='outputdir', eval_file=''):

    # process = psutil.Process(os.getpid())
    # print(process.memory_info().rss / (1024 * 1024))

    # import shutil
    # shutil.rmtree(output_dir)

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    def find_ans_span(qas_tokens, ans_tokens):
        """ Find the span of ans tokens in question tokens, 
            None if not found.
        """
        ans_len = len(ans_tokens)
        for i, qt in enumerate(qas_tokens):
            if qt == ans_tokens[0]:
                found = True
                for j, at in enumerate(ans_tokens):
                    if i + j >= len(qas_tokens):
                        print('overflow alert', i + j, qt, i)
                        break
                    if qas_tokens[i + j] != at:
                        found = False
                        break
                if found:
                    return (i, i+ans_len)

        return None


    with open(filename, "r") as fh:
        source = json.load(fh)
        print('the num of articles', len(source["data"]))

        qas_embeddings = []
        start_positions = []
        end_positions = []
        file_num = 0
        indices = []
        input_masks = []
        count = 0
        counts = []
        num_clipped = 0
        span_not_found = 0
        ids = []
        eval_examples = {}
        for article in tqdm(source["data"]):
            #print('number of paragraphs in article', len(article["paragraphs"]))
            for para in article["paragraphs"]:
                context = para['context']
                #context_tokens = tokenizer.tokenize(context)
                context_tokens, tokens_to_context_map = map_tokens_to_context(context, tokenizer)

                for qas in para['qas']:
                    id = qas['id']
                    question = qas['question']

                    question_tokens = tokenizer.tokenize(question)

                    max_context_len = text_limit - len(question_tokens) - 3

                    if len(context_tokens) > max_context_len:
                        context_tokens = context_tokens[:max_context_len]
                        num_clipped += 1

                    ## For every question in the para, tokenize and get bert indices
                    # qas_text = ' '.join([cls, context, sep, question])
                    # qas_tokens = tokenizer.tokenize(qas_text)

                    qas_tokens = [cls] + question_tokens + [sep] + context_tokens + [sep]
                    qas_indices = tokenizer.encode(qas_tokens)
                    ques_inds = tokenizer.encode(question_tokens)

                    ## If the length of context + question is less than text limit,
                    ## pad the indices with pad_token

                    pad_len = 0
                    input_mask = [1] * text_limit
                    if len(qas_indices) < text_limit:
                        pad_len = text_limit - len(qas_indices)
                        qas_indices += [pad_token] * pad_len
                        input_mask[:pad_len] = [0] * pad_len

                    #input_mask = [1]*len(qas_indices) + [0]*pad_len
                    
                    input_masks.append(input_mask)
                    indices.append(qas_indices)

                    assert len(input_mask) == len(qas_indices)

                    ## For each answer for the question, get the span and insert into list
                    if qas['is_impossible']:
                        #print('for file num ', file_num, ' the answer is impossible')
                        span = (-1, -1)

                    else:
                        ans = qas['answers'][0]
                        ans_text = ans['text']
                        ans_tokens = tokenizer.tokenize(ans_text)
                        start = ans['answer_start']
                        span = find_ans_span(context_tokens, ans_tokens)
                        if span is None:
                            span = (-1, -1)
                            span_not_found += 1

                    start_positions.append(span[0] + 1)
                    end_positions.append(span[1] + 1)
                    count += 1
                    ids.append(id)
                    counts.append(count)

                    eval_examples[str(count)] = {"context": context,
                                                 "context_length" : len(context_tokens),
                                                 "question": question,
                                                 "input_mask": input_mask,
                                                 "tokens_to_context_map" : tokens_to_context_map,
                                                 "spans": span,
                                                 "answers": ans,
                                                 "uuid": qas["id"]}
            #         if count >= 3:
            #             print('going to break after 1 count')
            #             break
            #     break
            # break

        out_file = os.path.join(output_dir, output_file)
        print('number of examples', len(indices))
        print('Saving the out file', out_file)
        np.savez(out_file,
                 qas_indices=np.array(indices),
                 input_mask=np.array(input_masks),
                 y1s=np.array(start_positions),
                 y2s=np.array(end_positions),
                 ids=np.array(counts))

        eval_file = os.path.join(output_dir, eval_file)
        save(eval_file, eval_examples, 'Saving eval file: ' + eval_file)

        print(f"Clipped: {num_clipped}")
        print(f"Span not found: {span_not_found}")



if __name__ == '__main__':
    # Get command-line args
    args_ = get_setup_args()


    # Preprocess dataset
    # args_.train_file = url_to_data_path(args_.train_url)
    # args_.dev_file = url_to_data_path(args_.dev_url)
    # if args_.include_test_examples:
    #     args_.test_file = url_to_data_path(args_.test_url)
    # glove_dir = url_to_data_path(args_.glove_url.replace('.zip', ''))
    # glove_ext = f'.txt' if glove_dir.endswith('d') else f'.{args_.glove_dim}d.txt'
    # args_.glove_file = os.path.join(glove_dir, os.path.basename(glove_dir) + glove_ext)
    #pre_process(args_)

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    train_file = 'data/train-v2.0.json'
    dev_file = 'data/dev-v2.0.json'

    process_file(dev_file, tokenizer, output_file='dev.npz', batch_size=64, cls='[CLS]',
                 para_limit=384, sep='[SEP]', output_dir='data', eval_file='dev_eval.json')

    process_file(train_file, tokenizer, output_file='train.npz', batch_size=64,
                 cls='[CLS]', para_limit=384, sep='[SEP]', output_dir='data', eval_file='train_eval.json')

    
