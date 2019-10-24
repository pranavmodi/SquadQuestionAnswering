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


# def build_features(args, examples, data_type, out_file, word2idx_dict, char2idx_dict, is_test=False):
#     para_limit = args.test_para_limit if is_test else args.para_limit
#     ques_limit = args.test_ques_limit if is_test else args.ques_limit
#     ans_limit = args.ans_limit
#     char_limit = args.char_limit

#     def drop_example(ex, is_test_=False):
#         if is_test_:
#             drop = False
#         else:
#             drop = len(ex["context_tokens"]) > para_limit or \
#                    len(ex["ques_tokens"]) > ques_limit or \
#                    (is_answerable(ex) and
#                     ex["y2s"][0] - ex["y1s"][0] > ans_limit)

#         return drop

#     print(f"Converting {data_type} examples to indices...")
#     total = 0
#     total_ = 0
#     meta = {}
#     context_idxs = []
#     context_char_idxs = []
#     ques_idxs = []
#     ques_char_idxs = []
#     y1s = []
#     y2s = []
#     ids = []
#     for n, example in tqdm(enumerate(examples)):
#         total_ += 1

#         if drop_example(example, is_test):
#             continue

#         total += 1

#         def _get_word(word):
#             for each in (word, word.lower(), word.capitalize(), word.upper()):
#                 if each in word2idx_dict:
#                     return word2idx_dict[each]
#             return 1

#         def _get_char(char):
#             if char in char2idx_dict:
#                 return char2idx_dict[char]
#             return 1

#         context_idx = np.zeros([para_limit], dtype=np.int32)
#         context_char_idx = np.zeros([para_limit, char_limit], dtype=np.int32)
#         ques_idx = np.zeros([ques_limit], dtype=np.int32)
#         ques_char_idx = np.zeros([ques_limit, char_limit], dtype=np.int32)

#         for i, token in enumerate(example["context_tokens"]):
#             context_idx[i] = _get_word(token)
#         context_idxs.append(context_idx)

#         for i, token in enumerate(example["ques_tokens"]):
#             ques_idx[i] = _get_word(token)
#         ques_idxs.append(ques_idx)

#         for i, token in enumerate(example["context_chars"]):
#             for j, char in enumerate(token):
#                 if j == char_limit:
#                     break
#                 context_char_idx[i, j] = _get_char(char)
#         context_char_idxs.append(context_char_idx)

#         for i, token in enumerate(example["ques_chars"]):
#             for j, char in enumerate(token):
#                 if j == char_limit:
#                     break
#                 ques_char_idx[i, j] = _get_char(char)
#         ques_char_idxs.append(ques_char_idx)

#         if is_answerable(example):
#             start, end = example["y1s"][-1], example["y2s"][-1]
#         else:
#             start, end = -1, -1

#         y1s.append(start)
#         y2s.append(end)
#         ids.append(example["id"])

#     np.savez(out_file,
#              context_idxs=np.array(context_idxs),
#              ques_idxs=np.array(ques_idxs),
#              y1s=np.array(y1s),
#              y2s=np.array(y2s),
#              ids=np.array(ids))
#     print(f"Built {total} / {total_} instances of features in total")
#     meta["total"] = total
#     return meta


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



def process_file(filename, tokenizer, output_file, batch_size=64, cls='[CLS]', para_limit=384,
                 pad_token=0, text_limit=512, sep='[SEP]', output_dir='outputdir'):

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
                    if qas_tokens[i + j] != at:
                        found = False
                        break
                if found:
                    return (i, i+ans_len)

        # print('returning None, span not found')
        # print(qas_tokens, ans_tokens)
        return None

    with open(filename, "r") as fh:
        source = json.load(fh)
        print('the num of articles', len(source["data"]))

        qas_embeddings = []
        start_positions = []
        end_positions = []
        file_num = 0
        indices = []
        count = 0
        num_clipped = 0
        for article in tqdm(source["data"]):
            #print('number of paragraphs in article', len(article["paragraphs"]))
            for para in article["paragraphs"]:
                context = para['context']
                context_tokens = tokenizer.tokenize(context)

                for qas in para['qas']:
                    question = qas['question']
                    question = qas['question']

                    question_tokens = tokenizer.tokenize(question)
                    max_context_len = text_limit - len(question_tokens) - 2

                    if len(context_tokens) > max_context_len:
                        context_tokens = context_tokens[:max_context_len]
                        num_clipped += 1

                    ## For every question in the para, tokenize and get bert indices
                    # qas_text = ' '.join([cls, context, sep, question])
                    # qas_tokens = tokenizer.tokenize(qas_text)

                    qas_tokens = [cls] + context_tokens + [sep] + question_tokens
                    qas_indices = tokenizer.encode(qas_tokens)

                    if len(qas_indices) < text_limit:
                        pad_len = text_limit - len(qas_indices)
                        qas_indices += [pad_token] * pad_len

                    indices.append(qas_indices)

                    ## For each answer for the question, get the span and insert into list
                    if qas['is_impossible']:
                        #print('for file num ', file_num, ' the answer is impossible')
                        span = (-1, -1)

                    else:
                        ans = qas['answers'][0]
                        ans_text = ans['text']
                        ans_tokens = tokenizer.tokenize(ans_text)
                        start = ans['answer_start']
                        span = find_ans_span(qas_tokens, ans_tokens)
                        if span is None:
                            span = (-1, -1)
                        break
                        
                        # if len(answers) > 1:
                        #     print('more than one ansers')
                        #     print(answers)
                        #     import sys
                        #     sys.exit()
                        # for ans in answers:
                        #     ans_text = ans['text']
                        #     ans_tokens = tokenizer.tokenize(ans_text)
                        #     start = ans['answer_start']
                        #     span = find_ans_span(qas_tokens, ans_tokens)
                        #     if span is None:
                        #         span = (-1, -1)
                        #     break

                    start_positions.append(span[0])
                    end_positions.append(span[1])
                    count += 1

                    # if count % batch_size == 0:
                    #     count = 0
                    #     batch, att_mask = batchify_indices(indices)
                    #     #bert_embed = model(batch, attention_mask=att_mask)[0]
                    #     assert len(start_positions) == bert_embed.size()[0]
                    #     output_file = 'embedding_' + str(file_num) + '.npz'
                    #     writeto = os.path.join(output_dir, output_file)
                    #     with torch.no_grad():
                    #         np.savez(writeto,
                    #                  embeddings=bert_embed.detach(),
                    #                  start_positions=np.array(start_positions),
                    #                  end_positions=np.array(end_positions))
                    #     file_num += 1
                    #     start_positions = []
                    #     end_positions = []
                    #     indices = []
                    #     gc.collect()

        out_file = os.path.join(output_dir, output_file)

        print('number of examples', len(indices))
        np.savez(out_file,
                 context_idxs=np.array(indices),
                 y1s=np.array(start_positions),
                 y2s=np.array(end_positions))
        #print(f"Built {total} / {total_} instances of features in total")

        print(f"Clipped {num_clipped} instances of features in total")




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

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_file = 'data/train-v2.0.json'
    dev_file = 'data/dev-v2.0.json'

    process_file(train_file, tokenizer, output_file='train.npz', batch_size=64,
                 cls='[CLS]', para_limit=384, sep='[SEP]', output_dir='data')

    process_file(dev_file, tokenizer, output_file='dev.npz', batch_size=64, cls='[CLS]',
                 para_limit=384, sep='[SEP]', output_dir='data')
    
