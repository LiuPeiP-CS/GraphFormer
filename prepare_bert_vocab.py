import argparse
import json
import pickle
import numpy as np
import torch
from collections import Counter
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from transformers import logging
logging.set_verbosity_error()

from utils import vocab, constant, helper
from data.Finetuning_BertCRF.Bert_Feature import GetBertFeature

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare vocab for relation extraction.')
    parser.add_argument('--data_dir', default='dataset/pubmed', help='TACRED directory.')
    parser.add_argument('--glove_dir', default='dataset/glove', help='GloVe directory.')
    parser.add_argument('--wv_file', default='glove.840B.300d.txt', help='GloVe vector file.')
    parser.add_argument('--wv_dim', type=int, default=300, help='GloVe vector dimension.')
    parser.add_argument('--vocab_dir', default='dataset/bert_vocab_pubmed', help='Output vocab directory.')
    parser.add_argument('--bert_model', default='dataset/bert/bert-base-uncased', help='BERT model name.')
    parser.add_argument('--lower', action='store_true', help='If specified, lowercase all words.')
    parser.add_argument('--min_freq', type=int, default=0, help='If > 0, use min_freq as the cutoff.')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Input files
    train_file = args.data_dir + '/train.json'
    test_file = args.data_dir + '/test.json'
    wv_file = args.glove_dir + '/' + args.wv_file
    wv_dim = args.wv_dim

    # Output files
    helper.ensure_dir(args.vocab_dir)
    vocab_file = args.vocab_dir + '/vocab.pkl'
    emb_file = args.vocab_dir + '/embedding.npy'

    # Load files
    print("Loading files...")
    train_tokens, train_tokens_list = load_tokens(train_file, args.data_dir)
    test_tokens, test_tokens_list = load_tokens(test_file, args.data_dir)
    if args.lower:
        train_tokens = [t.lower() for t in train_tokens]
        test_tokens = [t.lower() for t in test_tokens]



    print("--------------------------构建vocab.pkl-------------------------------")
    # if args.lower:
    #     train_tokens, test_tokens = [[t.lower() for t in tokens] for tokens in \
    #                                  (train_tokens, test_tokens)]
    #
    # # load glove
    # print("loading glove...")
    # glove_vocab = vocab.load_glove_vocab(wv_file, wv_dim)
    # print("{} words loaded from glove.".format(len(glove_vocab)))
    #
    # print("building vocab...")
    # v = build_vocab(train_tokens, glove_vocab, args.min_freq, args.data_dir)
    #
    # # Save vocabulary
    # print("Saving vocab...")
    # with open(vocab_file, 'wb') as outfile:
    #     pickle.dump(v, outfile)

    # ----------------------------------------------------------构建bert词汇表new
    # Load BERT tokenizer
    print("Loading BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, model_max_length=128)
    print("构建之前：")
    print(tokenizer.tokenize('StoneDrill'))
    print(tokenizer.tokenize('adenocarcinoma'))
    print(tokenizer.tokenize('COVID'))
    print(tokenizer.tokenize('hospitalization'))
    all_tokens = train_tokens + test_tokens
    # print(all_tokens)
    # new_tokens = ['COVID', 'hospitalization']
    # num_added_toks = tokenizer.add_tokens(new_tokens)
    num_added_toks = tokenizer.add_tokens(all_tokens)

    print(f"添加新词的数量为：{num_added_toks}")
    model = BertModel.from_pretrained(args.bert_model)
    model.resize_token_embeddings(len(tokenizer))
    tokenizer.save_pretrained(args.bert_model)
    #重新加载词汇表
    # tokenizer = BertTokenizer.from_pretrained(args.bert_model, use_fast=True)
    # model = BertModel.from_pretrained(args.bert_model)

    # 获取BERT生成的词汇表
    bert_vocab = tokenizer.get_vocab()
    print("构建之后：")
    print(tokenizer.tokenize('StoneDrill'))
    print(tokenizer.tokenize('adenocarcinoma'))
    print(tokenizer.tokenize('COVID'))
    print(tokenizer.tokenize('hospitalization'))

    # 将映射字典保存到文件
    with open(vocab_file, 'wb') as f:
        pickle.dump((bert_vocab), f)
    print("--------------------------结束构建vocab.pkl-------------------------------")


    # Extract BERT embeddings
    print("--------------------------------------开始构建bert——embeddings-------------------------")
    lables = []
    for sent in train_tokens_list:
        lable = []
        for i in range(len(sent)):
            lable.append("O")
        lables.append(lable)
    for sent in test_tokens_list:
        lable = []
        for i in range(len(sent)):
            lable.append("O")
        lables.append(lable)

    data = train_tokens_list + test_tokens_list
    token_len = max(len(x) for x in data)
    print(f"句子最大长度：{token_len}")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if token_len > 128:
        bert = GetBertFeature(device, token_len, 100)
        bert_embeddings = bert.get_bert_feature(train_tokens_list, lables, device, token_len, 100)
    else:
        bert = GetBertFeature(device, token_len, 100)
        bert_embeddings = bert.get_bert_feature(train_tokens_list, lables, device, token_len, 100)
    print()

    print("--------------------------------------构建完毕bert——embeddings-------------------------")
    # Save embeddings
    print(f"bert_embeddings类型为：{type(bert_embeddings)},形状为：{bert_embeddings.shape}")
    print("Saving embeddings...")
    bert_embeddings = bert_embeddings.reshape(bert_embeddings.shape[0] * bert_embeddings.shape[1], 768).cpu()


    # 调整形状
    # embedding = np.concatenate((train_embeddings, test_embeddings), axis=0)
    desired_size = len(bert_vocab)  # 期望的行数

    assert bert_embeddings.shape[1] == 768  # 检查嵌入维度是否正确

    if bert_embeddings.shape[0] < desired_size:
        print("Warning: The current embedding matrix has fewer rows than the desired size. Padding with zeros.")
        embedding_resized = np.pad(bert_embeddings, ((0, desired_size - bert_embeddings.shape[0]), (0, 0)),
                                   mode='constant')
    else:
        embedding_resized = bert_embeddings[:desired_size, :]

    print(embedding_resized.shape)
    np.save(emb_file, embedding_resized)  # 将调整后的嵌入矩阵保存为 embedding.npy 文件
    print("BERT preprocessing complete.")

def load_tokens(filename, data_dir):
    with open(filename, encoding='utf-8') as infile:
        data = json.load(infile)
        tokens = []
        tokens_list = []
        for d in data:
            ts = d['token']
            tokens_list.append(ts)
            tokens += list(filter(lambda t: t != '<PAD>', ts))
    print("{} tokens from {} examples loaded from {}.".format(len(tokens), len(data), filename))
    return tokens,tokens_list


# def build_vocab(tokens, glove_vocab, min_freq, data_dir):
#     """ build vocab from tokens and glove words. """
#     counter = Counter(t for t in tokens)
#     # if min_freq > 0, use min_freq, otherwise keep all glove words
#     if min_freq > 0:
#         v = sorted([t for t in counter if counter.get(t) >= min_freq], key=counter.get, reverse=True)
#     else:
#         v = sorted([t for t in counter if t in glove_vocab], key=counter.get, reverse=True)
#     # add special tokens and entity mask tokens
#     v = constant.VOCAB_PREFIX  + v
#     print("vocab built with {}/{} words.".format(len(v), len(counter)))
#     return v

# def build_vocab(tokens, min_freq=0):
#     counter = Counter(t for t in tokens)
#     v = sorted([t for t in counter if counter[t] >= min_freq], key=counter.get, reverse=True)
#     v = constant.VOCAB_PREFIX + v  # Add special tokens
#     return v


def tokenize_and_convert_to_ids(tokens, tokenizer):
    tokenized = tokenizer(tokens, truncation=True, padding='longest', return_tensors='pt')
    return tokenized['input_ids'][0].tolist()


if __name__ == '__main__':
    main()
