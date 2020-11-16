""" input data preprocess.
"""
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import pandas as pd

import io
import utils

data_prefix = '../dataset/'

def load_vec(emb_path, nmax= 50000):
    vectors = []
    word2id = {}
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            assert word not in word2id, 'word found twice'
            vectors.append(vect)
            word2id[word] = len(word2id)
            if len(word2id) == nmax:
                break
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.vstack(vectors)
    return embeddings, id2word, word2id

def find_max_fixed_length(all_paths):
    max_len=0
    for p in all_paths:    
        for line in open(p):
            arr = line.strip().split('\t')
            utt = [w for w in arr[1].split(' ')]
            
            if (len(utt) >= max_len):
                max_len = len(utt)

    return max_len

def preprocess_text(train_path,word2idx, embedding, id2word, sc_dict, max_len):
    sent = []
    mask = []
    x_train = []
    x_len = []
    y_train = []
    for line in open(train_path):
        arr = line.strip().split('\t') 
        tokens = [w for w in arr[1].split(' ')]
        label = [w for w in arr[0].split(' ')]
        cname = ' '.join(label)
        if not (cname in sc_dict):
            continue    
        tok_idx = utils.generate_tok_idx(tokens, word2idx)
        x_len.append(len(tokens))
        y_train.append(sc_dict[cname])
        sent.append (arr[1])
        tmp, current_mask = utils.create_mask(tokens, max_len, tok_idx)
        x_train.append(tmp) 
        mask.append(current_mask)
    
    return np.asarray(x_train), np.array(y_train), np.array(x_len),np.array(sent), np.array(mask)


def load_label_dict(label_fname):
    label_dict = {}
    idx = 0
    for line in open(label_fname):
        label_dict[line.strip()] = idx
        idx += 1
    return label_dict

def load_new_intents(label_dict, new_intents_fname):
    new_intents = []
    for line in open(new_intents_fname):
        new_intents.append(label_dict[line.strip()])
    return new_intents

def train_intents(support_x,support_y, support_len, max_support_len, mask_support, new_intents):

    unseen_idx  = []
    for i in new_intents:
        unseen = np.where(support_y == i)[0]
        unseen_idx.append(unseen)

    support_x_tr = np.delete(support_x, unseen_idx,axis=0)
    support_y_tr = np.delete(support_y, unseen_idx, axis=0)
    support_len_tr = np.delete(support_len, unseen_idx, axis=0)
    max_support_len_tr = np.delete(max_support_len, unseen_idx, axis=0)
    
    mask_support_tr = np.delete(mask_support, unseen_idx, axis=0)
    return support_x_tr, support_y_tr, support_len_tr, max_support_len_tr, mask_support_tr

def save_data(data_type, x,y,len,text, mask, data):
    data['x_' + data_type] = x
    data['y_' + data_type] = y
    data['len_' + data_type] = len
    data['text_' + data_type] = text
    data['mask_' + data_type] = mask
    return data

def load_fasttext(fasttext_path, support_path, support_path_tr, training_data_path, test_data_path, data,label_dict):
    print ("------------------load FASTTEXT begin-------------------")
    emb_file = fasttext_path
    embedding, id2word, word2idx = load_vec(emb_file)
    print ("------------------load FASTTEXT end-------------------")
    max_len = 0
    all_paths = [support_path, support_path_tr, training_data_path, test_data_path]
    max_len = find_max_fixed_length(all_paths)
    print ("----- Load support --------")
    support_x, support_y, support_len, support_text, support_mask = preprocess_text(support_path, word2idx, embedding, id2word, label_dict, max_len) 
    support_x_tr, support_y_tr, support_len_tr, support_text_tr, support_mask_tr = preprocess_text(support_path_tr, word2idx, embedding, id2word, label_dict, max_len) 
         
    print ("----- Load training data ------")
    x_tr, y_tr, x_len_tr, tr_sent, tr_mask = preprocess_text(training_data_path, word2idx, embedding, id2word, label_dict, max_len)
    print ("------ Load test data  --------")
    x_te, y_te, x_len_te,te_sent,te_mask = preprocess_text(test_data_path, word2idx, embedding, id2word,label_dict, max_len)

    data = save_data('tr', x_tr,y_tr,x_len_tr, tr_sent, tr_mask, data)        
    data = save_data('te', x_te,y_te,x_len_te, te_sent, te_mask, data)
    data = save_data('support_tr', support_x_tr,support_y_tr,support_len_tr, support_text_tr, support_mask_tr, data)
    data = save_data('support', support_x, support_y, support_len, support_text, support_mask, data)

    data['vocab_size'] = embedding.shape[0]
    data['word_emb_size'] = embedding.shape[1]
    data['embedding'] = embedding
    data['max_len'] = max_len
    return data

def read_datasets(num_shots, dataset, num_fold,src, tgt, fasttext_path):
    print ("------------------read datasets begin-------------------")
    data = {}
    if (dataset == 'SNIPS'):    
        data_path = data_prefix + dataset + '/'

        label_dict = {"book restaurant": 0,
            "get weather": 1,
            "play music": 2,
            "search creative work": 3,
            "search screening event": 4,
            "add to playlist": 5,
            "rate book": 6}
        new_intents = [5, 6]
        old_intents = [0,1,2,3,4]
        reverse_label_dict = { 0:  "book restaurant",
                1: "get weather",
                2: "play music",
                3: "search creative work",
                4: "search screening event",
                5: "add to playlist",
                6: "rate book"}

    else:
        data_path = data_prefix + dataset + '/' + 'KFold_' + str(num_fold) + '/'
        label_file = data_prefix + 'NLUE' + '/label.tsv'
        new_intents_file = data_prefix + 'NLUE' + '/new_intents'
        label_dict = load_label_dict(label_file)
        new_intents = load_new_intents(label_dict, new_intents_file)
        #print ("Label dict", label_dict)
        #print ("New intents", new_intents)
        reverse_label_dict={}
        for k,v in label_dict.items():
            reverse_label_dict[v] = k

    training_data_path = data_path + 'train_' + src + '.tsv'
   # Load support, train, test
    if (num_shots == 1):
        support_path =  data_path + 'support_1_shots_'+ tgt
        support_path_tr =  data_path + 'support_1_shots_'+ src

        print ("Loading support 1")
    elif (num_shots == 5):
        support_path = data_path + 'support_5_shots_' + tgt
        support_path_tr =  data_path + 'support_5_shots_'+ src
        print ("Loading support 5")

# Read data
    test_data_path = data_path + 'test_' + tgt + '.tsv'
    df_train = pd.read_csv(training_data_path, sep='\t')
    df_test = pd.read_csv(test_data_path, sep='\t')
    
    data = load_fasttext(fasttext_path, support_path, support_path_tr, training_data_path, test_data_path, data,label_dict)
    data['intent_dict'] = label_dict
    data['label_dict'] = label_dict
    data['new_intents'] = new_intents
    data['reverse_dict'] = reverse_label_dict
    print ("------------------read datasets end---------------------")
    return data
