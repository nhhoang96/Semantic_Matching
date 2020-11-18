import numpy as np
import copy

def load_new_intents(label_dict, new_intents_fname):
    new_intents = []
    for line in open(new_intents_fname):
        new_intents.append(label_dict[line.strip()])
    return new_intents

def load_test(data,eval): 
    print ("------ Testing ------")
    x_te = data['x_te']
    y_te = data['y_te']
    te_len = data['len_te']
    te_mask = data['mask_te']
    text_te = data['text_te']

    return x_te, y_te, te_len, te_mask, text_te

def load_support(data, train):
    if (train == True):
        x_support = data['x_support_tr']
        y_support = data['y_support_tr']
        support_m = data['mask_support_tr']
        x_len_support = data['len_support_tr']
        support_text = data['text_support_tr']

    else:
        x_support = data['x_support']
        y_support = data['y_support']
        support_m = data['mask_support']
        x_len_support = data['len_support']
        support_text = data['text_support']

    return x_support,y_support,x_len_support, support_m, support_text

#------- Helper function for episodic training/ testing -------#
def init_support_query(size, feature_size, num_class):
    feature_arr = np.zeros(shape=(size*num_class, feature_size), dtype=np.float64)
    class_arr = np.zeros(shape = (size*num_class,), dtype = np.int64)
    len_arr = np.zeros(shape=(size*num_class,), dtype=np.int64)
    ind_arr = np.zeros(shape=(size*num_class, num_class), dtype=np.int64)

    mask_arr = np.zeros(shape=(size*num_class,feature_size), dtype=np.float64)
    return feature_arr, class_arr, len_arr, ind_arr, mask_arr

# Create one-hot encoded label (0.0 0.0 1.0) from label (2) (example)
def create_index(y):
    sample_num = y.shape[0]
    labels = np.unique(y)
    class_num = labels.shape[0]
    ind = np.zeros((sample_num, class_num),dtype=np.float32)
    labels = range(class_num)
    for i in range(class_num):
        ind[y == labels[i],i] = 1
    return ind


#----Data Loading Helper -----#
def generate_tok_idx(tokens, word2idx):
    tok_idx = []
    for t in tokens:
        if not (t in word2idx):
            tok_idx.append(word2idx['null'])
        else:
            tok_idx.append(word2idx[t])

    tok_idx = np.array(tok_idx)
    return tok_idx


def create_mask(tokens, max_len, tok_idx):
    if (len(tokens) < max_len):
        tmp = np.append(tok_idx, np.zeros((max_len - len(tokens),), dtype= np.int64))
    else:
        tmp = tok_idx[0:max_len]

    current_mask = np.ones(shape=tmp.shape)
    for j in range (tmp.shape[0]):
        if (j >= len(tokens)):
            current_mask[j]  = 0.0
    return tmp, current_mask

def shuffle_data(x,y,len,mask,text):
    index = np.arange(y.shape[0])
    np.random.shuffle(index)
    new_y = y[index]
    new_x = x[index]
    new_len = len[index]
    new_mask = mask[index]
    new_text = text[index]
    return new_x,new_y,new_len,new_mask, new_text

#Produce n classes from the total number of classes
def produce_chosen_class(available_classes, y_tr_copy, num_class):
    index_list = []
    chosen_classes = copy.deepcopy(available_classes)
    # Randomly choose n classes from data
    np.random.shuffle(chosen_classes)
    cs = chosen_classes[:num_class]
    #------ Ensure the chosen classes meet the requirements (have at least num_shot + num_query) ---------#
    # Create the updated indexes of the chosen class (this is dynamic from episodes to episodes)
    for val in available_classes:
        index = np.where(y_tr_copy == val)[0]
        index_list.append(index)
    
    return cs, index_list

def create_samples(feature,label,len,ind,mask,text, num_sample, chosen_classes):
    cur_loc = 0
    old_loc = 0
    sub_feature, sub_class, sub_len, sub_ind, sub_mask = init_support_query(num_sample, feature.shape[1], chosen_classes.shape[0])
    sub_text = np.empty(shape=(num_sample * chosen_classes.shape[0],),dtype=text.dtype)
    for cur_class in chosen_classes:

        class_index = np.where(label == cur_class)[0]
        while (int(class_index.shape[0]) < num_sample):
            shuffle_index = copy.deepcopy(class_index)
            np.random.shuffle(shuffle_index)
            class_index= np.concatenate((class_index, shuffle_index[: int(num_sample - class_index.shape[0])]))
        np.random.shuffle(class_index)
        support_index = class_index[:num_sample]
        old_loc = cur_loc
        cur_loc = cur_loc + num_sample
        sub_feature[old_loc: cur_loc,:] = feature[support_index]
        sub_class[old_loc: cur_loc,] = label[support_index]
        sub_len[old_loc:cur_loc,] = len[support_index] 
        sub_mask[old_loc:cur_loc,] = mask[support_index] 
        sub_text[old_loc:cur_loc,] = text[support_index] 
    index_key = {}
    
    for i in range (chosen_classes.shape[0]):
        cur_index = np.where(sub_class == chosen_classes[i])[0]
        index_key[chosen_classes[i]] = cur_index
    # Reset class name (0-2-4) => (0-1-2)
    for i in range (chosen_classes.shape[0]):
        sub_class[index_key[chosen_classes[i]]] = i
    sub_ind = create_index(sub_class)
    return sub_feature, sub_class, sub_len, sub_ind, sub_mask, sub_text

def create_query_support(x_support, y_support, len_support, ind_support, mask_support, text_support, x,y,x_len,y_ind, mask, text, config,num_class): 
	all_classes = np.unique(y)
	all_classes = np.sort(all_classes)
	# Sample sub-classes
	cs, index_list = produce_chosen_class(all_classes, y, num_class)
	train_classes = cs
	if (config['tgt'] == 'joint'):
		copy_classes = copy.deepcopy(all_classes)
		ix = np.array([], dtype=np.int32)
		for c in cs:
			ix = np.append(ix, np.where(all_classes==c)[0])

		seen_classes = np.delete(copy_classes, ix, axis=0)
		new_class,_ = produce_chosen_class(seen_classes, y, num_class)
		train_classes = np.append(train_classes, new_class)
	support_feature, support_class, support_len, support_ind,support_mask, support_text = create_samples(x_support, y_support, len_support, ind_support, mask_support, text_support, config['num_samples_per_class'], train_classes)

	query_feature, query_class, query_len, query_ind,query_mask, query_text = create_samples(x, y, x_len, y_ind, mask, text, config['num_query_per_class'], train_classes)

	return support_feature, support_class, support_len, support_ind, support_mask, support_text, query_feature, query_class, query_len, query_ind, query_mask, query_text
