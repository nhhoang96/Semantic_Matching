# Import libraries
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import math

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import copy
import argparse
import csv
import os
import random

#Import supporting file
from read_input_data import read_datasets
import sman
import utils
import ast

def mask_logsoftmax(vec, device):
	mask = torch.zeros(vec.shape).to(device)
	idx = torch.where(vec != 0.0)
	mask[idx] = 1.0
	mask_vec = vec + (mask + 1e-45).log()
	mask_vec = F.log_softmax(mask_vec, dim=1)
	
	return mask_vec

def set_seed(seed_val):
	random.seed(seed_val)
	torch.cuda.cudnn_enabled = False
	np.random.seed(seed_val)
	torch.manual_seed(seed_val)
	torch.cuda.manual_seed(seed_val)
	torch.cuda.manual_seed_all(seed_val)
	torch.backends.cudnn.deterministic=True
	torch.backends.cudnn.benchmark=False

def mask_softmax(vec, device):
	mask = torch.zeros(vec.shape).to(device)
	idx = torch.where(vec != 0.0)
	mask[idx] = 1.0
	mask_vec = vec + (mask + 1e-45)
	mask_vec = F.softmax(mask_vec, dim=1)
	return mask_vec

def attn_loss(support_attn,support_labels, query_attn, query_labels, num_class, num_support, num_head, num_query, device, kl_loss):
	num_query = query_labels.shape[0]
	batch=1
	
	support_attn	= support_attn.unsqueeze(1).view(num_class,num_support, num_head, -1).contiguous() # NQ, 1,len, D
	support_attn = support_attn.unsqueeze(0).view(1, num_class,num_support,num_head,-1).contiguous()
	support_attn = support_attn.unsqueeze(0).view(batch,1,num_class,num_support,num_head,-1).contiguous()
	support_attn = support_attn.expand(batch, num_query,num_class,num_support,num_head, -1).contiguous()
	
	support_attn = support_attn.view(batch*num_query*num_class*num_support,num_head, -1) #NQN, max_len, D)
	
	if (num_support == 1):
		support_attn = support_attn.view(batch*num_query*num_class*num_support,num_head, -1) #NQN, max_len, D)

		query_attn = query_attn.unsqueeze(1).view(num_query, 1, num_head, -1).contiguous() # NQ, 1,len, D
		query_attn = query_attn.unsqueeze(0).view(batch, num_query,1,num_head,-1).contiguous()
		query_attn = query_attn.expand(batch, num_query,num_class,num_head, -1).contiguous()
		query_attn = query_attn.view(batch*num_query*num_class,num_head, -1) #NQN, max_len, D)
		
		support_labels = support_labels.view(-1,).contiguous()
		support_labels = support_labels.repeat(num_query,).to(device)
		
		query_labels = query_labels.view(-1,).contiguous()
		query_labels = query_labels.repeat(num_class,).to(device)

		similar = (support_labels.float() == query_labels.float()).double()
		dissimlar = (support_labels.float() != query_labels.float()).double()
		
		sim_indices= torch.nonzero(similar, as_tuple=True)[0] # row indexes
		dis_indices= torch.nonzero(dissimlar, as_tuple=True)[0]
		
		similar_sup = support_attn[sim_indices,:,:]
		similar_query = query_attn[sim_indices,:,:]
		similar_sup =mask_softmax(similar_sup.sum(2), device)
		similar_query =mask_logsoftmax(similar_query.sum(2), device)

		dis_sup = support_attn[dis_indices,:,:]
		dis_query = query_attn[dis_indices,:,:]
		dis_sup = mask_softmax(dis_sup.sum(2), device)
		dis_query = mask_logsoftmax(dis_query.sum(2), device)
		
		sim_loss = kl_loss(similar_query, similar_sup)
		dis_loss = kl_loss(dis_query, dis_sup)
		new_loss = sim_loss - dis_loss
	else:
		support_attn = support_attn.view(batch*num_query*num_class * num_support,num_head, -1) #NQN, max_len, D)

		query_attn = query_attn.unsqueeze(1).view(num_query, 1, num_head, -1).contiguous() # NQ, 1,len, D
		query_attn = query_attn.unsqueeze(0).view(batch, num_query,1,num_head,-1).contiguous()
		query_attn = query_attn.expand(batch, num_query,num_class,num_head, -1).contiguous()
		query_attn = query_attn.view(batch*num_query*num_class,num_head, -1) #NQN, max_len, D)
		query_attn =	query_attn.unsqueeze(1).repeat(1,num_support,1,1)
		query_attn = query_attn.view(batch*num_query*num_class*num_support,num_head,-1).contiguous()

		support_labels = support_labels.view(-1,).contiguous()
		support_labels = support_labels.repeat(num_query,).to(device)
		
		query_labels = query_labels.view(-1,).contiguous()
		query_labels = query_labels.repeat(num_class*num_support,).to(device)

		similar = (support_labels.float() == query_labels.float()).double()
		dissimlar = (support_labels.float() != query_labels.float()).double()
		
		sim_indices= torch.nonzero(similar, as_tuple=True)[0] # row indexes
		dis_indices= torch.nonzero(dissimlar, as_tuple=True)[0]
		similar_sup = support_attn[sim_indices,:,:]
		similar_query = query_attn[sim_indices,:,:]
		similar_sup =mask_softmax(similar_sup.sum(2), device)	
		similar_query =mask_logsoftmax(similar_query.sum(2), device)
		
		dis_sup = support_attn[dis_indices,:,:]
		dis_query = query_attn[dis_indices,:,:]
		dis_sup = mask_softmax(dis_sup.sum(2), device)
		dis_query = mask_logsoftmax(dis_query.sum(2), device)
		
		sim_loss = kl_loss(similar_query, similar_sup)
		dis_loss = kl_loss(dis_query, dis_sup)
		new_loss = sim_loss - dis_loss

	return new_loss

# Add self-attention loss
def selfattn_loss(logits, label, attention, device):
	self_atten_mul = torch.matmul(attention, attention.permute([0, 2, 1])).float()
	sample_num, att_matrix_size, _ = self_atten_mul.shape
	self_atten_loss = (torch.norm(self_atten_mul - torch.from_numpy(np.identity(att_matrix_size)).to(device).float()).float()) ** 2
	return torch.mean(self_atten_loss)

def word_distr_loss(support_attn, support_len, query_attn,query_len, kl_loss, device): # batch, head, D
	word_support_attn = support_attn.sum(1)
	
	mask = torch.zeros(word_support_attn.shape).to(device)
	idx = torch.where(word_support_attn != 0.0)
	
	mask_idx = torch.where(word_support_attn == 0.0)
	mask[idx] = 1.0
	
	word_support_attn = mask_logsoftmax(word_support_attn, device)	
	kl_sup_word = 0.0

	for j in range (word_support_attn.shape[0]):
		sen = word_support_attn[j, :support_len[j]]
		uniform_dist = torch.zeros(sen.shape).uniform_(0,1).to(device)
		temp = kl_loss(sen, uniform_dist).double()
		kl_sup_word += (temp)
	
	word_query_attn = query_attn.sum(1)
	query_mask = torch.zeros(word_query_attn.shape).to(device)

	mask_idx = torch.where(word_query_attn == 0.0)
	idx = torch.where(word_query_attn != 0.0)
	query_mask[idx] = 1.0
	word_query_attn = mask_logsoftmax(word_query_attn, device)
	kl_query_word = 0.0
	
	for j in range (word_query_attn.shape[0]):
		sen = word_query_attn[j, :query_len[j]] 
		uniform_dist = torch.zeros(sen.shape).uniform_(0,1).to(device)
		temp = kl_loss(sen, uniform_dist).double()
		kl_query_word += (temp)
	
	return kl_sup_word, kl_query_word

def compute_loss(prediction, support_attn, support_ind, support_len, labels, query_attn,	query_class, query_ind, query_len,	config, loss_fn,kl_loss):
	loss_val = loss_fn(prediction.double(), torch.Tensor(query_class).to(config['device']).long())
	tr_pred = np.argmax(prediction.cpu().detach().clone(), 1)
	#bsz, r, len
	if (config['tgt'] == 'joint'):
		num_class = config['num_class'] * 2
	else:
		num_class = config['num_class']

	# Self-attention regularization
	add_loss = config['self_attn_loss'] * selfattn_loss(prediction, query_ind.reshape(-1, query_ind.shape[1]), query_attn, config['device']).double()
	add_loss += config['self_attn_loss'] * selfattn_loss(prediction, support_ind.reshape(-1, support_ind.shape[1]), support_attn, config['device']).double()
	loss_val += add_loss

	# Head distribution regularization
	new_attn_loss = attn_loss(support_attn,labels.float(), query_attn, tr_pred.float(), num_class, config['num_samples_per_class'], config['r'], config['num_query_per_class'], config['device'],kl_loss)
	loss_val += config['same_intent_loss'] * new_attn_loss.double()

	# Head uniform regularization
	kl_sup_word, kl_query_word = word_distr_loss(support_attn, support_len, query_attn, query_len, kl_loss, config['device'])
	loss_val += config['uniform_loss'] * kl_sup_word
	loss_val += config['uniform_loss'] * kl_query_word

	return loss_val

def convert_to_tensor(x,y,y_ind,x_len, mask, device):
	x_tensor = torch.LongTensor(x).to(device)
	y_id_tensor = torch.LongTensor(y).to(device)
	y_ind_tensor = torch.FloatTensor(y_ind).to(device)
	x_len_tensor = torch.LongTensor(x_len).to(device)
	mask_tensor = torch.LongTensor(mask).to(device)
	return x_tensor, y_id_tensor, y_ind_tensor, x_len_tensor, mask_tensor

def parse_argument():
	parser = argparse.ArgumentParser()

	parser.add_argument('--ckpt_dir', type=str, default = './ckpt/')
	parser.add_argument('--device', type=str, default='cpu')
	parser.add_argument('--num_class', type = int, default=2)
	parser.add_argument('--num_test_class', type = int, default=7)
	parser.add_argument('--num_samples_per_class', type=int, default=1)
	parser.add_argument('--num_query_per_class', type=int, default=20)
	parser.add_argument('--num_episodes', type=int, default=1000)
	parser.add_argument('--learning_rate', type=float, default=0.0001)
	parser.add_argument('--num_run', type=int, default=1)
	parser.add_argument('--dataset', type=str, default='SNIPS')
	parser.add_argument('--num_fold', type=int, default=1)
	parser.add_argument('--r', type=int, default=4)
	parser.add_argument('--l', type=int, default=5)
	parser.add_argument('--hidden_size',type=int,default=64)
	parser.add_argument('--same_intent_loss', type=float, default=0.01)
	parser.add_argument('--uniform_loss', type=float, default=1e-5)
	parser.add_argument('--self_attn_loss', type=float, default=0.0001)
	parser.add_argument('--eps', type=str, default='noneps')
	parser.add_argument('--src', type=str, default='seen')
	parser.add_argument('--tgt',type=str, default='joint')
	parser.add_argument('--d_a', type=int, default='20')
	parser.add_argument('--fasttext_path',type=str, default='../fasttext/vectors-en.txt')
	args = parser.parse_args()
	config = args.__dict__
	return config

#------ Episode testing ----------------------#
def evaluate_episode(data, config, model, loss_fn, eval):

	x_te, y_te, te_len, te_mask, text_te = utils.load_test(data, eval)
	y_te_ind = utils.create_index(y_te)

	reverse_dict = data['reverse_dict']
	
	x_support, y_support, x_len_support, support_m, sup_text = utils.load_support(data, False)
	y_support_ind = utils.create_index(y_support)
	
	total_prediction = np.array([], dtype=np.int64)
	total_y_test = np.array([], dtype=np.int64)
	cum_acc = []
	
	with torch.no_grad():
		for episode in range (config['num_episodes']):
			support_feature, support_class, support_len, support_ind, support_mask, support_text, query_feature, query_class, query_len, query_ind, query_mask,query_text = utils.create_query_support(x_support, y_support, x_len_support, y_support_ind, support_m, sup_text, x_te, y_te, te_len, y_te_ind, te_mask, text_te, config, config['num_test_class'])

			support_feature, support_id, support_ind, support_len, support_mask = convert_to_tensor(support_feature, support_class, support_ind, support_len, support_mask, config['device'])
			query_feature, query_id, query_ind, query_len, query_mask = convert_to_tensor(query_feature, query_class, query_ind, query_len, query_mask, config['device'])
			prediction,incons_loss, support_attn, query_attn = model.forward(support_feature, support_len, support_mask, query_feature, query_len, query_mask)
			 
			pred = np.argmax(prediction.cpu().detach().numpy(), 1)
			cur_acc = accuracy_score(query_class, pred)
			cum_acc.append(cur_acc)
 
	cum_acc = np.array(cum_acc)
	avg_acc, std_acc = np.mean(cum_acc), np.std(cum_acc)
	print ("Average accuracy", avg_acc) 
	print ("STD", std_acc)
	return avg_acc
#------ End episode testing --------#

#---------START GFSL ---------------------------------#
def evaluate_nonepisode(data, config, model, loss_fn, eval):
		
	x_te, y_te, te_len, te_mask, text_te = utils.load_test(data, eval)
	x_te,y_te,te_len, te_mask,text_te = utils.shuffle_data(x_te,y_te,te_len,te_mask,text_te)
	y_te_ind = utils.create_index(y_te)
	
	reverse_dict = data['reverse_dict']
	num_class = np.unique(y_te)
	
	num_test_query = config['num_query_per_class'] * num_class.shape[0]
	x_support, y_support, x_len_support, support_m, support_text = utils.load_support(data, False)
	y_support_ind = utils.create_index(y_support)
	test_batch = int(math.ceil(x_te.shape[0] / float(num_test_query))) 
	
	total_prediction = np.array([], dtype=np.int64)
	total_y_test = np.array([], dtype=np.int64)

	with torch.no_grad():
		for batch in range (test_batch):
			support_feature, support_class, support_len, support_ind, support_mask = utils.init_support_query(config['num_samples_per_class'], x_te.shape[1], num_class.shape[0])
			query_feature, query_class, query_len, query_ind, query_mask = utils.init_support_query(config['num_query_per_class'], x_te.shape[1], num_class.shape[0])
			
			begin_index = batch * (num_test_query)
			end_index = min((batch + 1) * num_test_query, x_te.shape[0])
			query_feature = x_te[begin_index : end_index]
			query_len = te_len[begin_index : end_index]
			query_class = y_te[begin_index: end_index]
			query_mask = te_mask[begin_index: end_index] 
			query_text = text_te[begin_index: end_index]
			
			support_idx = 0
			num_class = np.unique(y_support)
			for counter in range (num_class.shape[0]):
				class_index = np.where(y_support ==num_class[counter])[0]
				old_support_idx = support_idx
				support_idx = support_idx + config['num_samples_per_class']
				support_feature[old_support_idx:support_idx] = x_support[class_index]
				support_class[old_support_idx:support_idx] = y_support[class_index]
				support_len[old_support_idx:support_idx] = x_len_support[class_index]
				support_mask[old_support_idx:support_idx] = support_m[class_index] 
				support_text[old_support_idx:support_idx] = support_text[class_index]
			cs = np.unique(query_class)
			#Obtain indexes
			q_ind_key = {}
			s_ind_key = {}
			for i in range (len(cs)):
				q_index = np.where(query_class == cs[i])[0]
				s_index = np.where(support_class == cs[i])[0]
				q_ind_key[cs[i]] = q_index
				s_ind_key[cs[i]] = s_index
		 # Reset class index
			for i in range (len(cs)):
				query_class[q_ind_key[cs[i]]] = i
				support_class[s_ind_key[cs[i]]] = i
			support_ind = utils.create_index(support_class)
			query_ind = utils.create_index(query_class)	
			support_feature, support_id, support_ind, support_len, support_mask = convert_to_tensor(support_feature, support_class, support_ind, support_len, support_mask, config['device'])
			query_feature, query_id, query_ind, query_len, query_mask = convert_to_tensor(query_feature, query_class, query_ind, query_len, query_mask, config['device'])
			prediction,_,support_attn,query_attn = model.forward(support_feature, support_len, support_mask, query_feature, query_len, query_mask)

			pred = np.argmax(prediction.cpu().detach().numpy(), 1)
			total_prediction = np.concatenate((total_prediction, pred))
			total_y_test = np.concatenate((total_y_test, query_class))
	acc = accuracy_score(total_y_test, total_prediction)

	cnf = confusion_matrix(total_y_test, total_prediction) 
	print ("Confusion matrix:")
	print (cnf)
	return acc

def MakeDirectory(path):
		if (os.path.exists(path)):
			pass
		else:
			os.mkdir(path)

def load_model(config, embedding):
	model = sman.SMAN(config,embedding).to(config['device']) 
	print ("------Start training model	------------------------")
	print ("---Source-target : ", config['src'], config['tgt'])
	print ("------- Num shot is:", config['num_samples_per_class'], "----------") 
	optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

	loss_fn = nn.CrossEntropyLoss().to(config['device'])
	if os.path.exists(config['ckpt_dir'] + 'best_model.pth'):
		print("Restoring weights from previously trained model.")
		model.load_state_dict(torch.load(config['ckpt_dir'] + 'best_model.pth' ))
	return model, optimizer, loss_fn

def train_episode(x_support_tr, y_support_tr, x_len_support_tr, y_support_ind_tr, support_m_tr, sup_text_tr,x_tr, y_tr, x_len_tr, y_ind_tr, tr_mask,tr_text,config,model,loss_fn, optimizer,current_directory):

	kl_loss = torch.nn.KLDivLoss(reduction='batchmean').to(config['device'])
	all_classes = np.unique(y_tr)
	# Store original indexes of each class in dictionary
	idx_key = {}
	for val in all_classes:
		index = np.where(y_tr == val)[0]
		idx_key[val] = index

	best_acc = 0
	avg_loss = 0.0
	avg_acc = 0.0
	
	early_stop_count =0.0
	prev_loss = float("inf")
	for episode in range (config['num_episodes']):
					
		support_feature, support_class, support_len, support_ind, support_mask, support_text_tr, query_feature, query_class, query_len, query_ind, query_mask, query_text = utils.create_query_support(x_support_tr, y_support_tr, x_len_support_tr, y_support_ind_tr, support_m_tr, sup_text_tr, x_tr, y_tr, x_len_tr, y_ind_tr, tr_mask,tr_text, config, config['num_class'])

		support_feature, support_id, support_ind, support_len, support_mask = convert_to_tensor(support_feature, support_class, support_ind, support_len, support_mask, config['device'])
		query_feature, query_id, query_ind, query_len, query_mask = convert_to_tensor(query_feature, query_class, query_ind, query_len, query_mask, config['device'])
		prediction,incons_loss, support_attn, query_attn = model.forward(support_feature, support_len, support_mask, query_feature, query_len, query_mask)
								
		loss_val = compute_loss(prediction, support_attn, support_ind, support_len, support_id, query_attn, query_class, query_ind, query_len, config, loss_fn, kl_loss)
		avg_loss += loss_val.item()
		optimizer.zero_grad()
		loss_val.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(),1)
		optimizer.step()

		tr_pred = np.argmax(prediction.cpu().detach().clone(), 1)
		acc = accuracy_score(query_class.reshape(-1,), tr_pred) 
		avg_acc += acc
		if ((episode % 100 == 99)):
			cur_acc = avg_acc / (episode + 1)
			if (cur_acc >= best_acc):
				print ("--Saving model --")
				best_acc = cur_acc
				torch.save(model.state_dict(), current_directory + 'best_model.pth')
			 
			print ("Average accuracy after", episode + 1, 'episodes: ', avg_acc / (episode+1))
			print ("Average loss after", episode + 1, 'episodes: ', avg_loss / (episode+1))
						
def train(data,config, current_directory):
	x_tr = data['x_tr']
	y_tr = data['y_tr']
	y_ind_tr = utils.create_index( y_tr)
	tr_mask = data['mask_tr']
	x_len_tr = data['len_tr']
	x_text = data['text_tr']
	
	x_support_tr, y_support_tr, x_len_support_tr, support_m_tr, support_text_tr = utils.load_support(data, True)
	y_support_ind_tr = utils.create_index(y_support_tr)
	
	embedding = data['embedding']
	model, optimizer, loss_fn = load_model(config, embedding)
	model.train()

	train_episode(x_support_tr, y_support_tr, x_len_support_tr, y_support_ind_tr, support_m_tr, support_text_tr, x_tr, y_tr, x_len_tr, y_ind_tr, tr_mask, x_text, config, model,loss_fn,optimizer, current_directory)

	return loss_fn

if __name__ == "__main__": 
	# load settings
	config = parse_argument()
	print ("Config", config)
	# load data
	print("\n")
	print ("----------------- START NEW MODEL --------------------------")
	# Training cycle
	MakeDirectory(config['ckpt_dir'])
	
	joint_accs = np.array([], dtype=np.float64) 
	seen_accs = np.array([], dtype=np.float64)
	novel_accs = np.array([], dtype=np.float64)
	h_accs = np.array([], dtype=np.float64)
	 
	if (config['dataset'] == 'NLUE'):
		num_run = config['num_fold']
	elif (config['dataset'] =='SNIPS'):
		num_run = config['num_run']
	
	seed_vals = np.arange(num_run)
	run_acc =[]
	for n_r in range(num_run): 
		if (config['dataset'] == 'NLUE'):
			if (config['num_fold'] != -1):
				n_r = config['num_fold'] - 1
		set_seed(seed_vals[n_r])

		print ("============================================")
		print ("---------Experiment Run START ------- ", n_r)
		print ("----- Evaluation Type: ", config['eps'])
		data = read_datasets(config['num_samples_per_class'], config['dataset'], n_r + 1, config['src'], config['tgt'], config['fasttext_path'])

		embedding = data['embedding']
		current_directory = config['ckpt_dir'] + 'run_' + str(n_r) + '/'
		MakeDirectory(current_directory)
		print('Initializing Variables')
	 	#--------Start training cycle---------------------
		loss_fn = train(data,config, current_directory) 
		best_model = sman.SMAN(config, embedding).to(config['device'])
		print ("--------Loading pretrained weights ----------")
		best_model.load_state_dict(torch.load(current_directory + 'best_model.pth'))
		print ("------- Done loading pretrained weights -------")
		
		if (config['eps'] == 'noneps'):
			avg_acc = evaluate_nonepisode(data, config, best_model,loss_fn, eval=False) 
			print ("----End of the run " + str(n_r) + " ---------") 
		else:
			avg_acc = evaluate_episode(data, config, best_model, loss_fn, eval=False)
		run_acc.append(avg_acc)
	run_acc = np.array(run_acc)
	print("------Experiment Summary --------")
	print ("Avg accuracy:",np.average(run_acc))
	print ("STD:", np.std(run_acc))
	print ("---------- END MODEL --------------------------")
	print ("================================================")
