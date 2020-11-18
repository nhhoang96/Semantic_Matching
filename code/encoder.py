import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import warnings

class Encoder(nn.Module):
	"""
			Multi-head attention encoder
	"""

	def __init__(self, config, embedding,input_channels=1):
		super(Encoder, self).__init__()

		# Generic parameters
		self.device = config['device']

		self.drop = nn.Dropout(0.2)
		emb_layer,num_emb, emb_dim = create_emb_layer(embedding)
		self.embed = emb_layer
		self.hidden_size=config['hidden_size']
		self.r = config['r']

		self.d_a = config['d_a']
		self.bilstm = nn.LSTM(emb_dim, self.hidden_size, bidirectional=True, batch_first=True)
		self.ws1 = nn.Linear(self.hidden_size * 2, self.d_a, bias=False)
		self.ws2 = nn.Linear(self.d_a, self.r, bias=False)
		self.tanh = nn.Tanh()
		self.softmax = nn.Softmax(dim=1)
	
	def forward(self,input, len, mask,segmented_input = None, segment_id=None, segmented_len = None, segmented_mask = None):
		warnings.filterwarnings('ignore')
		
		#Pack padded sequence process
		sorted_inputs, sorted_sequence_lengths, restoration_indices,permut_index = sort_batch_by_length(input, len, self.device)
		embedded_inputs = self.embed(sorted_inputs)

		sorted_mask = mask.index_select(0, permut_index)
		packed_emb = pack_padded_sequence(embedded_inputs, sorted_sequence_lengths, batch_first=True)

		#Initialize hidden states
		h_0 = Variable(torch.zeros(2, input.shape[0], self.hidden_size))
		c_0 = Variable(torch.zeros(2, input.shape[0], self.hidden_size))
		h_0 = h_0.to(self.device)
		c_0 = c_0.to(self.device)
		
		outp = self.bilstm(packed_emb, (h_0, c_0))[0] ## [bsz, len, d_h * 2]
		mod_outp = pad_packed_sequence(outp)[0].transpose(0,1).contiguous()
		mod_outp = mod_outp.index_select(0, restoration_indices)
		mod_outp = mod_outp.contiguous()
		noattn_rep = self.drop(mod_outp) #bsz, len, d_h

		# bsz, #seg, len, d_h	
		size = mod_outp.size()
		compressed_embeddings = mod_outp.view(-1, size[2])	# [bsz * len, d_h * 2]
		compressed_embeddings = self.drop(compressed_embeddings)
		hbar = self.tanh(self.ws2(self.ws1(compressed_embeddings)))
		alphas = hbar.view(size[0], size[1], -1)			# [bsz, len, hop]
		attention = torch.transpose(alphas, 1, 2)
		attention = attention.contiguous ()  # [bsz, hop, len]
		
		current_mask = torch.narrow(mask, dim=1, start=0, length=attention.shape[-1])
		repeated_mask = current_mask.unsqueeze(1).repeat(1,attention.shape[1], 1)
		masked_attention = masked_softmax(attention, repeated_mask,2) #[bsz, hop, len]
		
		multihead_rep = torch.bmm(masked_attention, mod_outp) #[bsz, hop,d_h*2]
		sentence_embedding = torch.sum(multihead_rep,1)/ self.r
		
		return sentence_embedding,masked_attention,multihead_rep,noattn_rep


#---Helper function -----#
def sort_batch_by_length(tensor: torch.Tensor, sequence_lengths: torch.Tensor, device):
	"""
	Sort a batch first tensor by some specified lengths.
	Parameters
	----------
	tensor : torch.FloatTensor, required.
A batch first Pytorch tensor.
	sequence_lengths : torch.LongTensor, required.
A tensor representing the lengths of some dimension of the tensor which
we want to sort by.
	Returns
	------ 
	sorted_tensor : torch.FloatTensor
The original tensor sorted along the batch dimension with respect to sequence_lengths.
	sorted_sequence_lengths : torch.LongTensor
The original sequence_lengths sorted by decreasing size.
	restoration_indices : torch.LongTensor
Indices into the sorted_tensor such that
``sorted_tensor.index_select(0, restoration_indices) == original_tensor``
	permuation_index : torch.LongTensor
The indices used to sort the tensor. This is useful if you want to sort many
tensors using the same ordering.
	"""

	sorted_sequence_lengths, permutation_index = sequence_lengths.sort(0, descending=True)

	permutation_index = permutation_index.to(device)
	sorted_tensor = tensor.index_select(0, permutation_index)
	index_range = Variable(torch.arange(0, len(sequence_lengths)).long())
	index_range = index_range.to(device)
	# This is the equivalent of zipping with index, sorting by the original
	# sequence lengths and returning the now sorted indices.
	#_, reverse_mapping = permutation_index.sort(0, descending=False)
	
	_, reverse_mapping = permutation_index.sort(0, descending=False)
	restoration_indices = index_range.index_select(0, reverse_mapping)

	return sorted_tensor, sorted_sequence_lengths, restoration_indices, permutation_index

def create_emb_layer(weights_matrix, non_trainable=False):
	weights_matrix = torch.from_numpy(weights_matrix)
	num_embeddings, embedding_dim = weights_matrix.size()
	emb_layer = nn.Embedding(num_embeddings, embedding_dim)
	emb_layer.load_state_dict({'weight': weights_matrix})
	if non_trainable:
					emb_layer.weight.requires_grad = False

	return emb_layer, num_embeddings, embedding_dim


def masked_softmax(vec, mask, dim=1):
	masked_vec = vec * mask.float()
	max_vec = torch.max(masked_vec, dim=dim, keepdim=True)[0]
	exps = torch.exp(masked_vec-max_vec)
	masked_exps = exps * mask.float()
	masked_sums = masked_exps.sum(dim, keepdim=True)
	zeros=(masked_sums == 0)
	masked_sums += zeros.float()
	return masked_exps/masked_sums



