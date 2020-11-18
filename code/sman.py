# Acknowledgement: https://github.com/galsang/BIMPM-pytorch
#                  https://github.com/ZhixiuYe/MLMAN

import torch
import torch.nn as nn
import torch.nn.functional as F
import encoder

class SMAN(nn.Module):
        """
        Multi-level Matching and Aggregation
        """
        def __init__(self, config, embedding):
                super(SMAN, self).__init__()
                # Generic parameters
                self.device = config['device']

                self.k_per = config['num_samples_per_class']
                self.q_per = config['num_query_per_class']
                self.drop = nn.Dropout(0.2)

                self.hidden_size = config['hidden_size']                  
                self.encoder = encoder.Encoder(config, embedding)
   
                self.l = config['l']
                self.r = config['r']

                input_proj = self.hidden_size * 8
                output_proj = self.hidden_size * 2
                
                input_multilayer = self.hidden_size * 4
                # For 2-way interaction 
                input_multilayer = input_multilayer*2
                output_multilayer = self.hidden_size
 
                self.multilayer = nn.Sequential(
                        nn.Linear(input_multilayer, output_multilayer),
                        nn.ReLU(),
                        nn.Linear(output_multilayer,1)
                )
                #---Determine size for Aggregation Layer -----#
                head_size=8
                
                for i in range(1, head_size + 1):
                    setattr(self, f'head_w{i}',
                      nn.Parameter(torch.rand(self.l, self.hidden_size))) 
                    self.aggregation_head = nn.LSTM(
                                input_size=self.l *head_size,
                                hidden_size=self.hidden_size,
                                num_layers=1,
                                bidirectional=True,
                                batch_first=True
                      )

        # ---Helper function for matching -----# 
        def mp_matching_func(self,v1, v2, w):
                """
                :param v1: (batch, seq_len, hidden_size)
                :param v2: (batch, seq_len, hidden_size) or (batch, hidden_size)
                :param w: (l, hidden_size)
                :return: (batch, l)
                """
                seq_len = v1.size(1)
                if len(v2.size()) == 2:
                        v2 = torch.stack([v2] * seq_len, dim=1)
                m = []
                for i in range(self.l):
                        # v1: (batch, seq_len, hidden_size)
                        # v2: (batch, seq_len, hidden_size)
                        # w: (1, 1, hidden_size)
                        # -> (batch, seq_len)
                        m.append(F.cosine_similarity(w[i].view(1, 1, -1) * v1, w[i].view(1, 1, -1) * v2, dim=2))
                        # list of (batch, seq_len) -> (batch, seq_len, l)
                
                m = torch.stack(m, dim=2)
                return m

        def mp_matching_func_pairwise(self,v1, v2, w):
                """
                :param v1: (batch, seq_len1, hidden_size)
                :param v2: (batch, seq_len2, hidden_size)
                :param w: (l, hidden_size)
                :return: (batch, l, seq_len1, seq_len2)
                """
                m = []
                for i in range(self.l):
                        # (1, 1, hidden_size)
                        w_i = w[i].view(1, 1, -1)
                        # (batch, seq_len1, hidden_size), (batch, seq_len2, hidden_size)
                        v1, v2 = w_i * v1, w_i * v2
                        # (batch, seq_len, hidden_size->1)
                        v1_norm = v1.norm(p=2, dim=2, keepdim=True)
                        v2_norm = v2.norm(p=2, dim=2, keepdim=True)
                        # (batch, seq_len1, seq_len2)
                        n = torch.matmul(v1, v2.permute(0, 2, 1))
                        d = v1_norm * v2_norm.permute(0, 2, 1)
                        m.append(div_with_small_value(n, d))
                # list of (batch, seq_len1, seq_len2) -> (batch, seq_len1, seq_len2, l)
                m = torch.stack(m, dim=3)

                return m
       
        def attention(self,v1, v2):
            """
            :param v1: (batch, seq_len1, hidden_size)
            :param v2: (batch, seq_len2, hidden_size)
            :return: (batch, seq_len1, seq_len2)
            """
            # (batch, seq_len1, 1)
            v1_norm = v1.norm(p=2, dim=2, keepdim=True)
            # (batch, 1, seq_len2)
            v2_norm = v2.norm(p=2, dim=2, keepdim=True).permute(0, 2, 1)

            # (batch, seq_len1, seq_len2)
            a = torch.bmm(v1, v2.permute(0, 2, 1))
            d = v1_norm * v2_norm

            return div_with_small_value(a, d)
       
        def headwise_match(self,support_fw, support_bw, query_fw, query_bw,w1,w2):
            num_head = support_fw.shape[1]

            # (batch, seq_len1, seq_len2, l)
            mv_max_fw = self.mp_matching_func_pairwise(support_fw, query_fw, w1)
            mv_max_bw = self.mp_matching_func_pairwise(support_bw, query_bw, w2)
            
            p_full_fw_list = []
            h_full_fw_list = []
            p_full_bw_list = []
            h_full_bw_list = []
            for i in range (num_head):
                p_full_fw_list.append(mv_max_fw[:,i,i,:].unsqueeze(1))
                p_full_bw_list.append(mv_max_bw[:,i,i,:].unsqueeze(1))
                h_full_fw_list.append(mv_max_fw[:,i,i,:].unsqueeze(1))
                h_full_bw_list.append(mv_max_bw[:,i,i,:].unsqueeze(1))
            
            mv_p_full_fw = torch.cat((p_full_fw_list),1)
            mv_p_full_bw = torch.cat((p_full_bw_list),1)
            mv_h_full_fw = torch.cat((h_full_fw_list),1)
            mv_h_full_bw = torch.cat((h_full_bw_list),1)

            return mv_p_full_fw, mv_p_full_bw, mv_h_full_fw, mv_h_full_bw
             
        def maxpool_match(self,support_fw, support_bw, query_fw, query_bw,w1,w2):     
            # (batch, seq_len1, seq_len2, l)
            mv_max_fw = self.mp_matching_func_pairwise(support_fw, query_fw, w1)
            mv_max_bw = self.mp_matching_func_pairwise(support_bw, query_bw, w2)
            
            # (batch, seq_len, l)
            mv_p_max_fw, _ = mv_max_fw.max(dim=2)
            mv_p_max_bw, _ = mv_max_bw.max(dim=2)
            mv_h_max_fw, _ = mv_max_fw.max(dim=1)
            mv_h_max_bw, _ = mv_max_bw.max(dim=1)
            return mv_p_max_fw, mv_p_max_bw, mv_h_max_fw, mv_h_max_bw
        
        #support_fw/bw: [bsz, r, D]
        # p_thres: [bsz,r, 1]
        def attentive_aggr(self,support_fw, support_bw, query_fw, query_bw, p_thres, h_thres):
            
            #(bsz,r,1) -> (bsz, r, D)
            p_thres = p_thres.repeat(1,1,conf_p_fw.shape[-1])

            # (batch, seq_len1, seq_len2)
            att_fw = self.attention(support_fw, query_fw)
            att_bw = self.attention(support_bw, query_bw)
            
            # (batch, seq_len2, hidden_size) -> (batch, 1, seq_len2, hidden_size)
            # (batch, seq_len1, seq_len2) -> (batch, seq_len1, seq_len2, 1)
            # -> (batch, seq_len1, seq_len2, hidden_size)
            att_h_fw = query_fw.unsqueeze(1) * att_fw.unsqueeze(3)
            att_h_bw = query_bw.unsqueeze(1) * att_bw.unsqueeze(3)
            # (batch, seq_len1, hidden_size) -> (batch, seq_len1, 1, hidden_size)
            # (batch, seq_len1, seq_len2) -> (batch, seq_len1, seq_len2, 1)
            # -> (batch, seq_len1, seq_len2, hidden_size)
            att_p_fw = support_fw.unsqueeze(2) * att_fw.unsqueeze(3)
            att_p_bw = support_bw.unsqueeze(2) * att_bw.unsqueeze(3)

            # (batch, seq_len1, hidden_size) / (batch, seq_len1, 1) -> (batch, seq_len1, hidden_size)
            att_mean_h_fw = div_with_small_value(att_h_fw.sum(dim=2), att_fw.sum(dim=2, keepdim=True))
            att_mean_h_bw = div_with_small_value(att_h_bw.sum(dim=2), att_bw.sum(dim=2, keepdim=True))

            # (batch, seq_len2, hidden_size) / (batch, seq_len2, 1) -> (batch, seq_len2, hidden_size)
            att_mean_p_fw = div_with_small_value(att_p_fw.sum(dim=1), att_fw.sum(dim=1, keepdim=True).permute(0, 2, 1))
            att_mean_p_bw = div_with_small_value(att_p_bw.sum(dim=1), att_bw.sum(dim=1, keepdim=True).permute(0, 2, 1))

            # (batch, seq_len, l)
            mv_p_att_mean_fw = self.mp_matching_func(support_fw, att_mean_h_fw, self.mp_w5)
            mv_p_att_mean_bw = self.mp_matching_func(support_bw, att_mean_h_bw, self.mp_w6)
            mv_h_att_mean_fw = self.mp_matching_func(query_fw, att_mean_p_fw, self.mp_w5)
            mv_h_att_mean_bw = self.mp_matching_func(query_bw, att_mean_p_bw, self.mp_w6)

            return mv_p_att_mean_fw, mv_p_att_mean_bw, mv_h_att_mean_fw, mv_h_att_mean_bw, att_h_fw, att_h_bw, att_p_fw, att_p_bw
        
        def attentive_match(self,support_fw, support_bw, query_fw, query_bw,w5, w6):

            # (batch, seq_len1, seq_len2)
            att_fw = self.attention(support_fw, query_fw)
            att_bw = self.attention(support_bw, query_bw)
            # (batch, seq_len2, hidden_size) -> (batch, 1, seq_len2, hidden_size)
            # (batch, seq_len1, seq_len2) -> (batch, seq_len1, seq_len2, 1)
            # -> (batch, seq_len1, seq_len2, hidden_size)
            att_h_fw = query_fw.unsqueeze(1) * att_fw.unsqueeze(3)
            att_h_bw = query_bw.unsqueeze(1) * att_bw.unsqueeze(3)
            # (batch, seq_len1, hidden_size) -> (batch, seq_len1, 1, hidden_size)
            # (batch, seq_len1, seq_len2) -> (batch, seq_len1, seq_len2, 1)
            # -> (batch, seq_len1, seq_len2, hidden_size)
            att_p_fw = support_fw.unsqueeze(2) * att_fw.unsqueeze(3)
            att_p_bw = support_bw.unsqueeze(2) * att_bw.unsqueeze(3)

            # (batch, seq_len1, hidden_size) / (batch, seq_len1, 1) -> (batch, seq_len1, hidden_size)
            att_mean_h_fw = div_with_small_value(att_h_fw.sum(dim=2), att_fw.sum(dim=2, keepdim=True))
            att_mean_h_bw = div_with_small_value(att_h_bw.sum(dim=2), att_bw.sum(dim=2, keepdim=True))

            # (batch, seq_len2, hidden_size) / (batch, seq_len2, 1) -> (batch, seq_len2, hidden_size)
            att_mean_p_fw = div_with_small_value(att_p_fw.sum(dim=1), att_fw.sum(dim=1, keepdim=True).permute(0, 2, 1))
            att_mean_p_bw = div_with_small_value(att_p_bw.sum(dim=1), att_bw.sum(dim=1, keepdim=True).permute(0, 2, 1))

            # (batch, seq_len, l)
            mv_p_att_mean_fw = self.mp_matching_func(support_fw, att_mean_h_fw, w5)
            mv_p_att_mean_bw = self.mp_matching_func(support_bw, att_mean_h_bw, w6)
            mv_h_att_mean_fw = self.mp_matching_func(query_fw, att_mean_p_fw, w5)
            mv_h_att_mean_bw = self.mp_matching_func(query_bw, att_mean_p_bw, w6)

            return mv_p_att_mean_fw, mv_p_att_mean_bw, mv_h_att_mean_fw, mv_h_att_mean_bw, att_h_fw, att_h_bw, att_p_fw, att_p_bw
        
        def max_attentive_match(self,support_fw, support_bw, query_fw, query_bw, att_h_fw, att_h_bw, att_p_fw, att_p_bw, w7, w8):
        
            # (batch, seq_len1, hidden_size)
            att_max_h_fw, _ = att_h_fw.max(dim=2)
            att_max_h_bw, _ = att_h_bw.max(dim=2)
            # (batch, seq_len2, hidden_size)
            att_max_p_fw, _ = att_p_fw.max(dim=1)
            att_max_p_bw, _ = att_p_bw.max(dim=1)

            # (batch, seq_len, l)
            mv_p_att_max_fw = self.mp_matching_func(support_fw, att_max_h_fw, w7)
            mv_p_att_max_bw = self.mp_matching_func(support_bw, att_max_h_bw, w8)
            mv_h_att_max_fw = self.mp_matching_func(query_fw, att_max_p_fw, w7)
            mv_h_att_max_bw = self.mp_matching_func(query_bw, att_max_p_bw, w8)
            return mv_p_att_max_fw, mv_p_att_max_bw, mv_h_att_max_fw, mv_h_att_max_bw

        def expand_local_matching(self,support_heads, query_heads):
            match_support = []
            match_query = []

            num_samples= support_heads.shape[1]
            support_f = []
            query_f = []
            for i in range (num_samples):
                    support_ = support_heads[:,i,:]
                    support_t = self.head_match(support_, query_heads)
                    query_t = self.head_match(query_heads, support_)
                    
                    support_f.append(support_t.unsqueeze(1))
                    query_f.append(query_t.unsqueeze(1))
                
            support_f = torch.cat(support_f, dim=1)
            query_f = torch.cat(query_f,dim=1)
            query_f = torch.mean(query_f,dim=1)
            
            return support_f, query_f


        # Head match
        # Support: [bsz, r, D_h*2]
        # Query: [bsz, r, D_h*2]
        def head_match(self, enhance_support, enhance_query):

                support_fw, query_fw = enhance_support[:,:,:self.hidden_size], enhance_query[:,:,:self.hidden_size]
                support_bw, query_bw = enhance_support[:,:,self.hidden_size:], enhance_query[:,:,self.hidden_size:]
                 
            	# 1. Head-wise Matching
            	# (batch, seq_len, hidden_size), (batch, hidden_size)
            	# -> (batch, seq_len, l)
                mv_p_full_fw, mv_p_full_bw, mv_h_full_fw, mv_h_full_bw = self.headwise_match(support_fw, support_bw, query_fw,query_bw, self.head_w7,self.head_w8)

		# 2. Maxpooling-Matching
                mv_p_max_fw, mv_p_max_bw, mv_h_max_fw, mv_h_max_bw = self.maxpool_match(support_fw, support_bw, query_fw, query_bw, self.head_w1, self.head_w2)

		# 3. Attentive-Matching
                mv_p_att_mean_fw, mv_p_att_mean_bw, mv_h_att_mean_fw, mv_h_att_mean_bw, att_h_fw, att_h_bw, att_p_fw, att_p_bw = self.attentive_match(support_fw, support_bw, query_fw, query_bw, self.head_w3, self.head_w4)

		# 4. Max-Attentive-Matching
                mv_p_att_max_fw, mv_p_att_max_bw, mv_h_att_max_fw, mv_h_att_max_bw = self.max_attentive_match(support_fw, support_bw, query_fw, query_bw, att_h_fw, att_h_bw, att_p_fw, att_p_bw, self.head_w5, self.head_w6)   

                mv_p = torch.cat(
                    [
	            mv_p_full_fw,
	            mv_p_max_fw, mv_p_att_mean_fw, mv_p_att_max_fw,
	            mv_p_full_bw,
	            mv_p_max_bw, mv_p_att_mean_bw, mv_p_att_max_bw
	            ], dim=2)

                mv_h = torch.cat(
                    [
	            mv_h_full_fw,
                    mv_h_max_fw, mv_h_att_mean_fw, mv_h_att_max_fw,
                    mv_h_full_bw,
                    mv_h_max_bw, mv_h_att_mean_bw, mv_h_att_max_bw,
                    ], dim=2)
                
            # ----- Aggregation Layer -----
            # (batch, seq_len, l * 8) -> (2, batch, hidden_size)
                _, (agg_p_last, _) = self.aggregation_head(mv_p)
                _, (agg_h_last, _) = self.aggregation_head(mv_h)
                agg_p_last = self.drop(agg_p_last)
                agg_h_last = self.drop(agg_h_last)
            
            # 2 * (2, batch, hidden_size) -> 2 * (batch, hidden_size * 2) -> (batch, hidden_size * 4)    
                x = torch.cat(
                        [agg_p_last.permute(1, 0, 2).contiguous().view(-1, self.hidden_size * 2),
                        agg_h_last.permute(1, 0, 2).contiguous().view(-1, self.hidden_size * 2)], dim=1)
                return x


        def forward(self, support_sets, support_len,support_mask, query_sets, query_len, query_mask):
            """
                Param:
                    support_sets: [num_support * num_class (NK) ,features]: support
                    support_labels: [num_support, num_class, 1] dimensional vector
                    query_sets: [num_query (NQ), dimensional vec] 
            """

            num_support = self.k_per #per class
            num_class = support_sets.shape[0] // self.k_per
            num_query = query_sets.shape[0] # per class
            max_len = support_sets.shape[-1]
            batch= 1
        
            #---- Encoder ---------
            aggr_sup, support_attn, support_heads,nonhead_support = self.encoder(support_sets, support_len, support_mask)  
            aggr_query, query_attn, query_heads,nonhead_query = self.encoder(query_sets, query_len, query_mask) 

            nonhead_support, support_heads, support_len, support_mask = reshape_dim(nonhead_support, support_heads,support_len, support_mask,num_support,num_query, num_class, max_len,batch, 'support', self.r)
            nonhead_query, query_heads, query_len, query_mask = reshape_dim(nonhead_query, query_heads, query_len, query_mask, num_support,num_query, num_class, max_len,batch, 'query', self.r)

            # -- Semantic Matching Approach -----#
            enhance_support, enhance_query = self.expand_local_matching(support_heads, query_heads)

            #--------- Instance Matching -----------#
            tmp_query = enhance_query.unsqueeze(1).repeat(1,num_support,1)
            cat_seq = torch.cat([tmp_query, enhance_support], 2)
            alpha = self.multilayer(cat_seq)
            one_enhance_support = (enhance_support.transpose(1,2) @ F.softmax(alpha,1)).squeeze(2)
            
            #--------- Class Matching -------------------#
            J_incon = torch.sum((one_enhance_support.unsqueeze(1) - enhance_support) **2, 2).mean()
            cat_seq = torch.cat([enhance_query, one_enhance_support], 1)
            logits = self.multilayer(cat_seq)
            logits = logits.view(num_query, num_class)
            
            return logits, J_incon, support_attn, query_attn

#-----Helper function ---------------#
def reshape_dim(nonhead, head, len, mask, num_support,num_query, num_class, max_len,batch, type, r):
        
    """
        nonhead [bs, len, D_W]: contextualized embedding (nonpp)
        head [bs, r, D_W]: head embedding
        len [bs,]: len of each sentence
        mask [bs, max_len]: mask for each sentence
    """
    hidden_dim = nonhead.shape[-1]
    if (type == 'support'):
        # Reshape length (feature-based)
        out_len = len.view(-1,).contiguous()
        out_len = out_len.repeat(num_query,)
                        
        # Reshape mask(feature-based)
        out_mask = mask.view(-1,max_len).contiguous()
        out_mask = out_mask.repeat(num_query,1)
                                        
        # Reshape embeddings (feature based) 
        out_nonhead = nonhead.unsqueeze(0).view(num_class, num_support, -1, hidden_dim).contiguous()
        out_nonhead = out_nonhead.unsqueeze(0).view(1,num_class,num_support,-1, hidden_dim).contiguous()
        out_nonhead = out_nonhead.unsqueeze(0).view(batch,1, num_class,num_support,-1,hidden_dim).contiguous()
        out_nonhead = out_nonhead.expand(batch, num_query,num_class,num_support, -1, hidden_dim).contiguous()
        out_nonhead = out_nonhead.view(batch*num_query*num_class, num_support, -1,hidden_dim).contiguous()
        out_head = head.repeat(num_query,1,1)

        out_head = head.unsqueeze(0).view(num_class, num_support, -1, hidden_dim).contiguous()
        out_head = out_head.unsqueeze(0).view(1,num_class,num_support,-1,hidden_dim).contiguous()
        out_head = out_head.unsqueeze(0).view(batch,1, num_class,num_support,-1,hidden_dim).contiguous()
        out_head = out_head.expand(batch, num_query,num_class,num_support, -1, hidden_dim).contiguous()
        out_head = out_head.view(batch*num_query*num_class,num_support,-1, hidden_dim).contiguous() 
    
    else: # Query
        out_len = len.view(-1,).contiguous()
        out_len =out_len.repeat(num_class,)

        out_mask = mask.view(-1,max_len).contiguous()
        out_mask =out_mask.repeat(num_class,1)
        
        out_nonhead = nonhead.unsqueeze(1).view(num_query, 1, -1, hidden_dim).contiguous() # NQ, 1,len, D
        out_nonhead = out_nonhead.unsqueeze(0).view(batch, num_query,1,-1,hidden_dim).contiguous()
        out_nonhead = out_nonhead.expand(batch, num_query,num_class,-1, hidden_dim).contiguous()
        out_nonhead = out_nonhead.view(batch*num_query*num_class,-1, hidden_dim) #NQN, max_len, D)
        
        out_head = head.unsqueeze(1).view(num_query, 1, -1, hidden_dim).contiguous() # NQ, 1,len, D
        out_head = out_head.unsqueeze(0).view(batch, num_query,1,-1,hidden_dim).contiguous()
        out_head = out_head.expand(batch, num_query,num_class,-1, hidden_dim).contiguous()
        out_head = out_head.view(batch*num_query*num_class,-1, hidden_dim) #NQN, max_len, D)
    return out_nonhead, out_head, out_len, out_mask

def div_with_small_value(n, d, eps=1e-8):
    # too small values are replaced by 1e-8 to prevent it from exploding.
    d = d * (d > eps).float() + eps * (d <= eps).float()
    return n / d
