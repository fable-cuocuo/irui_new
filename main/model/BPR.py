import torch
import torch.nn as nn


class BPR(nn.Module):
	def __init__(self, user_num, item_num, factor_num):
		super(BPR, self).__init__()
		"""
		user_num: number of users;
		item_num: number of items;
		factor_num: number of predictive factors.
		embedding为32维的
		"""		
		#user_nums*32 6040*32
		self.embed_user = nn.Embedding(user_num, factor_num)
		#3706*32
		self.embed_item = nn.Embedding(item_num, factor_num)
		#print(self.embed_item.size())
		#std=0.01的正态分布填入embedding的weight，weight:ueser_nums*32
		#假设theta服从正态分布
		nn.init.normal_(self.embed_user.weight, std=0.01)
		nn.init.normal_(self.embed_item.weight, std=0.01)

	def forward(self, user, item_i, item_j):
		#train loader 中限制了bs，所以大小是bs
		user = self.embed_user(user)
		item_i = self.embed_item(item_i)
		item_j = self.embed_item(item_j)
		#-1 减一维度
		#user_nums*32 * item_nums*32 = 
		print(user.size())
		print(item_i.size())
		print((user * item_i).size())
		prediction_i = (user * item_i).sum(dim=-1)
		
		print(prediction_i.size())
		prediction_j = (user * item_j).sum(dim=-1)
		return prediction_i, prediction_j
