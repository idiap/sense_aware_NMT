'''
This software is derived from the OpenNMT project at 
https://github.com/OpenNMT/OpenNMT.
Modified 2017, Idiap Research Institute, http://www.idiap.ch/ (Xiao PU)
'''


"""
Global attention takes a matrix and a query vector. It
then computes a parameterized convex combination of the matrix
based on the input query.


        H_1 H_2 H_3 ... H_n
          q   q   q       q
            |  |   |       |
              \ |   |      /
                      .....
                  \   |  /
                          a

Constructs a unit mapping.
    $$(H_1 + H_n, q) => (a)$$
    Where H is of `batch x n x dim` and q is of `batch x dim`.

    The full def is  $$\tanh(W_2 [(softmax((W_1 q + b_1) H) H), q] + b_2)$$.:

"""

import torch
import torch.nn as nn
import math


class Attention_Xiao(nn.Module):
	def __init__(self, hidden_size, input_size=None):
		super(Attention_Lesly, self).__init__()
		self.input_size = hidden_size if input_size is None else input_size 
		self.hidden_size = hidden_size
		self.linear_h = nn.Linear(self.input_size, hidden_size, bias=False)
		self.linear_context = nn.Linear(hidden_size, hidden_size, bias=False)
		self.linear_V = nn.Linear(hidden_size, 1)
		self.sm = nn.Softmax()
		self.tanh = nn.Tanh()		
		self.mask = None

	def applyMask(self, mask):
		self.mask = mask

	def forward(self, hidden, context):

		nbatch, nsent, ndim = context.size()


		hidden_lin = self.linear_h(hidden) # batch x dim
		hidden_lin = hidden_lin.unsqueeze(1).expand_as(context) # batch x sentL x dim

		context_lin = self.linear_context(context.view(-1,ndim)) # batch * sentL x dim
		context_lin = context_lin.view(nbatch,-1,ndim) # batch x sentL x dim

		sum_lin = hidden_lin + context_lin
		non_lin = self.tanh(sum_lin.view(-1,ndim)) # batch * sentL x dim
		score = self.linear_V(non_lin) # batch * sentL  x 1
		score = score.view(nbatch, nsent) # batch x sentL
		# print 'BEFORE:'
		# print score
		if self.mask is not None:
			score.data.masked_fill_(self.mask, -float('inf'))
			

		weight = self.sm(score).unsqueeze(1) # batch x 1 x sentL

		weightedContext = torch.bmm(weight, context).squeeze(1)  # batch x dim
	
		return weightedContext, weight
