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
		  q   q   q	   q
			|  |   |	   |
			  \ |   |	  /
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

class GlobalAttention(nn.Module):
	def __init__(self, dim, dim_out=None):
		super(GlobalAttention, self).__init__()
		if dim_out is None: dim_out = dim
		self.linear_in = nn.Linear(dim, dim, bias=False)
		self.sm = nn.Softmax()
		self.linear_out = nn.Linear(dim*2, dim_out, bias=False)
		self.tanh = nn.Tanh()
		self.mask = None

	def applyMask(self, mask):
		self.mask = mask

	def forward(self, input, context):
		"""
		input: batch x dim
		context: batch x sourceL x dim
		"""
		targetT = self.linear_in(input).unsqueeze(2)  # batch x dim x 1

		# Get attention
		attn = torch.bmm(context, targetT).squeeze(2)  # batch x sourceL
		if self.mask is not None:
			attn.data.masked_fill_(self.mask, -float('inf'))
		attn = self.sm(attn)
		attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL

		weightedContext = torch.bmm(attn3, context).squeeze(1)  # batch x dim
		contextCombined = torch.cat((weightedContext, input), 1)

		contextOutput = self.tanh(self.linear_out(contextCombined))

		return contextOutput, attn

class GlobalAttention_mod(nn.Module):
	def __init__(self, dim):
		super(GlobalAttention_mod, self).__init__()
		self.linear_in = nn.Linear(dim, dim, bias=False)
		self.sm = nn.Softmax()
		self.linear_out = nn.Linear(dim*2, dim, bias=False)
		self.tanh = nn.Tanh()
		self.mask = None

	def applyMask(self, mask):
		self.mask = mask

	def forward(self, input, context):
		"""
		input: batch x dim
		context: batch x sourceL x dim
		"""
		targetT = self.linear_in(input).unsqueeze(2)  # batch x dim x 1

		# Get attention
		attn = torch.bmm(context, targetT).squeeze(2)  # batch x sourceL
		if self.mask is not None:
			attn.data.masked_fill_(self.mask, -float('inf'))
		attn = self.sm(attn)
		attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL

		weightedContext = torch.bmm(attn3, context).squeeze(1)  # batch x dim
		contextCombined = torch.cat((weightedContext, input), 1)

		contextOutput = self.tanh(self.linear_out(contextCombined))

		return contextOutput, attn, weightedContext

class Attention_weight(nn.Module):
	def __init__(self, dim_in, dim_out):
		super(Attention_weight, self).__init__()
		self.linear_in = nn.Linear(dim_in, dim_out, bias=False)
		self.sm = nn.Softmax()
		#self.linear_out = nn.Linear(dim*2, dim, bias=False)
		self.tanh = nn.Tanh()
		self.mask = None

	def applyMask(self, mask):
		self.mask = mask

	def forward(self, input, context):
		"""
		input: batch x dim
		context: batch x sourceL x dim
		"""
		targetT = self.linear_in(input).unsqueeze(2)  # batch x dim x 1

		# Get attention
		attn = torch.bmm(context, targetT).squeeze(2)  # batch x sourceL
		if self.mask is not None:
			attn.data.masked_fill_(self.mask, -float('inf'))
		attn = self.sm(attn)
		attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL

		weightedContext = torch.bmm(attn3, context).squeeze(1)  # batch x dim
		#contextCombined = torch.cat((weightedContext, input), 1)

		#contextOutput = self.tanh(self.linear_out(contextCombined))

		return weightedContext, attn

class Attention_simple(nn.Module):
	def __init__(self, dim):
		super(Attention_simple, self).__init__()
		self.linear = nn.Linear(dim, dim, bias=False)
		self.sm = nn.Softmax()
		self.tanh = nn.Tanh()
		self.mask = None
		self.linear_V = nn.Linear(dim, 1)

	def applyMask(self, mask):
		self.mask = mask

	def forward(self, context):
		"""
		input: batch x dim
		context: batch x sourceL x dim
		"""
		nbatch, nsent, ndim = context.size()

		l_ctxt = self.linear(context.view(-1, ndim))
		l_ctxt = self.tanh(l_ctxt)
		
		# Get attention
		attn = self.linear_V(l_ctxt)
		attn = attn.view(nbatch, nsent)
		
		if self.mask is not None:
			attn.data.masked_fill_(self.mask, -float('inf'))
		attn = self.sm(attn)

		weightedContext = torch.bmm(attn.unsqueeze(1), context).squeeze(1)
 

		return weightedContext, attn

class Attention_simple_lin(nn.Module):
	def __init__(self, dim):
		super(Attention_simple_lin, self).__init__()
		self.linear = nn.Linear(dim, dim)
		self.linear_ctx = nn.Linear(dim, dim)
		self.sm = nn.Softmax()
		self.tanh = nn.Tanh()
		self.mask = None
		self.linear_V = nn.Linear(dim, 1)

	def applyMask(self, mask):
		self.mask = mask

	def forward(self, context):
		"""
		input: batch x dim
		context: batch x sourceL x dim
		"""
		nbatch, nsent, ndim = context.size()

		l_ctxt = self.linear(context.view(-1, ndim))
		l_ctxt = self.tanh(l_ctxt)

		l_ctxt_2 = self.linear_ctx(context.view(-1, ndim))
		l_ctxt_2 = self.tanh(l_ctxt)
		l_ctxt_2 = l_ctxt_2.view(nbatch, nsent, ndim)
		
		# Get attention
		attn = self.linear_V(l_ctxt)
		attn = attn.view(nbatch, nsent)
		
		if self.mask is not None:
			attn.data.masked_fill_(self.mask, -float('inf'))
		attn = self.sm(attn)

		weightedContext = torch.bmm(attn.unsqueeze(1), l_ctxt_2).squeeze(1)
 

		return weightedContext, attn

