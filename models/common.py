import torch
from torch.nn import functional as F
from torch.autograd import Variable
import math
import os
import sys
import numpy as np
import pdb
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

INF = 1e10
EPSILON = 1e-10


class LSTMDecoder(nn.Module):

    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(LSTMDecoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            input = self.dropout(input)
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)


def positional_encodings_like(x, t=None):
    if t is None:
        positions = torch.arange(0., x.size(1))
        if x.is_cuda:
            positions = positions.cuda(x.get_device())
    else:
        positions = t
    encodings = torch.zeros(*x.size()[1:])
    if x.is_cuda:
        encodings = encodings.cuda(x.get_device())
    for channel in range(x.size(-1)):
        if channel % 2 == 0:
            encodings[:, channel] = torch.sin(
                positions / 10000 ** (channel / x.size(2)))
        else:
            encodings[:, channel] = torch.cos(
                positions / 10000 ** ((channel - 1) / x.size(2)))
    return Variable(encodings)


# torch.matmul can't do (4, 3, 2) @ (4, 2) -> (4, 3)
def matmul(x, y):
    if x.dim() == y.dim():
        return x @ y
    if x.dim() == y.dim() - 1:
        return (x.unsqueeze(-2) @ y).squeeze(-2)
    return (x @ y.unsqueeze(-2)).squeeze(-2)


def pad_to_match(x, y):
    x_len, y_len = x.size(1), y.size(1)
    if x_len == y_len:
        return x, y
    extra = x.new_ones((x.size(0), abs(y_len - x_len)))
    if x_len < y_len:
        return torch.cat((x, extra), 1), y
    return x, torch.cat((y, extra), 1)


class FreeEmbedding(nn.Module):

    def __init__(self, field, trained_dimension, dropout=0.0, project=True, requires_grad=False):
        super().__init__()
        self.field = field
        self.project = project
        dimension = 0
        pretrained_dimension = field.vocab.vectors.size(-1)
        self.pretrained_embeddings = nn.Embedding(len(field.vocab), pretrained_dimension)
        self.pretrained_embeddings.weight.data = field.vocab.vectors
        self.pretrained_embeddings.weight.requires_grad = requires_grad
        self.requires_grad = requires_grad
        dimension += pretrained_dimension
        # import pdb
        # pdb.set_trace()
        self.project = self.project or pretrained_dimension != trained_dimension
        if self.project:
            self.projection = Feedforward(dimension, trained_dimension)
        dimension = trained_dimension
        self.dropout = nn.Dropout(dropout)
        self.dimension = dimension

    def forward(self, x, lengths=None, device=-1):
        pretrained_embeddings = self.pretrained_embeddings(x).to(x.device)
        return self.projection(pretrained_embeddings) if self.project else pretrained_embeddings

    def set_embeddings(self, w):
        self.pretrained_embeddings.weight.data = w
        self.pretrained_embeddings.weight.requires_grad = self.requires_grad


class FinalEmbedding(nn.Module):

    def __init__(self, field, trained_dimension, dropout=0.0, project=True, requires_grad=False):
        super().__init__()
        self.field = field
        self.project = project
        pretrained_dimension = field.vocab.vectors.size(-1)
        self.pretrained_embeddings = nn.Embedding(len(field.vocab), pretrained_dimension)
        self.pretrained_embeddings.weight.data = field.vocab.vectors
        self.pretrained_embeddings.weight.requires_grad = requires_grad
        self.requires_grad = requires_grad
        self.project = self.project or pretrained_dimension != trained_dimension
        if self.project:
            self.projection = Feedforward(pretrained_dimension,
                                          trained_dimension,
                                          dropout=dropout)
        self.dimension = trained_dimension
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        pretrained_embeddings = self.pretrained_embeddings(x).to(x.device)
        pretrained_embeddings = self.projection(pretrained_embeddings) if self.project else pretrained_embeddings
        return self.dropout(pretrained_embeddings)

    def set_embeddings(self, w):
        self.pretrained_embeddings.weight.data = w
        self.pretrained_embeddings.weight.requires_grad = self.requires_grad


class LayerNorm(nn.Module):

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class ResidualBlock(nn.Module):

    def __init__(self, layer, d_model, dropout_ratio):
        super().__init__()
        self.layer = layer
        self.dropout = nn.Dropout(dropout_ratio)
        self.layernorm = LayerNorm(d_model)

    def forward(self, *x, padding=None):
        return self.layernorm(x[0] + self.dropout(self.layer(*x, padding=padding)))


class SumResidualBlock(nn.Module):

    def __init__(self, layer, d_model, dropout_ratio):
        super().__init__()
        self.layer = layer
        self.dropout = nn.Dropout(dropout_ratio)
        self.layernorm = LayerNorm(d_model)

    def forward(self, *x, padding=None):
        return self.layernorm(x[-1] + self.dropout(self.layer(*x, padding=padding)))


class Attention(nn.Module):

    def __init__(self, d_key, dropout_ratio, causal):
        super().__init__()
        self.scale = math.sqrt(d_key)
        self.dropout = nn.Dropout(dropout_ratio)
        self.causal = causal

    def forward(self, query, key, value, padding=None):
        dot_products = matmul(query, key.transpose(1, 2))
        if query.dim() == 3 and self.causal:
            tri = key.new_ones((key.size(1), key.size(1))).triu(1) * INF
            dot_products.sub_(tri.unsqueeze(0))
        if padding is not None:
            dot_products.masked_fill_(padding.unsqueeze(1).expand_as(dot_products), -INF)
        return matmul(self.dropout(F.softmax(dot_products / self.scale, dim=-1)), value)


class SumAttention(nn.Module):

    def __init__(self, d_key, dropout_ratio, causal):
        super().__init__()
        self.scale = math.sqrt(d_key)
        self.dropout = nn.Dropout(dropout_ratio)
        self.causal = causal

    def forward(self, query, key, value, padding=None):
        # query: (b, q_len, d_head)
        # key: (b, k_len, d_head)
        # value: (b, v_len, d_head)
        # padding: (b, v_len)
        dot_products = matmul(key, query.transpose(1, 2))
        if query.dim() == 3 and self.causal:
            tri = key.new_ones((key.size(1), key.size(1))).triu(1) * INF
            dot_products.sub_(tri.unsqueeze(0))
        # pdb.set_trace()
        if padding is not None:
            # dot_products: (b, q_len, k_len)
            dot_products.masked_fill_(padding.unsqueeze(-1).expand_as(dot_products), -INF)
        value_socre = F.softmax(torch.sum(dot_products, dim=-1) / self.scale, dim=1).unsqueeze(-1)
        scaled_value = value_socre * value
        return scaled_value

        # return matmul(self.dropout(F.softmax(dot_products / self.scale, dim=-1)), value)


class MultiHead(nn.Module):

    def __init__(self, d_key, d_value, n_heads, dropout_ratio, causal=False):
        super().__init__()
        self.attention = Attention(d_key, dropout_ratio, causal=causal)
        self.wq = Linear(d_key, d_key, bias=False)
        self.wk = Linear(d_key, d_key, bias=False)
        self.wv = Linear(d_value, d_value, bias=False)
        self.n_heads = n_heads

    def forward(self, query, key, value, padding=None):
        query, key, value = self.wq(query), self.wk(key), self.wv(value)
        query, key, value = (
            x.chunk(self.n_heads, -1) for x in (query, key, value))
        return torch.cat([self.attention(q, k, v, padding=padding)
                          for q, k, v in zip(query, key, value)], -1)


class SumMultiHead(nn.Module):

    def __init__(self, d_key, d_value, n_heads, dropout_ratio, causal=False):
        super().__init__()
        self.attention = SumAttention(d_key, dropout_ratio, causal=causal)
        self.wq = Linear(d_key, d_key, bias=False)
        self.wk = Linear(d_key, d_key, bias=False)
        self.wv = Linear(d_value, d_value, bias=False)
        self.n_heads = n_heads

    def forward(self, query, key, value, padding=None):
        query, key, value = self.wq(query), self.wk(key), self.wv(value)
        query, key, value = (
            x.chunk(self.n_heads, -1) for x in (query, key, value))
        return torch.cat([self.attention(q, k, v, padding=padding)
                          for q, k, v in zip(query, key, value)], -1)


class LinearReLU(nn.Module):

    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.feedforward = Feedforward(d_model, d_hidden, activation='relu')
        self.linear = Linear(d_hidden, d_model)

    def forward(self, x, padding=None):
        return self.linear(self.feedforward(x))


class TransformerEncoderLayer(nn.Module):

    def __init__(self, dimension, n_heads, hidden, dropout):
        super().__init__()
        self.selfattn = ResidualBlock(
            MultiHead(
                dimension, dimension, n_heads, dropout),
            dimension, dropout)
        self.feedforward = ResidualBlock(
            LinearReLU(dimension, hidden),
            dimension, dropout)

    def forward(self, x, padding=None):
        return self.feedforward(self.selfattn(x, x, x, padding=padding))


# interaction v1, dual interaction
class DualTransformerEncoderLayer(nn.Module):

    def __init__(self, dimension, n_heads, hidden, dropout):
        super().__init__()
        self.context_side_attn = ResidualBlock(
            MultiHead(dimension, dimension, n_heads, dropout),
            dimension, dropout)
        self.context_side_ffn = ResidualBlock(
            LinearReLU(dimension, hidden),
            dimension, dropout)

        self.question_side_attn = ResidualBlock(
            MultiHead(dimension, dimension, n_heads, dropout),
            dimension, dropout)
        self.question_side_ffn = ResidualBlock(
            LinearReLU(dimension, hidden),
            dimension, dropout)

    def forward(self, c, q, padding_c=None, padding_q=None):
        question_encoded = self.context_side_ffn(self.context_side_attn(q, c, c, padding=padding_c))
        context_encoded = self.question_side_ffn(self.question_side_attn(c, q, q, padding=padding_q))
        return context_encoded, question_encoded


class SumDualTransformerEncoderLayer(nn.Module):

    def __init__(self, dimension, n_heads, hidden, dropout):
        super().__init__()
        self.context_side_attn = SumResidualBlock(
            SumMultiHead(dimension, dimension, n_heads, dropout),
            dimension, dropout)
        self.context_side_ffn = ResidualBlock(
            LinearReLU(dimension, hidden),
            dimension, dropout)

        self.question_side_attn = SumResidualBlock(
            SumMultiHead(dimension, dimension, n_heads, dropout),
            dimension, dropout)
        self.question_side_ffn = ResidualBlock(
            LinearReLU(dimension, hidden),
            dimension, dropout)

    def forward(self, c, q, padding_c=None, padding_q=None):
        context_encoded = self.context_side_ffn(self.context_side_attn(q, c, c, padding=padding_c))
        question_encoded = self.question_side_ffn(self.question_side_attn(c, q, q, padding=padding_q))
        return context_encoded, question_encoded


class EqualCQBlockV2(nn.Module):
    def __init__(self, dimension, n_heads, hidden, dropout, bidirectional_lstm=True, num_layers_lstm=1,
                 dropout_lstm=0.0, batch_first_lstm=True):
        super().__init__()
        self.dual_transformer_encoder = DualTransformerEncoderLayer(dimension, n_heads, hidden, dropout)
        self.compression_lstm_c = PackedLSTM(2 * dimension, dimension, bidirectional=bidirectional_lstm,
                                             num_layers=num_layers_lstm, dropout=dropout_lstm,
                                             batch_first=batch_first_lstm)
        self.self_attn_q = TransformerEncoderLayer(dimension, n_heads, hidden, dropout)

    def forward(self, c, q, padding_c=None, padding_q=None, lengths_c=None, lengths_q=None):
        dual_attended_c, dual_attended_q = self.dual_transformer_encoder(c, q, padding_c, padding_q)
        summary_c = torch.cat([dual_attended_c, c], -1)
        condensed_c, (hidden_c, cell_c) = self.compression_lstm_c(summary_c, lengths_c)
        state_c = [self.reshape_rnn_state(x) for x in (hidden_c, cell_c)]

        self_attended_q = self.self_attn_q(dual_attended_q, padding_q)

        return condensed_c, self_attended_q, state_c, dual_attended_c, dual_attended_q

    def reshape_rnn_state(self, h):
        return h.view(h.size(0) // 2, 2, h.size(1), h.size(2)) \
            .transpose(1, 2).contiguous() \
            .view(h.size(0) // 2, h.size(1), h.size(2) * 2).contiguous()


class SumEqualCQBlockV1(nn.Module):
    def __init__(self, dimension, n_heads, hidden, dropout, bidirectional_lstm=True, num_layers_lstm=1,
                 dropout_lstm=0.0, batch_first_lstm=True):
        super().__init__()
        self.dual_transformer_encoder = SumDualTransformerEncoderLayer(dimension, n_heads, hidden, dropout)
        self.compress_dim = 2
        self.compression_lstm_c = PackedLSTM(self.compress_dim * dimension, dimension, bidirectional=bidirectional_lstm,
                                             num_layers=num_layers_lstm, dropout=dropout_lstm,
                                             batch_first=batch_first_lstm)
        self.compression_lstm_q = PackedLSTM(self.compress_dim * dimension, dimension, bidirectional=bidirectional_lstm,
                                             num_layers=num_layers_lstm, dropout=dropout_lstm,
                                             batch_first=batch_first_lstm)

    def forward(self, c, q, padding_c=None, padding_q=None, lengths_c=None, lengths_q=None):
        dual_attended_c, dual_attended_q = self.dual_transformer_encoder(c, q, padding_c, padding_q)
        summary_c = torch.cat([dual_attended_c, c], -1)
        summary_q = torch.cat([dual_attended_q, q], -1)
        condensed_c, (hidden_c, cell_c) = self.compression_lstm_c(summary_c, lengths_c)
        condensed_q, (hidden_q, cell_q) = self.compression_lstm_q(summary_q, lengths_q)

        return condensed_c, condensed_q, (hidden_c, cell_c), dual_attended_c, dual_attended_q


class SumEqualCQBlockV3(nn.Module):
    def __init__(self, dimension, n_heads, hidden, dropout, bidirectional_lstm=True, num_layers_lstm=1,
                 dropout_lstm=0.0, batch_first_lstm=True):
        super().__init__()
        self.dual_transformer_encoder = SumDualTransformerEncoderLayer(dimension, n_heads, hidden, dropout)

        self.compression_lstm_c = PackedLSTM(dimension, dimension, bidirectional=bidirectional_lstm,
                                             num_layers=num_layers_lstm, dropout=dropout_lstm,
                                             batch_first=batch_first_lstm)
        self.compression_lstm_q = PackedLSTM(dimension, dimension, bidirectional=bidirectional_lstm,
                                             num_layers=num_layers_lstm, dropout=dropout_lstm,
                                             batch_first=batch_first_lstm)

    def forward(self, c, q, padding_c=None, padding_q=None, lengths_c=None, lengths_q=None):
        dual_attended_c, dual_attended_q = self.dual_transformer_encoder(c, q, padding_c, padding_q)

        condensed_c, (hidden_c, cell_c) = self.compression_lstm_c(dual_attended_c, lengths_c)
        condensed_q, (hidden_q, cell_q) = self.compression_lstm_q(q, lengths_q)

        return condensed_c, condensed_q, (hidden_c, cell_c), dual_attended_c, dual_attended_q


class MyCoattentiveLayer(nn.Module):

    def __init__(self, d, dropout=0.2):
        super().__init__()
        self.proj = Feedforward(d, d, dropout=0.0)
        self.embed_sentinel = nn.Embedding(2, d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, context, question, context_padding, question_padding):
        context_padding = torch.cat(
            [context.new_zeros((context.size(0), 1), dtype=torch.long) == 1, context_padding], 1)
        question_padding = torch.cat(
            [question.new_zeros((question.size(0), 1), dtype=torch.long) == 1, question_padding], 1)

        # TODO why the sentinel necessary?
        context_sentinel = self.embed_sentinel(context.new_zeros((context.size(0), 1), dtype=torch.long))
        context = torch.cat(
            [context_sentinel, self.dropout(context)], 1)  # batch_size x (context_length + 1) x features

        # TODO why proj and tanh necessary?
        question_sentinel = self.embed_sentinel(question.new_ones((question.size(0), 1), dtype=torch.long))
        question = torch.cat([question_sentinel, question], 1)  # batch_size x (question_length + 1) x features
        question = torch.tanh(self.proj(question))  # batch_size x (question_length + 1) x features

        affinity = context.bmm(question.transpose(1, 2))  # batch_size x (context_length + 1) x (question_length + 1)
        # Fix the bug. Origin comment: batch_size x (context_length + 1) x 1
        attn_over_context = self.normalize(affinity,
                                           context_padding)  # batch_size x (context_length + 1) x (question_length + 1)
        attn_over_question = self.normalize(affinity.transpose(1, 2),
                                            question_padding)  # batch_size x (question_length + 1) x (context_length + 1)
        sum_of_context = self.attn(attn_over_context, context)  # batch_size x (question_length + 1) x features
        sum_of_question = self.attn(attn_over_question, question)  # batch_size x (context_length + 1) x features
        coattn_context = self.attn(attn_over_question, sum_of_context)  # batch_size x (context_length + 1) x features
        coattn_question = self.attn(attn_over_context, sum_of_question)  # batch_size x (question_length + 1) x features
        return coattn_context[:, 1:], coattn_question[:, 1:]

    @staticmethod
    def attn(weights, candidates):
        # TODO why not use normal form batch matrix multiplication
        w1, w2, w3 = weights.size()
        c1, c2, c3 = candidates.size()
        return weights.unsqueeze(3).expand(w1, w2, w3, c3).mul(candidates.unsqueeze(2).expand(c1, c2, w3, c3)).sum(
            1).squeeze(1)

    @staticmethod
    def normalize(original, padding):
        raw_scores = original.clone()
        raw_scores.masked_fill_(padding.unsqueeze(-1).expand_as(raw_scores), -INF)
        return F.softmax(raw_scores, dim=1)


# interaction v2, two transformer layer + co-att layer
# TODO, resource overload, maybe co-att on top of transformer
class CoattentiveTransformerEncoderLayer(nn.Module):

    def __init__(self, dimension, n_heads, hidden, dropout):
        super().__init__()
        self.context_side_attn = ResidualBlock(
            MultiHead(dimension, dimension, n_heads, dropout),
            dimension, dropout)
        self.context_side_ffn = ResidualBlock(
            LinearReLU(dimension, hidden),
            dimension, dropout)

        self.question_side_attn = ResidualBlock(
            MultiHead(dimension, dimension, n_heads, dropout),
            dimension, dropout)
        self.question_side_ffn = ResidualBlock(
            LinearReLU(dimension, hidden),
            dimension, dropout)
        self.coatt_layer = MyCoattentiveLayer(d=dimension, dropout=dropout)

    def forward(self, c, q, padding_c=None, padding_q=None):
        # print('c', c.size())
        # print('q', q.size())
        context_encoded = self.context_side_ffn(self.context_side_attn(c, c, c, padding=padding_c))
        question_encoded = self.question_side_ffn(self.question_side_attn(q, q, q, padding=padding_q))
        # print('context_encoded', context_encoded.size())
        # print('question_encoded', question_encoded.size())
        context_encoded, question_encoded = self.coatt_layer(context_encoded, question_encoded,
                                                             padding_c, padding_q)
        # print('context_encoded2', context_encoded.size())
        # print('question_encoded2', question_encoded.size())
        return context_encoded, question_encoded


# TODO interaction v3, interaction v1/v2 + (Max-pooling [MP] and CNN) layer
# for now, implement DualTransformerEncodeLayer + MP, CNN
# but after MP, the length will change, how to pad ?
# CNN used on context
# may be a soft masking on label? how
class FilteredDualTransformerEncoderLayer(nn.Module):

    def __init__(self):
        super(FilteredDualTransformerEncoderLayer, self).__init__()
        pass

    def forward(self, x):
        pass


class TransformerEncoder(nn.Module):

    def __init__(self, dimension, n_heads, hidden, num_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(dimension, n_heads, hidden, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding=None):
        x = self.dropout(x)
        encoding = [x]
        for layer in self.layers:
            x = layer(x, padding=padding)
            encoding.append(x)
        return encoding


class DualTransformerEncoder(nn.Module):
    def __init__(self, dimension, n_heads, hidden, num_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            [DualTransformerEncoderLayer(dimension, n_heads, hidden, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.num_layer = num_layers

    def forward(self, c, q, padding_c=None, padding_q=None):
        if self.num_layer > 1:
            c = self.dropout(c)
            q = self.dropout(q)
        encoding = [[c, q]]
        for layer in self.layers:
            c, q = layer(c, q, padding_c, padding_q)
            encoding.append([c, q])
        return encoding


class SumDualTransformerEncoder(nn.Module):
    def __init__(self, dimension, n_heads, hidden, num_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            [SumDualTransformerEncoderLayer(dimension, n_heads, hidden, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.num_layer = num_layers

    def forward(self, c, q, padding_c=None, padding_q=None):
        if self.num_layer > 1:
            c = self.dropout(c)
            q = self.dropout(q)
        encoding = [[c, q]]
        for layer in self.layers:
            c, q = layer(c, q, padding_c, padding_q)
            encoding.append([c, q])
        return encoding


class CoattentiveTransformerEncoder(nn.Module):
    def __init__(self, dimension, n_heads, hidden, num_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            [CoattentiveTransformerEncoderLayer(dimension, n_heads, hidden, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, c, q, padding_c=None, padding_q=None):
        c = self.dropout(c)
        q = self.dropout(q)
        encoding = [[c, q]]
        for layer in self.layers:
            c, q = layer(c, q, padding_c, padding_q)
            encoding.append([c, q])
        return encoding


class RefineMultiHead(nn.Module):
    def __init__(self, d_key, d_value, n_heads, dropout_ratio, causal=False):
        super().__init__()
        self.attention = Attention(d_key, dropout_ratio, causal=causal)
        self.wq = Linear(d_key, d_key, bias=False)
        self.wk = Linear(d_key, d_key, bias=False)
        self.wv = Linear(d_value, d_value, bias=False)
        self.n_heads = n_heads

    def forward(self, query, key, value, padding=None):
        query, key, value = self.wq(query), self.wk(key), self.wv(value)
        query, key, value = (
            x.chunk(self.n_heads, -1) for x in (query, key, value))
        # print("-----size of multiattn return-----\n", torch.cat([self.attention(q, k, v, padding=padding)
        #                   for q, k, v in zip(query, key, value)], -1).size())
        return torch.cat([self.attention(q, k, v, padding=padding)
                          for q, k, v in zip(query, key, value)], -1)


class RefinementLayer(nn.Module):

    def __init__(self, dimension, n_heads, hidden, dropout):
        super().__init__()
        self.selfattn = ResidualBlock(
            RefineMultiHead(
                dimension, dimension, n_heads, dropout),
            dimension, dropout)
        self.feedforward = ResidualBlock(
            LinearReLU(dimension, hidden),
            dimension, dropout)

    def forward(self, q, k, padding=None):
        return self.feedforward(self.selfattn(q, k, k, padding=padding))


class Refinement(nn.Module):

    def __init__(self, dimension, n_heads, hidden, num_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            [RefinementLayer(dimension, n_heads, hidden, dropout) for i in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, padding=None):
        k = self.dropout(k)
        encoding = [k]
        for layer in self.layers:
            x = layer(q, k,
                      padding=padding)
            encoding.append(x)
        return encoding


# The supervision of answer comes from this layer
# TODO why we only use self_attended_context
# TODO why not use self_attended_answer too?
# TODO In our experiment setting, we can input
# a, c and a, q (q may not good)
class TransformerDecoderLayer(nn.Module):

    def __init__(self, dimension, n_heads, hidden, dropout, causal=True):
        super().__init__()
        self.selfattn = ResidualBlock(
            MultiHead(dimension, dimension, n_heads,
                      dropout, causal),
            dimension, dropout)
        self.attention = ResidualBlock(
            MultiHead(dimension, dimension, n_heads,
                      dropout),
            dimension, dropout)
        self.feedforward = ResidualBlock(
            LinearReLU(dimension, hidden),
            dimension, dropout)

    def forward(self, x, encoding, context_padding=None, answer_padding=None):
        x = self.selfattn(x, x, x, padding=answer_padding)
        return self.feedforward(self.attention(x, encoding, encoding, padding=context_padding))


class TransformerDecoder(nn.Module):

    def __init__(self, dimension, n_heads, hidden, num_layers, dropout, causal=True):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(dimension, n_heads, hidden, dropout, causal=causal) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.d_model = dimension

    def forward(self, x, encoding, context_padding=None, positional_encodings=True, answer_padding=None):
        if positional_encodings:
            x = x + positional_encodings_like(x)
        x = self.dropout(x)
        for layer, enc in zip(self.layers, encoding[1:]):
            x = layer(x, enc, context_padding=context_padding, answer_padding=answer_padding)
        return x


def mask(targets, out, pad_idx=1):
    m = (targets != pad_idx)
    out_mask = m.unsqueeze(-1).expand_as(out).contiguous()
    out_after = out[out_mask].contiguous().view(-1, out.size(-1))
    targets_after = targets[m]
    return out_after, targets_after


class Highway(torch.nn.Module):
    def __init__(self, d_in, activation='relu', n_layers=1):
        super(Highway, self).__init__()
        self.d_in = d_in
        self._layers = torch.nn.ModuleList([Linear(d_in, 2 * d_in) for _ in range(n_layers)])
        for layer in self._layers:
            layer.bias[d_in:].fill_(1)
        self.activation = getattr(F, activation)

    def forward(self, inputs):
        current_input = inputs
        for layer in self._layers:
            projected_input = layer(current_input)
            linear_part = current_input
            nonlinear_part = projected_input[:, :self.d_in] if projected_input.dim() == 2 else projected_input[:, :,
                                                                                               :self.d_in]
            nonlinear_part = self.activation(nonlinear_part)
            gate = projected_input[:, self.d_in:(2 * self.d_in)] if projected_input.dim() == 2 else projected_input[:,
                                                                                                    :, self.d_in:(
                    2 * self.d_in)]
            gate = F.sigmoid(gate)
            current_input = gate * linear_part + (1 - gate) * nonlinear_part
        return current_input


class LinearFeedforward(nn.Module):

    def __init__(self, d_in, d_hid, d_out, activation='relu'):
        super().__init__()
        self.feedforward = Feedforward(d_in, d_hid, activation=activation)
        self.linear = Linear(d_hid, d_out)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        return self.dropout(self.linear(self.feedforward(x)))


class PackedLSTM(nn.Module):

    def __init__(self, d_in, d_out, bidirectional=False, num_layers=1,
                 dropout=0.0, batch_first=True):
        """A wrapper class that packs input sequences and unpacks output sequences"""
        super().__init__()
        if bidirectional:
            d_out = d_out // 2
        self.rnn = nn.LSTM(d_in, d_out,
                           num_layers=num_layers,
                           dropout=dropout,
                           bidirectional=bidirectional,
                           batch_first=batch_first)
        self.batch_first = batch_first

    def forward(self, inputs, lengths, hidden=None):
        lens, indices = torch.sort(inputs.new_tensor(lengths, dtype=torch.long), 0, True)
        inputs = inputs[indices] if self.batch_first else inputs[:, indices]
        outputs, (h, c) = self.rnn(pack(inputs, lens.tolist(),
                                        batch_first=self.batch_first), hidden)
        outputs = unpack(outputs, batch_first=self.batch_first)[0]
        _, _indices = torch.sort(indices, 0)
        outputs = outputs[_indices] if self.batch_first else outputs[:, _indices]
        h, c = h[:, _indices, :], c[:, _indices, :]
        return outputs, (h, c)


class Linear(nn.Linear):

    def forward(self, x):
        size = x.size()
        return super().forward(
            x.contiguous().view(-1, size[-1])).view(*size[:-1], -1)


class Feedforward(nn.Module):

    def __init__(self, d_in, d_out, activation=None, bias=True, dropout=0.2):
        super().__init__()
        if activation is not None:
            self.activation = getattr(torch, activation)
        else:
            self.activation = lambda x: x
        self.linear = Linear(d_in, d_out, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.activation(self.linear(self.dropout(x)))


class Embedding(nn.Module):

    def __init__(self, field, trained_dimension, dropout=0.0, project=True, requires_grad=False):
        super().__init__()
        self.field = field
        self.project = project
        dimension = 0
        pretrained_dimension = field.vocab.vectors.size(-1)
        self.pretrained_embeddings = [nn.Embedding(len(field.vocab), pretrained_dimension)]
        self.pretrained_embeddings[0].weight.data = field.vocab.vectors
        self.pretrained_embeddings[0].weight.requires_grad = requires_grad
        dimension += pretrained_dimension
        if self.project:
            self.projection = Feedforward(dimension, trained_dimension)
        dimension = trained_dimension
        self.dropout = nn.Dropout(dropout)
        self.dimension = dimension

    def forward(self, x, lengths=None, device=-1):
        pretrained_embeddings = self.pretrained_embeddings[0](x.cpu()).to(x.device).detach()
        return self.projection(pretrained_embeddings) if self.project else pretrained_embeddings

    # def set_embeddings(self, w):
    #     self.pretrained_embeddings[0].weight.data = w
    #     self.pretrained_embeddings[0].weight.requires_grad = False


class SemanticFusionUnit(nn.Module):

    def __init__(self, d, l):
        super().__init__()
        self.r_hat = Feedforward(d * l, d, 'tanh')
        self.g = Feedforward(d * l, d, 'sigmoid')
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        c = self.dropout(torch.cat(x, -1))
        r_hat = self.r_hat(c)
        g = self.g(c)
        o = g * r_hat + (1 - g) * x[0]
        return o


class LSTMDecoderAttention(nn.Module):
    def __init__(self, dim, dot=False):
        super().__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.linear_out = nn.Linear(2 * dim, dim, bias=False)
        self.tanh = nn.Tanh()
        self.mask = None
        self.dot = dot

    def applyMasks(self, context_mask):
        self.context_mask = context_mask

    def forward(self, input, context):
        if not self.dot:
            targetT = self.linear_in(input).unsqueeze(2)  # batch x dim x 1
        else:
            targetT = input.unsqueeze(2)

        context_scores = torch.bmm(context, targetT).squeeze(2)
        context_scores.masked_fill_(self.context_mask, -float('inf'))
        context_attention = F.softmax(context_scores, dim=-1) + EPSILON
        # context_attention = F.softmax(context_scores, dim=-1).clamp(min=EPSILON)
        context_alignment = torch.bmm(context_attention.unsqueeze(1), context).squeeze(1)

        combined_representation = torch.cat([input, context_alignment], 1)
        output = self.tanh(self.linear_out(combined_representation))

        return output, context_attention, context_alignment


class CoattentiveLayer(nn.Module):

    def __init__(self, d, dropout=0.2):
        super().__init__()
        # print(10 * '-', 'd', d)
        self.proj = Feedforward(d, d, dropout=0.0)
        self.embed_sentinel = nn.Embedding(2, d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, context, question, context_padding, question_padding):
        context_padding = torch.cat(
            [context.new_zeros((context.size(0), 1), dtype=torch.long) == 1, context_padding], 1)
        question_padding = torch.cat(
            [question.new_zeros((question.size(0), 1), dtype=torch.long) == 1, question_padding], 1)

        # TODO why the sentinel necessary?
        context_sentinel = self.embed_sentinel(context.new_zeros((context.size(0), 1), dtype=torch.long))
        context = torch.cat(
            [context_sentinel, self.dropout(context)], 1)  # batch_size x (context_length + 1) x features

        # TODO why proj and tanh necessary?
        question_sentinel = self.embed_sentinel(question.new_ones((question.size(0), 1), dtype=torch.long))
        question = torch.cat([question_sentinel, question], 1)  # batch_size x (question_length + 1) x features
        question = torch.tanh(self.proj(question))  # batch_size x (question_length + 1) x features

        affinity = context.bmm(question.transpose(1, 2))  # batch_size x (context_length + 1) x (question_length + 1)
        # Fix the bug. Origin comment: batch_size x (context_length + 1) x 1
        attn_over_context = self.normalize(affinity,
                                           context_padding)  # batch_size x (context_length + 1) x (question_length + 1)
        attn_over_question = self.normalize(affinity.transpose(1, 2),
                                            question_padding)  # batch_size x (question_length + 1) x (context_length + 1)
        sum_of_context = self.attn(attn_over_context, context)  # batch_size x (question_length + 1) x features
        sum_of_question = self.attn(attn_over_question, question)  # batch_size x (context_length + 1) x features
        coattn_context = self.attn(attn_over_question, sum_of_context)  # batch_size x (context_length + 1) x features
        coattn_question = self.attn(attn_over_context, sum_of_question)  # batch_size x (question_length + 1) x features
        return (torch.cat([coattn_context, sum_of_question], 2)[:, 1:],
                torch.cat([coattn_question, sum_of_context], 2)[:, 1:])

    @staticmethod
    def attn(weights, candidates):
        # TODO why not use normal form batch matrix multiplication
        w1, w2, w3 = weights.size()
        c1, c2, c3 = candidates.size()
        return weights.unsqueeze(3).expand(w1, w2, w3, c3).mul(candidates.unsqueeze(2).expand(c1, c2, w3, c3)).sum(
            1).squeeze(1)

    @staticmethod
    def normalize(original, padding):
        raw_scores = original.clone()
        raw_scores.masked_fill_(padding.unsqueeze(-1).expand_as(raw_scores), -INF)
        return F.softmax(raw_scores, dim=1)


class MutualAttentiveLayer(nn.Module):

    def __init__(self, d, dropout=0.2):
        super().__init__()
        self.proj_question = Feedforward(d, d, dropout=0.0)
        self.proj_context = Feedforward(d, d, dropout=0.0)
        self.embed_sentinel = nn.Embedding(2, d)
        self.dropout = nn.Dropout(dropout)
        # self.t_context = math.sqrt(d)
        self.t_context = 5
        self.t_question = 1

    def forward(self, context, question, context_padding, question_padding):
        context_padding = torch.cat(
            [context.new_zeros((context.size(0), 1), dtype=torch.long) == 1, context_padding], 1)
        question_padding = torch.cat(
            [question.new_zeros((question.size(0), 1), dtype=torch.long) == 1, question_padding], 1)

        context_sentinel = self.embed_sentinel(context.new_zeros((context.size(0), 1), dtype=torch.long))
        context = torch.cat([context_sentinel, self.dropout(context)],
                            1)  # batch_size x (context_length + 1) x features
        weighted_context = torch.tanh(self.proj_context(context))  # batch_size x (context_length + 1) x features

        question_sentinel = self.embed_sentinel(question.new_ones((question.size(0), 1), dtype=torch.long))
        question = torch.cat([question_sentinel, question], 1)  # batch_size x (question_length + 1) x features
        weighted_question = torch.tanh(self.proj_question(question))  # batch_size x (question_length + 1) x features

        # weighted_context = F.normalize(weighted_context, dim=-1)
        # weighted_question = F.normalize(weighted_question, dim=-1)

        affinity = weighted_context.bmm(
            weighted_question.transpose(1, 2))  # batch_size x (context_length + 1) x (question_length + 1)

        attn_over_context = self.normalize(affinity,
                                           context_padding)  # batch_size x (context_length + 1) x (question_length + 1)
        attn_over_question = self.normalize(affinity.transpose(1, 2),
                                            question_padding)  # batch_size x (question_length + 1) x (context_length + 1)

        # context_score: (b, (context_length + 1), 1)
        context_score = F.softmax(torch.sum(attn_over_context, dim=-1) / self.t_context, dim=1).unsqueeze(-1)
        # question_score: (b, (question_length + 1), 1)
        question_score = F.softmax(torch.sum(attn_over_question, dim=-1) / self.t_question, dim=1).unsqueeze(-1)
        # scaled_context: (b, (context_length + 1), h)
        # scaled_context = context_score * context
        scaled_context = context_score * weighted_context
        # scaled_question: (b, (question_length + 1), h)
        scaled_question = question_score * weighted_question
        # scaled_question = question_score * question

        sum_of_context = self.attn(attn_over_context, context)  # batch_size x (question_length + 1) x features
        sum_of_question = self.attn(attn_over_question, question)  # batch_size x (context_length + 1) x features
        # coattn_context = self.attn(attn_over_question, sum_of_context) # batch_size x (context_length + 1) x features
        # coattn_question = self.attn(attn_over_context, sum_of_question) # batch_size x (question_length + 1) x features

        return torch.cat([scaled_context, sum_of_question], 2)[:, 1:], torch.cat([scaled_question, sum_of_context], 2)[
                                                                       :, 1:]
        # return torch.cat([coattn_context, sum_of_question], 2)[:, 1:], torch.cat([coattn_question, sum_of_context], 2)[:, 1:]
        # return coattn_context[:, 1:], coattn_question[:, 1:]

    @staticmethod
    def attn(weights, candidates):
        w1, w2, w3 = weights.size()
        c1, c2, c3 = candidates.size()
        return weights.unsqueeze(3).expand(w1, w2, w3, c3).mul(candidates.unsqueeze(2).expand(c1, c2, w3, c3)).sum(
            1).squeeze(1)

    @staticmethod
    def normalize(original, padding):
        raw_scores = original.clone()
        raw_scores.masked_fill_(padding.unsqueeze(-1).expand_as(raw_scores), -INF)
        # return F.softmax(raw_scores, dim=1)
        return raw_scores


class MutualAttentiveSubLayer(nn.Module):

    def __init__(self, d, dropout=0.2):
        super().__init__()
        self.proj_question = Feedforward(d, d, dropout=0.0)
        self.proj_context = Feedforward(d, d, dropout=0.0)
        self.embed_sentinel = nn.Embedding(2, d)
        self.dropout = nn.Dropout(dropout)
        # self.t_context = math.sqrt(d)
        self.t_context = 5
        self.t_question = 1

    def forward(self, context, question, context_padding, question_padding):
        context_padding = torch.cat([context.new_zeros((context.size(0), 1), dtype=torch.long) == 1, context_padding],
                                    1)
        question_padding = torch.cat(
            [question.new_zeros((question.size(0), 1), dtype=torch.long) == 1, question_padding], 1)

        context_sentinel = self.embed_sentinel(context.new_zeros((context.size(0), 1), dtype=torch.long))
        context = torch.cat([context_sentinel, self.dropout(context)],
                            1)  # batch_size x (context_length + 1) x features
        weighted_context = torch.tanh(self.proj_context(context))  # batch_size x (context_length + 1) x features

        question_sentinel = self.embed_sentinel(question.new_ones((question.size(0), 1), dtype=torch.long))
        question = torch.cat([question_sentinel, question], 1)  # batch_size x (question_length + 1) x features
        weighted_question = torch.tanh(self.proj_question(question))  # batch_size x (question_length + 1) x features

        # weighted_context = F.normalize(weighted_context, dim=-1)
        # weighted_question = F.normalize(weighted_question, dim=-1)

        affinity = weighted_context.bmm(
            weighted_question.transpose(1, 2))  # batch_size x (context_length + 1) x (question_length + 1)

        attn_over_context = self.normalize(affinity,
                                           context_padding)  # batch_size x (context_length + 1) x (question_length + 1)
        attn_over_question = self.normalize(affinity.transpose(1, 2),
                                            question_padding)  # batch_size x (question_length + 1) x (context_length + 1)

        # context_score: (b, (context_length + 1), 1)
        context_score = F.softmax(torch.sum(attn_over_context, dim=-1) / self.t_context, dim=1).unsqueeze(-1) + EPSILON
        # question_score: (b, (question_length + 1), 1)
        question_score = F.softmax(torch.sum(attn_over_question, dim=-1) / self.t_question, dim=1).unsqueeze(
            -1) + EPSILON
        # scaled_context: (b, (context_length + 1), h)
        # scaled_context = context_score * context
        scaled_context = context_score * weighted_context
        # scaled_question: (b, (question_length + 1), h)
        scaled_question = question_score * weighted_question
        # scaled_question = question_score * question

        # sum_of_context = self.attn(attn_over_context, context) # batch_size x (question_length + 1) x features
        sum_of_context = self.attn(attn_over_context, context)  # batch_size x (question_length + 1) x features
        # sum_of_question = self.attn(attn_over_question, question) # batch_size x (context_length + 1) x features
        sum_of_question = self.attn(attn_over_question, question)  # batch_size x (context_length + 1) x features
        # coattn_context = self.attn(attn_over_question, sum_of_context) # batch_size x (context_length + 1) x features
        # coattn_question = self.attn(attn_over_context, sum_of_question) # batch_size x (question_length + 1) x features

        # return torch.cat([scaled_context, sum_of_question], 2)[:, 1:], torch.cat([scaled_question, sum_of_context], 2)[
        #                                                                :, 1:]
        # return torch.cat([coattn_context, sum_of_question], 2)[:, 1:], torch.cat([coattn_question, sum_of_context], 2)[:, 1:]
        return context[:, 1:], question[:, 1:], scaled_context[:, 1:], scaled_question[:, 1:], \
               sum_of_question[:, 1:], sum_of_context[:, 1:]

    @staticmethod
    def attn(weights, candidates):
        w1, w2, w3 = weights.size()
        c1, c2, c3 = candidates.size()
        return weights.unsqueeze(3).expand(w1, w2, w3, c3).mul(candidates.unsqueeze(2).expand(c1, c2, w3, c3)).sum(
            1).squeeze(1)

    @staticmethod
    def normalize(original, padding):
        raw_scores = original.clone()
        raw_scores.masked_fill_(padding.unsqueeze(-1).expand_as(raw_scores), -INF)
        # return F.softmax(raw_scores, dim=1)
        return raw_scores


class MutualAttentiveBlock(nn.Module):

    def __init__(self, dimension, dropout):
        super().__init__()
        self.mutualattn = MutualAttentiveSubLayer(dimension, dropout)
        # self.linear_compress_context = Linear(2 * dimension, dimension)
        # self.linear_compress_question = Linear(2 * dimension, dimension)
        # self.linear_compress_context2 = Linear(dimension, dimension)
        # self.linear_compress_question2 = Linear(dimension, dimension)
        # self.feedforward = ResidualBlock(
        #     LinearReLU(dimension, dimension),
        #     dimension, dropout)
        # self.layernorm_context = LayerNorm(dimension)
        # self.layernorm_question = LayerNorm(dimension)
        # self.condense_context = PackedLSTM(2 * dimension, dimension,
        #                                    batch_first=True, bidirectional=True, num_layers=1)
        # self.condense_question = PackedLSTM(2 * dimension, dimension,
        #                                     batch_first=True, bidirectional=True, num_layers=1)

    def forward(self, context, question, context_padding, question_padding, context_lengths, question_lengths):
        # weighted_context, weighted_question, attn_context, attn_question = self.mutualattn(context, question,
        #                                                                                    context_padding,
        #                                                                                    question_padding)
        weighted_context, weighted_question, attn_context, attn_question, sum_of_question, sum_of_context = self.mutualattn(
            context, question,
            context_padding,
            question_padding)
        # cat_context = torch.cat([attn_context, sum_of_question], 2)
        # cat_question = torch.cat([attn_question, sum_of_context], 2)
        # subcompress_context = self.linear_compress_context2(torch.tanh(self.linear_compress_context(cat_context)))
        # subcompress_question = self.linear_compress_question2(torch.tanh(self.linear_compress_question(cat_question)))
        # subcompress_context, _ = self.condense_context(cat_context, context_lengths)
        # subcompress_question, _ = self.condense_context(cat_question, question_lengths)
        # subcompress_question, _ = self.condense_question(cat_question, question_lengths)
        # return subcompress_context, subcompress_question
        # normed_context = self.layernorm_context(weighted_context + attn_context)
        # normed_question = self.layernorm_question(weighted_question + attn_question)
        # return normed_context, normed_question
        # return torch.cat([normed_context, sum_of_question], 2), torch.cat([normed_question, sum_of_context], 2)
        # return cat_context, cat_question
        return attn_context, attn_question, sum_of_question, sum_of_context
