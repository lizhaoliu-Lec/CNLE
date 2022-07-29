import os
import math
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from collections import namedtuple
import sys
from typing import List, Tuple
from util import get_trainable_params

from cove import MTLSTM
from allennlp.modules.elmo import Elmo, batch_to_ids

from .common import positional_encodings_like, INF, EPSILON, TransformerEncoder, TransformerDecoder, PackedLSTM, \
    LSTMDecoderAttention, LSTMDecoder, Embedding, Feedforward, mask, CoattentiveLayer


class MultitaskQuestionAnsweringNetworkV1(nn.Module):

    def __init__(self, field, args):
        super().__init__()
        self.field = field
        self.args = args
        self.pad_idx = self.field.vocab.stoi[self.field.pad_token]

        def dp(args):
            return args.dropout_ratio if args.rnn_layers > 1 else 0.

        if self.args.glove_and_char:

            self.encoder_embeddings = Embedding(field, args.dimension,
                                                dropout=args.dropout_ratio, project=not args.cove)

            if self.args.cove or self.args.intermediate_cove:
                self.cove = MTLSTM(model_cache=args.embeddings, layer0=args.intermediate_cove, layer1=args.cove)
                cove_params = get_trainable_params(self.cove)
                for p in cove_params:
                    p.requires_grad = False
                cove_dim = int(args.intermediate_cove) * 600 + int(
                    args.cove) * 600 + 400  # the last 400 is for GloVe and char n-gram embeddings
                self.project_cove = Feedforward(cove_dim, args.dimension)

        if -1 not in self.args.elmo:
            options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
            weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
            self.elmo = Elmo(options_file, weight_file, 3, dropout=0.0, do_layer_norm=False)
            elmo_params = get_trainable_params(self.elmo)
            for p in elmo_params:
                p.requires_grad = False
            elmo_dim = 1024 * len(self.args.elmo)
            self.project_elmo = Feedforward(elmo_dim, args.dimension)
            if self.args.glove_and_char:
                self.project_embeddings = Feedforward(2 * args.dimension, args.dimension, dropout=0.0)

        # TODO why not use shared embedding
        self.decoder_embeddings = Embedding(field, args.dimension,
                                            dropout=args.dropout_ratio, project=True)

        self.bilstm_before_coattention = PackedLSTM(args.dimension, args.dimension,
                                                    batch_first=True, bidirectional=True, num_layers=1, dropout=0)
        self.coattention = CoattentiveLayer(args.dimension, dropout=0.3)
        dim = 2 * args.dimension + args.dimension + args.dimension

        self.context_bilstm_after_coattention = PackedLSTM(dim, args.dimension,
                                                           batch_first=True, dropout=dp(args), bidirectional=True,
                                                           num_layers=args.rnn_layers)
        self.self_attentive_encoder_context = TransformerEncoder(args.dimension, args.transformer_heads,
                                                                 args.transformer_hidden, args.transformer_layers,
                                                                 args.dropout_ratio)
        self.bilstm_context = PackedLSTM(args.dimension, args.dimension,
                                         batch_first=True, dropout=dp(args), bidirectional=True,
                                         num_layers=args.rnn_layers)

        self.question_bilstm_after_coattention = PackedLSTM(dim, args.dimension,
                                                            batch_first=True, dropout=dp(args), bidirectional=True,
                                                            num_layers=args.rnn_layers)
        self.self_attentive_encoder_question = TransformerEncoder(args.dimension, args.transformer_heads,
                                                                  args.transformer_hidden, args.transformer_layers,
                                                                  args.dropout_ratio)
        self.bilstm_question = PackedLSTM(args.dimension, args.dimension,
                                          batch_first=True, dropout=dp(args), bidirectional=True,
                                          num_layers=args.rnn_layers)

        self.self_attentive_decoder = TransformerDecoder(args.dimension, args.transformer_heads,
                                                         args.transformer_hidden, args.transformer_layers,
                                                         args.dropout_ratio)
        self.dual_ptr_rnn_decoder = DualPtrRNNDecoder(args.dimension, args.dimension,
                                                      dropout=args.dropout_ratio, num_layers=args.rnn_layers)

        self.generative_vocab_size = min(len(field.vocab), args.max_generative_vocab)
        self.out = nn.Linear(args.dimension, self.generative_vocab_size)

        self.decoder = nn.LSTMCell(input_size=2 * args.dimension,
                                   hidden_size=args.dimension,
                                   bias=True)

        self.dropout = nn.Dropout(0.4)

    def set_embeddings(self, embeddings):
        self.encoder_embeddings.set_embeddings(embeddings)
        self.decoder_embeddings.set_embeddings(embeddings)

    def step(self,
             Ybar_t: torch.Tensor,
             dec_state: Tuple[torch.Tensor, torch.Tensor],
             enc_hiddens: torch.Tensor,
             enc_hiddens_proj: torch.Tensor,
             enc_masks: torch.Tensor) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:
        """ Compute one forward step of the LSTM decoder, including the attention computation.

        @param Ybar_t: (Tensor), Concatenated Tensor of [Y_t o_prev], with shape (b, e + h). The input for the decoder,
                                where b = batch size, e = embedding size, h = hidden size.
        @param dec_state: (tuple(Tensor, Tensor)), Tuple of tensors both with shape (b, h), where b = batch size, h = hidden size.
                First tensor is decoder's prev hidden state, second tensor is decoder's prev cell.
        @param enc_hiddens: (Tensor), Encoder hidden states Tensor, with shape (b, src_len, h * 2), where b = batch size,
                                    src_len = maximum source length, h = hidden size.
        @param enc_hiddens_proj: (Tensor), Encoder hidden states Tensor, projected from (h * 2) to h. Tensor is with shape (b, src_len, h),
                                    where b = batch size, src_len = maximum source length, h = hidden size.
        @param enc_masks: (Tensor), Tensor of sentence masks shape (b, src_len),
                                    where b = batch size, src_len is maximum source length.

        @returns dec_state (tuple (Tensor, Tensor)): Tuple of tensors both shape (b, h), where b = batch size, h = hidden size.
                First tensor is decoder's new hidden state, second tensor is decoder's new cell.
        @returns combined_output (Tensor): Combined output Tensor at timestep t, shape (b, h), where b = batch size, h = hidden size.
        @returns e_t (Tensor): Tensor of shape (b, src_len). It is attention scores distribution.
                                Note: You will not use this outside of this function.
                                      We are simply returning this value so that we can sanity check
                                      your implementation.
        """

        combined_output = None

        # YOUR CODE HERE (~3 Lines)
        # TODO:
        # 1. Apply the decoder to `Ybar_t` and `dec_state`to obtain the new dec_state.
        # 2. Split dec_state into its two parts (dec_hidden, dec_cell)
        # 3. Compute the attention scores e_t, a Tensor shape (b, src_len).
        # Note: b = batch_size, src_len = maximum source length, h = hidden size.
        ###
        # Hints:
        # - dec_hidden is shape (b, h) and corresponds to h^dec_t in the PDF (batched)
        # - enc_hiddens_proj is shape (b, src_len, h) and corresponds to W_{attProj} h^enc (batched).
        # - Use batched matrix multiplication (torch.bmm) to compute e_t.
        # - To get the tensors into the right shapes for bmm, you will need to do some squeezing and unsqueezing.
        # - When using the squeeze() function make sure to specify the dimension you want to squeeze
        # over. Otherwise, you will remove the batch dimension accidentally, if batch_size = 1.
        ###
        # Use the following docs to implement this functionality:
        # Batch Multiplication:
        # https://pytorch.org/docs/stable/torch.html#torch.bmm
        # Tensor Unsqueeze:
        # https://pytorch.org/docs/stable/torch.html#torch.unsqueeze
        # Tensor Squeeze:
        # https://pytorch.org/docs/stable/torch.html#torch.squeeze

        # Ybar_t: (b, e + h), dec_state: ((b, h), (b, h))
        # dec_state: ((b, h), (b, h))
        dec_state = self.decoder(Ybar_t, dec_state)
        (dec_hidden, dec_cell) = dec_state

        # enc_hiddens_proj: (b, src_len, h), dec_hidden: (b, h) -> (b, h, 1)
        # e_t: (b, src_len, 1) -> (b, src_len)
        e_t = torch.squeeze(
            torch.bmm(enc_hiddens_proj, torch.unsqueeze(dec_hidden, dim=2)), dim=2)

        # END YOUR CODE

        # Set e_t to -inf where enc_masks has 1
        if enc_masks is not None:
            e_t.data.masked_fill_(enc_masks.bool(), -float('inf'))

        # YOUR CODE HERE (~6 Lines)
        # TODO:
        # 1. Apply softmax to e_t to yield alpha_t
        # 2. Use batched matrix multiplication between alpha_t and enc_hiddens to obtain the
        # attention output vector, a_t.
        # $$ Hints:
        #  - alpha_t is shape (b, src_len)
        # - enc_hiddens is shape (b, src_len, 2h)
        # - a_t should be shape (b, 2h)
        # - You will need to do some squeezing and unsqueezing.
        # Note: b = batch size, src_len = maximum source length, h = hidden size.
        ###
        # 3. Concatenate dec_hidden with a_t to compute tensor U_t
        # 4. Apply the combined output projection layer to U_t to compute tensor V_t
        # 5. Compute tensor O_t by first applying the Tanh function and then the dropout layer.
        ###
        # Use the following docs to implement this functionality:
        # Softmax:
        # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.softmax
        # Batch Multiplication:
        # https://pytorch.org/docs/stable/torch.html#torch.bmm
        # Tensor View:
        # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view
        # Tensor Concatenation:
        # https://pytorch.org/docs/stable/torch.html#torch.cat
        # Tanh:
        # https://pytorch.org/docs/stable/torch.html#torch.tanh

        # e_t: (b, src_len) -> alpha_t: (b, src_len)
        alpha_t = F.softmax(e_t, dim=1)

        # alpha_t: (b, src_len) -> (b, 1, src_len)
        # enc_hiddens: (b, src_len, 2*h)
        # a_t: (b, 1, 2*h) -> (b, 2*h)
        a_t = torch.squeeze(torch.bmm(torch.unsqueeze(alpha_t, dim=1), enc_hiddens),
                            dim=1)

        # dec_hidden: (b, h)
        # u_t: (b, 3*h)
        u_t = torch.cat([a_t, dec_hidden], dim=1)
        # v_t: (b, h)
        v_t = self.combined_output_projection(u_t)
        # O_t: (b, h)
        O_t = self.dropout(torch.tanh(v_t))

        # END YOUR CODE

        combined_output = O_t
        return dec_state, combined_output, e_t

    def forward(self, batch):
        context, context_lengths, context_limited, context_elmo = batch.context, batch.context_lengths, batch.context_limited, batch.context_elmo
        question, question_lengths, question_limited, question_elmo = batch.question, batch.question_lengths, batch.question_limited, batch.question_elmo
        answer, answer_lengths, answer_limited = batch.answer, batch.answer_lengths, batch.answer_limited
        oov_to_limited_idx, limited_idx_to_full_idx = batch.oov_to_limited_idx, batch.limited_idx_to_full_idx

        def map_to_full(x):
            return limited_idx_to_full_idx[x]

        self.map_to_full = map_to_full

        if -1 not in self.args.elmo:
            def elmo(z, layers, device):
                e = self.elmo(batch_to_ids(z).to(device))['elmo_representations']
                return torch.cat([e[x] for x in layers], -1)

            context_elmo = self.project_elmo(elmo(context_elmo, self.args.elmo, context.device).detach())
            question_elmo = self.project_elmo(elmo(question_elmo, self.args.elmo, question.device).detach())

        if self.args.glove_and_char:
            context_embedded = self.encoder_embeddings(context)
            question_embedded = self.encoder_embeddings(question)
            if self.args.cove:
                context_embedded = self.project_cove(
                    torch.cat([self.cove(context_embedded[:, :, -300:], context_lengths), context_embedded],
                              -1).detach())
                question_embedded = self.project_cove(
                    torch.cat([self.cove(question_embedded[:, :, -300:], question_lengths), question_embedded],
                              -1).detach())
            if -1 not in self.args.elmo:
                context_embedded = self.project_embeddings(torch.cat([context_embedded, context_elmo], -1))
                question_embedded = self.project_embeddings(torch.cat([question_embedded, question_elmo], -1))
        else:
            context_embedded, question_embedded = context_elmo, question_elmo

        context_encoded = self.bilstm_before_coattention(context_embedded, context_lengths)[0]
        question_encoded = self.bilstm_before_coattention(question_embedded, question_lengths)[0]

        context_padding = context.data == self.pad_idx
        question_padding = question.data == self.pad_idx

        coattended_context, coattended_question = self.coattention(context_encoded, question_encoded, context_padding,
                                                                   question_padding)

        context_summary = torch.cat([coattended_context, context_encoded, context_embedded], -1)
        condensed_context, _ = self.context_bilstm_after_coattention(context_summary, context_lengths)
        self_attended_context = self.self_attentive_encoder_context(condensed_context, padding=context_padding)
        final_context, (context_rnn_h, context_rnn_c) = self.bilstm_context(self_attended_context[-1], context_lengths)
        context_rnn_state = [self.reshape_rnn_state(x) for x in (context_rnn_h, context_rnn_c)]

        question_summary = torch.cat([coattended_question, question_encoded, question_embedded], -1)
        condensed_question, _ = self.question_bilstm_after_coattention(question_summary, question_lengths)
        self_attended_question = self.self_attentive_encoder_question(condensed_question, padding=question_padding)
        final_question, (question_rnn_h, question_rnn_c) = self.bilstm_question(self_attended_question[-1],
                                                                                question_lengths)
        question_rnn_state = [self.reshape_rnn_state(x) for x in (question_rnn_h, question_rnn_c)]

        context_indices = context_limited if context_limited is not None else context
        question_indices = question_limited if question_limited is not None else question
        answer_indices = answer_limited if answer_limited is not None else answer

        pad_idx = self.field.decoder_stoi[self.field.pad_token]
        context_padding = context_indices.data == pad_idx
        question_padding = question_indices.data == pad_idx

        # TODO 1 must apply mask
        self.dual_ptr_rnn_decoder.applyMasks(context_padding, question_padding)

        # TODO 2 feed the hidden state to the decoder
        # and get the output
        # be noted that there will be different action in training and testing / validation

        # for now
        # decoder input: context_rnn_state, final_context

        if self.training:
            answer_padding = (answer_indices.data == pad_idx)[:, :-1]
            answer_embedded = self.decoder_embeddings(answer)
            self_attended_decoded = self.self_attentive_decoder(answer_embedded[:, :-1].contiguous(),
                                                                self_attended_context, context_padding=context_padding,
                                                                answer_padding=answer_padding,
                                                                positional_encodings=self.args.positional_encoding)
            decoder_outputs = self.dual_ptr_rnn_decoder(self_attended_decoded,
                                                        final_context, final_question, hidden=context_rnn_state)
            rnn_output, context_attention, question_attention, context_alignment, question_alignment, vocab_pointer_switch, context_question_switch, rnn_state = decoder_outputs

            probs = self.probs(self.out, rnn_output, vocab_pointer_switch, context_question_switch,
                               context_attention, question_attention,
                               context_indices, question_indices,
                               oov_to_limited_idx)

            probs, targets = mask(answer_indices[:, 1:].contiguous(), probs.contiguous(), pad_idx=pad_idx)
            loss = F.nll_loss(probs.log(), targets)
            return loss, None
        else:
            return None, self.greedy(self_attended_context, final_context, final_question,
                                     context_indices, question_indices,
                                     oov_to_limited_idx, rnn_state=context_rnn_state).data

    def reshape_rnn_state(self, h):
        return h.view(h.size(0) // 2, 2, h.size(1), h.size(2)) \
            .transpose(1, 2).contiguous() \
            .view(h.size(0) // 2, h.size(1), h.size(2) * 2).contiguous()

    def probs(self, generator, outputs, vocab_pointer_switches, context_question_switches,
              context_attention, question_attention,
              context_indices, question_indices,
              oov_to_limited_idx):

        size = list(outputs.size())

        size[-1] = self.generative_vocab_size
        scores = generator(outputs.view(-1, outputs.size(-1))).view(size)
        p_vocab = F.softmax(scores, dim=scores.dim() - 1)
        scaled_p_vocab = vocab_pointer_switches.expand_as(p_vocab) * p_vocab

        effective_vocab_size = self.generative_vocab_size + len(oov_to_limited_idx)
        if self.generative_vocab_size < effective_vocab_size:
            size[-1] = effective_vocab_size - self.generative_vocab_size
            buff = scaled_p_vocab.new_full(size, EPSILON)
            scaled_p_vocab = torch.cat([scaled_p_vocab, buff], dim=buff.dim() - 1)

        # p_context_ptr
        scaled_p_vocab.scatter_add_(scaled_p_vocab.dim() - 1, context_indices.unsqueeze(1).expand_as(context_attention),
                                    (context_question_switches * (1 - vocab_pointer_switches)).expand_as(
                                        context_attention) * context_attention)

        # p_question_ptr
        scaled_p_vocab.scatter_add_(scaled_p_vocab.dim() - 1,
                                    question_indices.unsqueeze(1).expand_as(question_attention),
                                    ((1 - context_question_switches) * (1 - vocab_pointer_switches)).expand_as(
                                        question_attention) * question_attention)

        return scaled_p_vocab

    def greedy(self, self_attended_context, context, question, context_indices, question_indices, oov_to_limited_idx,
               rnn_state=None):
        B, TC, C = context.size()
        T = self.args.max_output_length
        outs = context.new_full((B, T), self.field.decoder_stoi['<pad>'], dtype=torch.long)
        hiddens = [self_attended_context[0].new_zeros((B, T, C))
                   for l in range(len(self.self_attentive_decoder.layers) + 1)]
        hiddens[0] = hiddens[0] + positional_encodings_like(hiddens[0])
        eos_yet = context.new_zeros((B,)).byte()

        rnn_output, context_alignment, question_alignment = None, None, None
        for t in range(T):
            if t == 0:
                embedding = self.decoder_embeddings(
                    self_attended_context[-1].new_full((B, 1), self.field.vocab.stoi['<init>'], dtype=torch.long),
                    [1] * B)
            else:
                embedding = self.decoder_embeddings(outs[:, t - 1].unsqueeze(1), [1] * B)
            hiddens[0][:, t] = hiddens[0][:, t] + (math.sqrt(self.self_attentive_decoder.d_model) * embedding).squeeze(
                1)
            for l in range(len(self.self_attentive_decoder.layers)):
                hiddens[l + 1][:, t] = self.self_attentive_decoder.layers[l].feedforward(
                    self.self_attentive_decoder.layers[l].attention(
                        self.self_attentive_decoder.layers[l].selfattn(hiddens[l][:, t], hiddens[l][:, :t + 1],
                                                                       hiddens[l][:, :t + 1])
                        , self_attended_context[l], self_attended_context[l]))
            decoder_outputs = self.dual_ptr_rnn_decoder(hiddens[-1][:, t].unsqueeze(1),
                                                        context, question,
                                                        context_alignment=context_alignment,
                                                        question_alignment=question_alignment,
                                                        hidden=rnn_state, output=rnn_output)
            rnn_output, context_attention, question_attention, context_alignment, question_alignment, vocab_pointer_switch, context_question_switch, rnn_state = decoder_outputs
            probs = self.probs(self.out, rnn_output, vocab_pointer_switch, context_question_switch,
                               context_attention, question_attention,
                               context_indices, question_indices,
                               oov_to_limited_idx)
            pred_probs, preds = probs.max(-1)
            preds = preds.squeeze(1)
            eos_yet = eos_yet | (preds == self.field.decoder_stoi['<eos>']).byte()
            outs[:, t] = preds.cpu().apply_(self.map_to_full)
            if eos_yet.all():
                break
        return outs


class DualPtrRNNDecoder(nn.Module):

    def __init__(self, d_in, d_hid, dropout=0.0, num_layers=1):
        super().__init__()
        self.d_hid = d_hid
        self.d_in = d_in
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.input_feed = True
        if self.input_feed:
            d_in += 1 * d_hid

        self.rnn = LSTMDecoder(self.num_layers, d_in, d_hid, dropout)
        self.context_attn = LSTMDecoderAttention(d_hid, dot=True)
        self.question_attn = LSTMDecoderAttention(d_hid, dot=True)

        self.vocab_pointer_switch = nn.Sequential(Feedforward(2 * self.d_hid + d_in, 1), nn.Sigmoid())
        self.context_question_switch = nn.Sequential(Feedforward(2 * self.d_hid + d_in, 1), nn.Sigmoid())

    def forward(self, input, context, question, output=None, hidden=None, context_alignment=None,
                question_alignment=None):
        context_output = output.squeeze(1) if output is not None else self.make_init_output(context)
        context_alignment = context_alignment if context_alignment is not None else self.make_init_output(context)
        question_alignment = question_alignment if question_alignment is not None else self.make_init_output(question)

        context_outputs, vocab_pointer_switches, context_question_switches, context_attentions, question_attentions, context_alignments, question_alignments = [], [], [], [], [], [], []
        for emb_t in input.split(1, dim=1):
            emb_t = emb_t.squeeze(1)
            context_output = self.dropout(context_output)
            if self.input_feed:
                emb_t = torch.cat([emb_t, context_output], 1)
            dec_state, hidden = self.rnn(emb_t, hidden)
            context_output, context_attention, context_alignment = self.context_attn(dec_state, context)
            question_output, question_attention, question_alignment = self.question_attn(dec_state, question)
            vocab_pointer_switch = self.vocab_pointer_switch(torch.cat([dec_state, context_output, emb_t], -1))
            context_question_switch = self.context_question_switch(torch.cat([dec_state, question_output, emb_t], -1))
            context_output = self.dropout(context_output)
            context_outputs.append(context_output)
            vocab_pointer_switches.append(vocab_pointer_switch)
            context_question_switches.append(context_question_switch)
            context_attentions.append(context_attention)
            context_alignments.append(context_alignment)
            question_attentions.append(question_attention)
            question_alignments.append(question_alignment)

        context_outputs, vocab_pointer_switches, context_question_switches, context_attention, question_attention = [
            self.package_outputs(x) for x in
            [context_outputs, vocab_pointer_switches, context_question_switches, context_attentions,
             question_attentions]]
        return context_outputs, context_attention, question_attention, context_alignment, question_alignment, vocab_pointer_switches, context_question_switches, hidden

    def applyMasks(self, context_mask, question_mask):
        self.context_attn.applyMasks(context_mask)
        self.question_attn.applyMasks(question_mask)

    def make_init_output(self, context):
        batch_size = context.size(0)
        h_size = (batch_size, self.d_hid)
        return context.new_zeros(h_size)

    def package_outputs(self, outputs):
        outputs = torch.stack(outputs, dim=1)
        return outputs
