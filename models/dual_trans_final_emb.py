import os
import math
import numpy as np
import pdb
import torch
from torch import nn
from torch.nn import functional as F

from util import get_trainable_params

from cove import MTLSTM
from allennlp.modules.elmo import Elmo, batch_to_ids

from .common import positional_encodings_like, INF, EPSILON, TransformerEncoder, TransformerDecoder, PackedLSTM, \
    LSTMDecoderAttention, LSTMDecoder, Embedding, Feedforward, mask, CoattentiveLayer, CoattentiveTransformerEncoder, \
    DualTransformerEncoder, FinalEmbedding


class DualTransNoSelfNetVFinal(nn.Module):

    def __init__(self, field, args):
        super().__init__()
        self.field = field
        self.args = args
        self.pad_idx = self.field.vocab.stoi[self.field.pad_token]

        def dp(args):
            return args.dropout_ratio if args.rnn_layers > 1 else 0.

        self.encoder_embeddings = FinalEmbedding(field, args.dimension,
                                                 dropout=args.dropout_ratio, project=True,
                                                 requires_grad=False)

        # TODO why not use shared embedding
        self.decoder_embeddings = FinalEmbedding(field, args.dimension,
                                                 dropout=args.dropout_ratio, project=True,
                                                 requires_grad=True)

        self.bilstm_share = PackedLSTM(args.dimension, args.dimension,
                                       batch_first=True, dropout=dp(args), bidirectional=True,
                                       num_layers=args.rnn_layers)

        self.coattention = DualTransformerEncoder(args.dimension, args.transformer_heads,
                                                  args.transformer_hidden, args.transformer_layers,
                                                  args.dropout_ratio)

        self.bilstm_context = PackedLSTM(args.dimension, args.dimension,
                                         batch_first=True, dropout=dp(args), bidirectional=True,
                                         num_layers=args.rnn_layers)

        self.bilstm_question = PackedLSTM(args.dimension, args.dimension,
                                          batch_first=True, dropout=dp(args), bidirectional=True,
                                          num_layers=args.rnn_layers)

        self.dual_ptr_rnn_decoder = DualPtrRNNDecoder(args.dimension, args.dimension,
                                                      dropout=args.dropout_ratio, num_layers=args.rnn_layers)

        self.generative_vocab_size = min(len(field.vocab), args.max_generative_vocab)
        self.out = nn.Linear(args.dimension, self.generative_vocab_size)

        self.dropout = nn.Dropout(0.4)

    def forward(self, batch):
        context, context_lengths, context_limited, context_elmo = batch.context, batch.context_lengths, batch.context_limited, batch.context_elmo
        question, question_lengths, question_limited, question_elmo = batch.question, batch.question_lengths, batch.question_limited, batch.question_elmo
        answer, answer_lengths, answer_limited = batch.answer, batch.answer_lengths, batch.answer_limited
        oov_to_limited_idx, limited_idx_to_full_idx = batch.oov_to_limited_idx, batch.limited_idx_to_full_idx

        def map_to_full(x):
            return limited_idx_to_full_idx[x]

        self.map_to_full = map_to_full

        context_embedded = self.encoder_embeddings(context)
        question_embedded = self.decoder_embeddings(question)

        context_padding = context.data == self.pad_idx
        question_padding = question.data == self.pad_idx

        encoded_context, _ = self.bilstm_share(context_embedded, context_lengths)
        encoded_question, _ = self.bilstm_share(question_embedded, question_lengths)
        encoded_context = self.dropout(encoded_context)
        encoded_question = self.dropout(encoded_question)

        coattended = self.coattention(encoded_context, encoded_question,
                                      context_padding, question_padding)

        coattended_context = [c[0] for c in coattended]
        coattended_question = [c[1] for c in coattended]
        final_context, (context_rnn_h, context_rnn_c) = self.bilstm_context(coattended_context[-1], context_lengths)
        context_rnn_state = [self.reshape_rnn_state(x) for x in (context_rnn_h, context_rnn_c)]

        final_question, (question_rnn_h, question_rnn_c) = self.bilstm_question(coattended_question[-1],
                                                                                question_lengths)

        context_indices = context_limited if context_limited is not None else context
        question_indices = question_limited if question_limited is not None else question
        answer_indices = answer_limited if answer_limited is not None else answer

        pad_idx = self.field.decoder_stoi[self.field.pad_token]
        context_padding = context_indices.data == pad_idx
        question_padding = question_indices.data == pad_idx

        self.dual_ptr_rnn_decoder.applyMasks(context_padding, question_padding)

        if self.training:
            answer_embedded = self.decoder_embeddings(answer)
            decoder_outputs = self.dual_ptr_rnn_decoder(answer_embedded[:, :-1],
                                                        final_context, final_question,
                                                        hidden=context_rnn_state)
            rnn_output, question_attention, _, context_question_switch, _ = decoder_outputs

            probs = self.probs(self.out, rnn_output, context_question_switch,
                               question_attention,
                               question_indices,
                               oov_to_limited_idx)

            probs, targets = mask(answer_indices[:, 1:].contiguous(), probs.contiguous(), pad_idx=pad_idx)
            loss = F.nll_loss(probs.log(), targets)
            return loss, None
        else:
            return None, self.greedy(final_context, final_question,
                                     context_indices, question_indices,
                                     oov_to_limited_idx, rnn_state=context_rnn_state).data

            # return None, self.beam_search(final_context, final_question,
            #                               context_indices, question_indices,
            #                               oov_to_limited_idx,
            #                               context_padding, question_padding,
            #                               rnn_state=context_rnn_state,
            #                               ).data

    def reshape_rnn_state(self, h):
        return h.view(h.size(0) // 2, 2, h.size(1), h.size(2)) \
            .transpose(1, 2).contiguous() \
            .view(h.size(0) // 2, h.size(1), h.size(2) * 2).contiguous()

    def probs(self, generator, outputs, context_question_switches,
              question_attention,
              question_indices,
              oov_to_limited_idx):

        size = list(outputs.size())

        size[-1] = self.generative_vocab_size
        scores = generator(outputs.view(-1, outputs.size(-1))).view(size)
        p_vocab = F.softmax(scores, dim=scores.dim() - 1)
        scaled_p_vocab = context_question_switches.expand_as(p_vocab) * p_vocab

        effective_vocab_size = self.generative_vocab_size + len(oov_to_limited_idx)
        if self.generative_vocab_size < effective_vocab_size:
            size[-1] = effective_vocab_size - self.generative_vocab_size
            buff = scaled_p_vocab.new_full(size, EPSILON)
            scaled_p_vocab = torch.cat([scaled_p_vocab, buff], dim=buff.dim() - 1)

        # p_question_ptr
        scaled_p_vocab.scatter_add_(scaled_p_vocab.dim() - 1,
                                    question_indices.unsqueeze(1).expand_as(question_attention),
                                    (1 - context_question_switches).expand_as(
                                        question_attention) * question_attention)

        return scaled_p_vocab

    def greedy(self, context, question, context_indices, question_indices, oov_to_limited_idx,
               rnn_state=None):
        B, TC, C = context.size()
        T = self.args.max_output_length
        outs = context.new_full((B, T), self.field.decoder_stoi['<pad>'], dtype=torch.long)
        hidden = context.new_zeros((B, T, C))

        eos_yet = context.new_zeros((B,)).byte()

        rnn_output, context_alignment, question_alignment = None, None, None
        for t in range(T):
            if t == 0:
                embedding = self.decoder_embeddings(
                    context.new_full((B, 1), self.field.vocab.stoi['<init>'], dtype=torch.long),
                    [1] * B)
            else:
                embedding = self.decoder_embeddings(outs[:, t - 1].unsqueeze(1), [1] * B)
            hidden[:, t] = hidden[:, t] + embedding.squeeze(1)

            decoder_outputs = self.dual_ptr_rnn_decoder(hidden[:, t].unsqueeze(1),
                                                        context, question,
                                                        question_alignment=question_alignment,
                                                        hidden=rnn_state, output=rnn_output)
            rnn_output, question_attention, question_alignment, context_question_switch, rnn_state = decoder_outputs
            probs = self.probs(self.out, rnn_output, context_question_switch,
                               question_attention,
                               question_indices,
                               oov_to_limited_idx)
            pred_probs, preds = probs.max(-1)
            preds = preds.squeeze(1)
            eos_yet = eos_yet | (preds == self.field.decoder_stoi['<eos>']).byte()
            outs[:, t] = preds.cpu().apply_(self.map_to_full)
            if eos_yet.all():
                break
        return outs

    def beam_search(self, context, question, context_indices, question_indices,
                    oov_to_limited_idx, context_padding, question_padding,
                    rnn_state=None):
        if rnn_state is not None:
            rnn_state = [r.squeeze(0) for r in rnn_state]

        outs = []

        for c, q, ci, qi, cp, qp, r0, r1 in zip(context, question, context_indices, question_indices,
                                                context_padding, question_padding,
                                                rnn_state[0], rnn_state[1]):
            out = self._beam_search(c, q, ci, qi, oov_to_limited_idx, cp, qp, [r0, r1])
            outs.append(out)

        return torch.stack(outs)

    def _beam_search(self, context, question, context_indices, question_indices,
                     oov_to_limited_idx,
                     context_padding, question_padding,
                     rnn_state=None):
        assert len(list(context.size())) == 2, 'beam search only support batch size as 1'
        TC, C = context.size()
        TQ, C = question.size()
        B = self.args.beam_size
        T = self.args.max_output_length
        context, question, context_indices, question_indices = context.unsqueeze(0), question.unsqueeze(
            0), context_indices.unsqueeze(0), question_indices.unsqueeze(0)
        context_padding, question_padding = context_padding.unsqueeze(0), question_padding.unsqueeze(0)
        outs = context.new_full((B, T), self.field.decoder_stoi['<pad>'], dtype=torch.long)
        hidden = context.new_zeros((B, T, C))

        eos_yet = context.new_zeros((B,)).byte()

        scores = context.new_ones(B)
        # we make the starting batch dimension to B to set uniform interface
        context_exp = context.expand(B, TC, C)
        question_exp = question.expand(B, TQ, C)
        context_indices_exp = context_indices.expand(B, TC)
        question_indices_exp = question_indices.expand(B, TQ)
        context_padding_exp = context_padding.expand(B, TC)
        question_padding_exp = question_padding.expand(B, TQ)
        self.dual_ptr_rnn_decoder.applyMasks(context_padding_exp, question_padding_exp)

        rnn_state_exp = [r.expand(1, B, C) for r in rnn_state]

        rnn_output, context_alignment, question_alignment = None, None, None
        for t in range(T):
            if t == 0:
                embedding = self.decoder_embeddings(
                    context_exp.new_full((B, 1), self.field.vocab.stoi['<init>'], dtype=torch.long),
                    [1] * B)
            else:
                embedding = self.decoder_embeddings(outs[:, t - 1].unsqueeze(1), [1] * B)
            hidden[:, t] = hidden[:, t] + embedding.squeeze(1)

            decoder_outputs = self.dual_ptr_rnn_decoder(hidden[:, t].unsqueeze(1),
                                                        context_exp, question_exp,
                                                        question_alignment=question_alignment,
                                                        hidden=rnn_state_exp, output=rnn_output)
            rnn_output, question_attention, question_alignment, context_question_switch, rnn_state_exp = decoder_outputs
            probs = self.probs(self.out, rnn_output, context_question_switch,
                               question_attention,
                               question_indices_exp,
                               oov_to_limited_idx)

            # probs： B, V
            # in here, we select top B word for every generated sent
            # then we update the scores for the B**2 item
            # then we mask out and only leave the top B item for the next generation
            topB_probs, topB_preds = probs.topk(B, -1)
            # pdb.set_trace()

            if t != 0:
                temp_scores = scores.repeat(B).reshape(-1)  # B*B
                topB_probs = topB_probs.T.reshape(-1)  # B*B
                topB_preds = topB_preds.T.reshape(-1)  # B*B
                temp_scores = temp_scores * topB_probs
            else:
                topB_probs, topB_preds = topB_probs[0].reshape(-1), topB_preds[0].reshape(-1)
                temp_scores = topB_probs

            topB_scores, topB_scores_idx = temp_scores.topk(B, -1)
            scores = topB_scores
            cur_outs = topB_preds[topB_scores_idx]

            # pdb.set_trace()

            if t != 0:
                prev_outs = outs[:, t - 1].repeat(B).reshape(-1)  # B*B
                outs[:, t - 1] = prev_outs[topB_scores_idx]
                rnn_state_exp = [r.squeeze(0).repeat(B, 1)[topB_scores_idx].unsqueeze(0) for r in rnn_state_exp]
                rnn_output = rnn_output.repeat(B, 1, 1)[topB_scores_idx]

            # cur_outs = cur_outs.squeeze(1)
            eos_yet = eos_yet | (cur_outs == self.field.decoder_stoi['<eos>']).byte()
            outs[:, t] = cur_outs.cpu().apply_(self.map_to_full)
            if eos_yet.all():
                break
        _, max_id = scores.max(-1)
        outs = outs[max_id]
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

        self.context_question_switch = nn.Sequential(Feedforward(2 * self.d_hid + d_in, 1), nn.Sigmoid())

    def forward(self, input, context, question, output=None, hidden=None,
                question_alignment=None):
        context_output = output.squeeze(1) if output is not None else self.make_init_output(context)
        question_alignment = question_alignment if question_alignment is not None else self.make_init_output(question)

        context_outputs, context_question_switches, question_attentions, question_alignments = [], [], [], []
        for emb_t in input.split(1, dim=1):
            emb_t = emb_t.squeeze(1)
            context_output = self.dropout(context_output)
            if self.input_feed:
                emb_t = torch.cat([emb_t, context_output], 1)
            dec_state, hidden = self.rnn(emb_t, hidden)
            context_output, context_attention, context_alignment = self.context_attn(dec_state, context)
            question_output, question_attention, question_alignment = self.question_attn(dec_state, question)
            context_question_switch = self.context_question_switch(torch.cat([dec_state, question_output, emb_t], -1))
            context_output = self.dropout(context_output)
            context_outputs.append(context_output)
            context_question_switches.append(context_question_switch)
            question_attentions.append(question_attention)
            question_alignments.append(question_alignment)

        context_outputs, context_question_switches, question_attention = [
            self.package_outputs(x) for x in
            [context_outputs, context_question_switches,
             question_attentions]]
        return context_outputs, question_attention, question_alignment, context_question_switches, hidden

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
