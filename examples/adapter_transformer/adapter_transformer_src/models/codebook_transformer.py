# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from typing import Any, Dict, List, Optional, Tuple
from torch import Tensor

import torch.nn as nn
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import (
    TransformerDecoder,
    TransformerEncoder,
    base_architecture,
)

from fairseq.models.transformer import TransformerModel
import torch.nn.functional as F


@register_model("codebook_transformer")
class CodebookTransformerModel(TransformerModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        TransformerModel.add_args(parser)
        parser.add_argument(
            "--codebook-size",
            default=1000,
            type=int,
            help="Number of entries in the codebook",
        )
        parser.add_argument(
            "--codebook-slices",
            default=1,
            type=int,
            help="Number of slices in codebook",
        )
        parser.add_argument(
            "--alternate-cont-chance",
            default=0.0,
            type=float,
            help="Probability of seeing continuous encoder hidden states",
        )
        parser.add_argument(
            "--quant-lang-tok",
            default=False,
            action="store_true",
            help="Init codebook embedding weights with uniform distribution",
        )
        # Add similarity loss

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return CodebookTransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return CodebookTransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
            )

class CodebookTransformerEncoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)

        assert args.codebook_size > 1, "Number of codebook entries must be great than 1"
        assert args.encoder_embed_dim % args.codebook_slices == 0, "Hidden dim must be divisible by number of slices"

        self.codebook = nn.Embedding(args.codebook_size, args.encoder_embed_dim)    #TODO: padding idx?

        print("LUT norm:", (self.codebook.weight ** 2).sum(-1).sqrt().mean())

        # if args.uniform_init_codebook:
        #     nn.init.uniform_(self.codebook.weight, -1.0, 1.0)

        self.n_slices = args.codebook_slices
        self.dims_per_slice = args.encoder_embed_dim // args.codebook_slices
        self.alternate_cont_chance = args.alternate_cont_chance
        self.quant_lang_tok = args.quant_lang_tok

        if self.quant_lang_tok:
            self.indicator_emb = nn.Embedding(1, args.encoder_embed_dim)

    def reorder_encoder_out(cls, encoder_out: Dict[str, List[Tensor]], new_order):
        enc_out = super().reorder_encoder_out(encoder_out, new_order)

        # for cont
        if len(encoder_out["encoder_out_cont"]) == 0:
            encoder_out_cont = []
        else:
            encoder_out_cont = [
                encoder_out["encoder_out_cont"][0].index_select(1, new_order)
            ]

        # for quant
        if len(encoder_out["encoder_out_quant"]) == 0:
            encoder_out_quant = []
        else:
            encoder_out_quant = [
                encoder_out["encoder_out_quant"][0].index_select(1, new_order)
            ]

        quantized_indices = encoder_out["quantized_indices"]

        enc_out["encoder_out_cont"] = encoder_out_cont
        enc_out["encoder_out_quant"] = encoder_out_quant
        enc_out["quantized_indices"] = quantized_indices

        return enc_out

    def forward_scriptable(
            self,
            src_tokens,
            src_lengths: Optional[torch.Tensor] = None,
            return_all_hiddens: bool = False,
            token_embeddings: Optional[torch.Tensor] = None,
    ):
        enc_out_dict = super().forward_scriptable(src_tokens,
                                                  src_lengths,
                                                  return_all_hiddens,
                                                  token_embeddings)
        # Look up codebook entries from encoder hidden states
        enc_out = enc_out_dict["encoder_out"][0]  # T x B x C
        quantized_indices, il_prob = self.lookup_nn(enc_out,
                                                    self.codebook,
                                                    slices=self.n_slices,
                                                    )
        if self.n_slices == 1:  # Only 1 slice
            # BxTxH --> TxBxH
            quantized = self.embedded_dropout(self.codebook.weight,
                                              quantized_indices[0],
                                              self.codebook.padding_idx
                                              ).transpose(0, 1)
        else:  # Multiple slices, concat
            quantized = []

            for s_idx, sliced_indices in enumerate(quantized_indices):
                s, e = s_idx * self.dims_per_slice, (s_idx + 1) * self.dims_per_slice
                sliced_context = self.embedded_dropout(self.codebook.weight[:, s:e],
                                                       sliced_indices,
                                                       self.codebook.padding_idx
                                                       ).transpose(0, 1)
                quantized.append(sliced_context)

            quantized = torch.cat(quantized, dim=-1)

        # Put language token in?
        if self.quant_lang_tok:
            quantized[0, :, :] += self.indicator_emb.weight

        # Straight-through
        enc_out_dict["encoder_out_quant"] = enc_out + (quantized - enc_out).detach()
        enc_out_dict["encoder_out_cont"] = enc_out

        # Toss a coin... to your witcher
#        if (self.alternate_cont_chance == 0) or (self.training and torch.rand(1)[0].item() > self.alternate_cont_chance):
        #     # no need to do another dropout
        enc_out_dict["encoder_out"][0] = enc_out_dict["encoder_out_quant"]
#        else:
#            enc_out_dict["encoder_out"][0] = enc_out_dict["encoder_out_cont"]

        enc_out_dict["quantized_indices"] = quantized_indices

        return enc_out_dict

    def embedded_dropout(self, embed_weight, words, padding_idx=None): #dropout=0.1, scale=None
        # masked_embed_weight = embed.weight
        # padding_idx = embed.padding_idx

        if padding_idx is None:
            padding_idx = -1

        x = F.embedding(
            words,
            embed_weight,
            padding_idx,
        )
        return x

    def lookup_nn(self,
                  il_lookup_input,
                  il_word_lut,
                  sampling=False,
                  temperature=10.0,
                  slices=1,
                  ignore_first_codebook_entry=False):
        out, out_ = [], []
        dist, dist_ = [], []

        dims_per_slice = int(il_lookup_input.shape[-1] / slices)

        slice_flat_input = torch.split(il_lookup_input, split_size_or_sections=dims_per_slice, dim=-1)
        slice_flat_lut = torch.split(il_word_lut.weight, split_size_or_sections=dims_per_slice, dim=-1)
        sliced_orig_shape = [il_lookup_input.shape[0], il_lookup_input.shape[1]]

        s_idx = 0
        for sliced_input, sliced_word_lut in zip(slice_flat_input, slice_flat_lut):
            sliced_flat_input = sliced_input.view(-1, dims_per_slice)
            # https://colab.research.google.com/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb
            # (T*B) x V
            distances = (torch.sum(sliced_flat_input ** 2, dim=1, keepdim=True)
                         + torch.sum(sliced_word_lut ** 2, dim=1)
                         - 2 * torch.matmul(sliced_flat_input, sliced_word_lut.t()))

            # If using CTC similarity loss, modify this to be never taking index 0 of LUT
            if ignore_first_codebook_entry:
                distances = distances[:, 1:]

            if sampling:
                il_tokens = torch.topk(distances, k=int(temperature), dim=1, largest=False)[1]  # (TxB) x k
                sampled_ids = torch.randint(low=0, high=int(temperature), size=(il_tokens.shape[0],),
                                            device=(il_tokens.device))
                il_tokens = il_tokens.gather(1, sampled_ids.view(-1, 1))
            else:
                il_tokens = torch.argmin(distances, dim=1).unsqueeze(1)

            il_tokens = il_tokens.squeeze(1).view(sliced_orig_shape[0], sliced_orig_shape[1]).transpose(0, 1)
            out.append(il_tokens)
            dist.append(distances)

            if torch.rand(1)[0].item() < 0.001:
                print(s_idx, il_tokens[0])

            s_idx += 1

        return out, dist


class CodebookTransformerDecoder(TransformerDecoder):
    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn, output_projection)

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        x, extra = super().extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )

        # Additionally return encoder output
        extra["encoder_out_quant"] = encoder_out["encoder_out_quant"]
        extra["encoder_out_cont"] = encoder_out["encoder_out_cont"]

        return x, extra


@register_model_architecture(
    "codebook_transformer", "codebook_transformer"
)
def codebook_transformer_architecture(args):
    base_architecture(args)


