# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
    label_smoothed_nll_loss
)
import torch
import math
from fairseq import utils


@register_criterion("encoder_similarity_label_smoothed_cross_entropy")
class EncoderSimilarityLabelSmoothedCrossEntropyCriterion(
    LabelSmoothedCrossEntropyCriterion
):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
        similarity_regularization_weight=0,
    ):
        super().__init__(
                task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy
                )
        self.similarity_regularization_weight = similarity_regularization_weight


    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # 1) forward pass for src -> tgt
        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )

        ####################################################
        # sample has the following keys:
            # id
            # nsentences
            # ntokens   --> number of total target tokens
            # net_input
                # src_tokens    --> source token indices
                # src_lengths   --> source length tensor (for each instance)
                # prev_output_tokens --> target (indices) shifted left
            # target --> target token (indices)
        ####################################################
        
        src_lengths = sample["net_input"]["src_lengths"]
        # shift 1 position on dimension 1
        src_tokens = sample["net_input"]["src_tokens"] #roll, 1, 1)
#        bos = src_tokens[:,0]
#        eos = [s[src_len[i] - 1] for i, s in enumerate(src_tokens)]
        prev_input_tokens = torch.roll(src_tokens, 1, 1)
        prev_input_tokens[:, 0] = 2 # EOS TODO: check EOS idx from model

#        print("BOS", bos)
#        print("EOS", eos)
        max_len = src_tokens.shape[1]
        for i, s in enumerate(prev_input_tokens):
            if src_lengths[i] != max_len:
                # change from EOS to PAD
                # not src_len - 1 because already rolled to right
                prev_input_tokens[i][src_lengths[i]] = 1 # PAD #TODO: check PAD idx from model

        # 2) forward pass for reverse direction (tgt -> src)
        net_output_reverse = model(
                src_tokens=sample["target"],
                src_lengths=None,
                prev_output_tokens=prev_input_tokens
                )

        loss_reverse, nll_loss_reverse = self.compute_loss(
                model, net_output_reverse, sample, reduce=reduce,
                reverse=True
                )
#        print("ORIG:", loss, nll_loss)
#        print("REVERSE:", loss_reverse, nll_loss_reverse)
        loss += loss_reverse
        
        ntokens_src = sample["net_input"]["src_lengths"].sum().item()
        sample_size_reverse = (
            sample["src_tokens"].size(0) if self.sentence_avg else ntokens_src
        )
        sample_size += sample_size_reverse

        # 3) loss by comparing encoder similarity
#        if self.similarity_regularization_weight:
        sim_loss = self.compute_similarity_loss(net_output, net_output_reverse)
#        print("SIM", sim_loss)
        loss += sim_loss * self.similarity_regularization_weight
        
        logging_output = {
            "loss": loss.data + loss_reverse.data,
            "nll_loss": nll_loss.data + nll_loss_reverse.data,
            "ntokens": sample["ntokens"] + ntokens_src,
            "nsentences": sample["target"].size(0) * 2,
            "sample_size": sample_size,
            "similarity_loss": sim_loss.item(),
        }

        if self.report_accuracy:    # TODO: make this for both directions too
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)

#        print(logging_output)
        return loss, sample_size, logging_output


    def compute_loss(self, model, net_output, sample, reduce=True, reverse=False):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample, reverse=reverse)

        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss


    def get_lprobs_and_target(self, model, net_output, sample, reverse=False):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        
        if not reverse:
            target = model.get_targets(sample, net_output)
        else:
            target = sample["net_input"]["src_tokens"]

        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)


    def compute_similarity_loss(self, net_output, net_output_reverse):
        # use [1] to access extra properties; 
        # look up key "encoder_out"; 
        # take first element of list
        enc_out = net_output[1]["encoder_out"][0]   # T x B x C
        enc_out_rev = net_output_reverse[1]["encoder_out"][0]
        enc_mask = net_output[1]["encoder_padding_mask"][0]    # B x T
        enc_mask_rev = net_output_reverse[1]["encoder_padding_mask"][0]
        
        # zero out padded positions
#        _enc_out = enc_out.masked_fill(enc_mask.transpose(0,1).unsqueeze(-1), 0.0).type_as(enc_out)
#        _enc_out_rev = enc_out_rev.masked_fill(enc_mask_rev.transpose(0,1).unsqueeze(-1), 0.0).type_as(enc_out_rev)
#        _enc_out[enc_mask.transpose(0,1)] = 0.0
#        _enc_out_rev[enc_mask_rev.transpose(0,1)] = 0.0
        
        # B x C / B x 1 --> B x C
        meanpool_enc_out = enc_out[~enc_mask.transpose(0,1)].sum(axis=0) / (~enc_mask).sum(axis=1).unsqueeze(-1)
        meanpool_enc_out_rev = enc_out_rev[~enc_mask_rev.transpose(0,1)].sum(axis=0) / (~enc_mask_rev).sum(axis=1).unsqueeze(-1)
    
        B, C = enc_out.shape[1], enc_out.shape[2]
        diff = ((((meanpool_enc_out - meanpool_enc_out_rev) / math.sqrt(C))) ** 2).sum()

        return diff


    @staticmethod
    def add_args(parser):
        super(
            EncoderSimilarityLabelSmoothedCrossEntropyCriterion,
            EncoderSimilarityLabelSmoothedCrossEntropyCriterion,
        ).add_args(parser)
        # fmt: off

        """Add criterion-specific arguments to the parser."""
        parser.add_argument(
            "--label-smoothing",
            default=0.0,
            type=float,
            metavar="D",
            help="epsilon for label smoothing, 0 means no label smoothing",
        )

        parser.add_argument(
            "--similarity-regularization-weight",
            default=0.0,
            type=float,
            metavar="D",
            help="weight for similarity regularization, 0 means no similarity regularization",
        )

        # TODO: 
        # assert padding for source and target have to be on the same side
        # assert language token setup: src on src; tgt on tgt

