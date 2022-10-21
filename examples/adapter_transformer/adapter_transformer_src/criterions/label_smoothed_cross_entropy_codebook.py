# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
    label_smoothed_nll_loss
)
import torch.nn.functional as F
from fairseq import utils, metrics
from math import sqrt
import torch

@register_criterion("codebook_label_smoothed_cross_entropy")
class CodebookLabelSmoothedCrossEntropyCriterion(
    LabelSmoothedCrossEntropyCriterion
):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
        codebook_loss_weight=0,
        commitment_loss_weight=0,
    ):
        super().__init__(
                task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy
                )
        self.codebook_loss_weight = codebook_loss_weight
        self.commitment_loss_weight = commitment_loss_weight
        # self.minibatch_cnt = 0
        # step warmup??? of weights?
        # self.codebook_commitment_loss_scale

    def forward(self, model, sample, reduce=True):

        # if (self.minibatch_cnt)/ 8 < 2500:
        #     scaler = ((self.minibatch_cnt) // 8) / 2500
        #     l1 = self.codebook_loss_weight * scaler
        #     l2 = self.commitment_loss_weight * scaler
        # else:
        # l1 = self.codebook_loss_weight
        # l2 = self.commitment_loss_weight

        # self.minibatch_cnt += 1

        net_output = model(**sample["net_input"])
        # Normal XE loss
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)

        src_mask = (sample["net_input"]["src_tokens"] == 1).transpose(1, 0)   # TODO: change to not hard coded

        # Additional loss for learning codebook
        codebook_loss, commitment_loss, cont_norm, quant_norm = self.compute_codebook_commitment_loss(net_output, src_mask=src_mask)
        loss += codebook_loss * self.codebook_loss_weight + commitment_loss * self.commitment_loss_weight
        # print(loss, codebook_loss)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )

        ntokens_src = sample["net_input"]["src_lengths"].sum()

        # # log norm of enc out (continuous and quantized)

        # log_codebook_loss_per_src_tok = codebook_loss.item() / ntokens_src
        # print("log codebook loss per src tok", log_codebook_loss_per_src_tok)

        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "codebook_loss": codebook_loss.item(),
            "ntokens_src": ntokens_src,
            "cont_norm": cont_norm.item(),
            "quant_norm": quant_norm.item(),
        }

        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)

        return loss, sample_size, logging_output

    def compute_codebook_commitment_loss(self, net_output, src_mask):
        cont_context = net_output[1]["encoder_out_cont"]
        quant_context = net_output[1]["encoder_out_quant"]
        # h = sqrt(cont_context.shape[-1])

        # Do padding!!!!
        # based on source length
        # print((cont_context.detach() - quant_context.detach()).mean(), (cont_context.detach() - quant_context.detach()).max())
        # diff = cont_context.detach() - quant_context.detach()
        # print(diff.max(), diff.min())

        # cast to fp32
        cont_context = cont_context.float()
        quant_context = quant_context.float()

        cont_norm = ((cont_context ** 2).sum(-1)).sqrt().sum(0).sum(0)
        quant_norm = ((quant_context ** 2).sum(-1)).sqrt().sum(0).sum(0)

        # with torch.amp.autocast(enabled=False):   need a higher torch version
            # diff; to the power 2; sum over all hidden dims; sqrt; sum over T and B
        # codebook_loss = torch.sqrt(((cont_context.detach() - quant_context) ** 2).sum(axis=-1)).sum()  #  / math.sqrt(h)
        # commitment_loss = torch.sqrt(((cont_context - quant_context.detach()) ** 2).sum(axis=-1)).sum()    #  / math.sqrt(h)

        # codebookaoloss = ((cont_context.detach() - quant_context) ** 2).mean(axis=-1).sum()  # / math.sqrt(h)
        # commitment_loss = ((cont_context - quant_context.detach()) ** 2).mean(axis=-1).sum()  # / math.sqrt(h)

        # codebook_loss = F.mse_loss(input=quant_context, target=cont_context.detach(), reduction='none').sum(-1).sum(0).sum(0)
        # commitment_loss = F.mse_loss(input=cont_context, target=quant_context.detach(), reduction='none').sum(-1).sum(0).sum(0)

        codebook_loss = ((quant_context - cont_context.detach()) ** 2).sum(-1)
        commitment_loss = ((quant_context.detach() - cont_context) ** 2).sum(-1)

        # set padded positions to 0
        codebook_loss[src_mask] = 0.0
        commitment_loss[src_mask] = 0.0

        codebook_loss = codebook_loss.sum(0).sum(0)
        commitment_loss = commitment_loss.sum(0).sum(0)

        return codebook_loss, commitment_loss, cont_norm, quant_norm

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        super().reduce_metrics(logging_outputs)
        ntokens_src = sum(log.get("ntokens_src", 0) for log in logging_outputs)

        codebook_loss_sum = sum(log.get("codebook_loss", 0.0) for log in logging_outputs)
        if torch.is_tensor(codebook_loss_sum):
            codebook_loss_sum = codebook_loss_sum.item()

        cont_norm = sum(log.get("cont_norm", 0.0) for log in logging_outputs)
        quant_norm = sum(log.get("quant_norm", 0.0) for log in logging_outputs)

        if torch.is_tensor(cont_norm):
            cont_norm = cont_norm.item()

        if torch.is_tensor(quant_norm):
            quant_norm = quant_norm.item()

        # commitment_loss_sum = sum(log.get("commitment_loss", 0.0) for log in logging_outputs)
        # print(codebook_loss_sum)
        # print(codebook_loss_sum.item(), commitment_loss_sum.item(), ntokens)
        metrics.log_scalar(
            "codebook_loss", codebook_loss_sum / ntokens_src, ntokens_src, round=3
        )
        metrics.log_scalar(
            "codebook_loss_rel", (sqrt(codebook_loss_sum) / ntokens_src) / (cont_norm / ntokens_src), ntokens_src, round=5
        )
        metrics.log_scalar(
            "cont_norm", cont_norm / ntokens_src, ntokens_src, round=3
        )
        metrics.log_scalar(
            "quant_norm", quant_norm / ntokens_src, ntokens_src, round=3
        )


    @staticmethod
    def add_args(parser):
        super(
            CodebookLabelSmoothedCrossEntropyCriterion,
            CodebookLabelSmoothedCrossEntropyCriterion,
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
            "--codebook-loss-weight",
            default=0.0,
            type=float,
            metavar="D",
            help="TBA",
        )

        parser.add_argument(
            "--commitment-loss-weight",
            default=0.0,
            type=float,
            metavar="D",
            help="TBA",
        )

        parser.add_argument(
            "--codebook-losses-warmup-steps",
            default=2500,
            type=int,
            help="Linearly warm up weights on the codebook and commitment losess in the first x steps",
        )
