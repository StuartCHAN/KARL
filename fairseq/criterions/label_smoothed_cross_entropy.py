# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

from fairseq import utils

from . import FairseqCriterion, register_criterion

import numpy as np 
import time 


@register_criterion('label_smoothed_cross_entropy')
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        } 
        prev_output_tokens = sample["net_input"]['prev_output_tokens']
        tgt_tokens = sample["target"]
        #print("logging_output = ", logging_output )
        return loss, sample_size, logging_output, prev_output_tokens, tgt_tokens 

    def compute_loss(self, model, net_output, sample, reduce=True):
        '''
        print("\n")
        print("net_output : ", len(net_output) )
        _ = [ print("\n ",  output) for output in net_output ]
        print("sample : ", sample.keys() )
        _ = [print(key," : ", sample[key]) for key in sample.keys()]
        print("\n - net_input : ")
        print('prev_output_tokens : ', sample["net_input"]['prev_output_tokens'])
        #samp_prev_output_tokens = sample["net_input"]['prev_output_tokens']
        print("\n - target : ")
        print(sample["target"])
        #samp_target = sample["target"] 
        save_tensors(sample)
        #print('prev_output_tokens : ', sample["net_input"]['prev_output_tokens'])'''
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        '''
        print("lprobs : ", lprobs.size() )
        print(lprobs)
        print("lprobs : ", lprobs.size() )
        print(lprobs)
        print("target : ", target.size() )
        print(target)'''
        non_pad_mask = target.ne(self.padding_idx)
        '''print("non_pad_mask : ", non_pad_mask.size() )
        print(non_pad_mask)'''
        nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        '''print("nll_loss : ", nll_loss.size() )
        print(nll_loss) 
        print("smooth_loss : ", smooth_loss.size() )
        print(smooth_loss)'''
        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss
        return loss, nll_loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }

"""
def save_tensors(sample):
    samp_prev_output_tokens = sample["net_input"]['prev_output_tokens'].detach().numpy()
    samp_target = sample["target"].detach().numpy()
    stamp = str(time.time())
    np.save("./translations/seethetensors/"+stamp+".samp_prev_output_tokens.np", samp_prev_output_tokens)
    np.save("./translations/seethetensors/"+stamp+".samp_target.np", samp_target)
    print("* Sample tensors saved!")
    #return """