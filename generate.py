#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Translate pre-processed data with a trained model.
"""

import torch
from fairseq import bleu, checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.meters import StopwatchMeter, TimeMeter
import time 
import pickle 
import json 
import rl_utils.generator_utils as generator_utils 


def interprete(encoded_sparql):
    decoded_sparql = generator_utils.decode(encoded_sparql)
    sparql_query = generator_utils.fix_URI(decoded_sparql)
    return sparql_query 


def main(args):
    assert args.path is not None, '--path required for generation!'
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert args.replace_unk is None or args.raw_text, \
        '--replace-unk requires a raw text dataset (--raw-text)'

    utils.import_user_module(args)

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000
    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)

    # Set dictionaries
    try:
        src_dict = getattr(task, 'source_dictionary', None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    #pickle.dump(src_dict, open("./translations/seethetensors/src_dict.pkl", "bw") )
    #pickle.dump(tgt_dict, open("./translations/seethetensors/tgt_dict.pkl", "bw") )

    #print("* args.remove_bpe : ", args.remove_bpe)
    #bpe_symbol = args.remove_bpe
    #pickle.dump(bpe_symbol, open("./translations/seethetensors/bpe_symbol.pkl", "bw") )
    
    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(':'),
        arg_overrides=eval(args.model_overrides),
        task=task,
        bert_ratio=args.bert_ratio if args.change_ratio else None,
        encoder_ratio=args.encoder_ratio if args.change_ratio else None,
        geargs=args,
    )

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)
    #pickle.dump(align_dict,  open("./translations/seethetensors/align_dict.pkl", "bw"))
    # Load dataset (possibly sharded)
    
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)

    # Initialize generator
    gen_timer = StopwatchMeter()
    generator = task.build_generator(args)

    # Generate and compute BLEU score
    if args.sacrebleu:
        scorer = bleu.SacrebleuScorer()
    else:
        scorer = bleu.Scorer(tgt_dict.pad(), tgt_dict.eos(), tgt_dict.unk())
    num_sentences = 0
    has_target = True
    resdict = {}
    #hypo_strings = [] ##!!! 
    #src_strings = [] ##!!! 
    results_path = args.results_path
    stamp = str(time.time())
    resfp = results_path+"/"+args.gen_subset+"."+stamp+".gen_sparql.json"
    #resfp_ = results_path+"/"+args.gen_subset+"."+stamp+".txt"
    #sampfp = results_path+"/"+args.gen_subset+"."+stamp+".sampleids.txt"

    with progress_bar.build_progress_bar(args, itr) as t:
        wps_meter = TimeMeter()
        for sample in t:
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            if 'net_input' not in sample:
                continue

            prefix_tokens = None
            if args.prefix_size > 0:
                prefix_tokens = sample['target'][:, :args.prefix_size]

            gen_timer.start()
            hypos = task.inference_step(generator, models, sample, prefix_tokens)
            num_generated_tokens = sum(len(h[0]['tokens']) for h in hypos)
            gen_timer.stop(num_generated_tokens)

            #open(sampfp, "w", encoding="UTF-8").writeline(str(sample['id'].tolist())) 
            for i, sample_id in enumerate(sample['id'].tolist()):
                has_target = sample['target'] is not None

                # Remove padding
                src_tokens = utils.strip_pad(sample['net_input']['src_tokens'][i, :], tgt_dict.pad())
                target_tokens = None
                if has_target:
                    target_tokens = utils.strip_pad(sample['target'][i, :], tgt_dict.pad()).int().cpu()

                # Either retrieve the original sentences or regenerate them from tokens.
                if align_dict is not None:
                    src_str = task.dataset(args.gen_subset).src.get_original_text(sample_id)
                    target_str = task.dataset(args.gen_subset).tgt.get_original_text(sample_id)
                else:
                    if src_dict is not None:
                        src_str = src_dict.string(src_tokens, args.remove_bpe)
                    else:
                        src_str = ""
                    if has_target:
                        target_str = tgt_dict.string(target_tokens, args.remove_bpe, escape_unk=True)
                #src_strings.append(src_str) ##!!!
                if not args.quiet:
                    if src_dict is not None:
                        print('S-{}\t{}'.format(sample_id, src_str))
                    if has_target:
                        print('T-{}\t{}'.format(sample_id, target_str))

                # Process top predictions
                for i, hypo in enumerate(hypos[i][:min(len(hypos), args.nbest)]):
                    hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                        hypo_tokens=hypo['tokens'].int().cpu(),
                        src_str=src_str,
                        alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                        align_dict=align_dict,
                        tgt_dict=tgt_dict,
                        remove_bpe=args.remove_bpe,
                    )
                    #hypo_strings.append(hypo_str)  ##!!! 
                    resdict[str(int(sample_id)+1)] = {"sparql":interprete(hypo_str) , "en":src_str }   ##!!! 
                    if not args.quiet:
                        print('H-{}\t{}\t{}'.format(sample_id, hypo['score'], hypo_str))
                        print('P-{}\t{}'.format(
                            sample_id,
                            ' '.join(map(
                                lambda x: '{:.4f}'.format(x),
                                hypo['positional_scores'].tolist(),
                            ))
                        ))

                        if args.print_alignment:
                            print('A-{}\t{}'.format(
                                sample_id,
                                ' '.join(map(lambda x: str(utils.item(x)), alignment))
                            ))

                    # Score only the top hypothesis
                    if has_target and i == 0:
                        if align_dict is not None or args.remove_bpe is not None:
                            # Convert back to tokens for evaluation with unk replacement and/or without BPE
                            target_tokens = tgt_dict.encode_line(target_str, add_if_not_exist=True)
                        if hasattr(scorer, 'add_string'):
                            scorer.add_string(target_str, hypo_str)
                        else:
                            scorer.add(target_tokens, hypo_tokens)

            wps_meter.update(num_generated_tokens)
            t.log({'wps': round(wps_meter.avg)})
            num_sentences += sample['nsentences']

    print('| Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)'.format(
        num_sentences, gen_timer.n, gen_timer.sum, num_sentences / gen_timer.sum, 1. / gen_timer.avg))
    if has_target:
        print('| Generate {} with beam={}: {}'.format(args.gen_subset, args.beam, scorer.result_string()))
    """
    with open(resfp, "a", encoding="UTF-8") as restore :
        for gen_str in hypo_strings:
            restore.write(gen_str+" \n")
        restore.close() 
    with open(resfp_, "a", encoding="UTF-8") as res_tore :
        for src_str in src_strings:
            res_tore.write(src_str+" \n")
        res_tore.close()"""
    with open(resfp, "w", encoding="UTF-8") as restore :
        json.dump(resdict, restore, ensure_ascii=False, indent=4 )
        restore.close()

    return scorer 


def cli_main():
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    scorer = main(args)
    return scorer 


if __name__ == '__main__':
    cli_main()


#INSTRUCTIONS:
"""
usage: generate.py [-h] [--no-progress-bar] [--log-interval N]
                   [--log-format {json,none,simple,tqdm}]
                   [--tensorboard-logdir DIR] [--tbmf-wrapper] [--seed N]
                   [--cpu] [--fp16] [--memory-efficient-fp16]
                   [--fp16-init-scale FP16_INIT_SCALE]
                   [--fp16-scale-window FP16_SCALE_WINDOW]
                   [--fp16-scale-tolerance FP16_SCALE_TOLERANCE]
                   [--min-loss-scale D]
                   [--threshold-loss-scale THRESHOLD_LOSS_SCALE]
                   [--user-dir USER_DIR]
                   [--criterion {adaptive_loss,composite_loss,cross_entropy,label_smoothed_cross_entropy,masked_lm_loss}]
                   [--optimizer {adadelta,adafactor,adagrad,adam,lamb,nag,sgd}]
                   [--lr-scheduler {cosine,fixed,inverse_sqrt,polynomial_decay,reduce_lr_on_plateau,triangular}]
                   [--task TASK] [--num-workers N]
                   [--skip-invalid-size-inputs-valid-test] [--max-tokens N]
                   [--max-sentences N] [--required-batch-size-multiple N]
                   [--dataset-impl FORMAT] [--gen-subset SPLIT]
                   [--num-shards N] [--shard-id ID] [--path FILE]
                   [--remove-bpe [REMOVE_BPE]] [--quiet]
                   [--model-overrides DICT] [--results-path RESDIR] [--beam N]
                   [--nbest N] [--max-len-a N] [--max-len-b N] [--min-len N]
                   [--match-source-len] [--no-early-stop] [--unnormalized]
                   [--no-beamable-mm] [--lenpen LENPEN] [--unkpen UNKPEN]
                   [--replace-unk [REPLACE_UNK]] [--sacrebleu]
                   [--score-reference] [--prefix-size PS]
                   [--no-repeat-ngram-size N] [--sampling]
                   [--sampling-topk PS] [--temperature N]
                   [--diverse-beam-groups N] [--diverse-beam-strength N]
                   [--print-alignment] [--change-ratio]

optional arguments:
  -h, --help            show this help message and exit
  --no-progress-bar     disable progress bar
  --log-interval N      log progress every N batches (when progress bar is
                        disabled)
  --log-format {json,none,simple,tqdm}
                        log format to use
  --tensorboard-logdir DIR
                        path to save logs for tensorboard, should match
                        --logdir of running tensorboard (default: no
                        tensorboard logging)
  --tbmf-wrapper        [FB only]
  --seed N              pseudo random number generator seed
  --cpu                 use CPU instead of CUDA
  --fp16                use FP16
  --memory-efficient-fp16
                        use a memory-efficient version of FP16 training;
                        implies --fp16
  --fp16-init-scale FP16_INIT_SCALE
                        default FP16 loss scale
  --fp16-scale-window FP16_SCALE_WINDOW
                        number of updates before increasing loss scale
  --fp16-scale-tolerance FP16_SCALE_TOLERANCE
                        pct of updates that can overflow before decreasing the
                        loss scale
  --min-loss-scale D    minimum FP16 loss scale, after which training is
                        stopped
  --threshold-loss-scale THRESHOLD_LOSS_SCALE
                        threshold FP16 loss scale from below
  --user-dir USER_DIR   path to a python module containing custom extensions
                        (tasks and/or architectures)
  --criterion {adaptive_loss,composite_loss,cross_entropy,label_smoothed_cross_entropy,masked_lm_loss}
  --optimizer {adadelta,adafactor,adagrad,adam,lamb,nag,sgd}
  --lr-scheduler {cosine,fixed,inverse_sqrt,polynomial_decay,reduce_lr_on_plateau,triangular}
  --task TASK           task
  --dataset-impl FORMAT
                        output dataset implementation

Dataset and data loading:
  --num-workers N       how many subprocesses to use for data loading
  --skip-invalid-size-inputs-valid-test
                        ignore too long or too short lines in valid and test
                        set
  --max-tokens N        maximum number of tokens in a batch
  --max-sentences N, --batch-size N
                        maximum number of sentences in a batch
  --required-batch-size-multiple N
                        batch size will be a multiplier of this value
  --gen-subset SPLIT    data subset to generate (train, valid, test)
  --num-shards N        shard generation over N shards
  --shard-id ID         id of the shard to generate (id < num_shards)

Generation:
  --path FILE           path(s) to model file(s), colon separated
  --remove-bpe [REMOVE_BPE]
                        remove BPE tokens before scoring (can be set to
                        sentencepiece)
  --quiet               only print final scores
  --model-overrides DICT
                        a dictionary used to override model args at generation
                        that were used during model training
  --results-path RESDIR
                        path to save eval results (optional)"
  --beam N              beam size
  --nbest N             number of hypotheses to output
  --max-len-a N         generate sequences of maximum length ax + b, where x
                        is the source length
  --max-len-b N         generate sequences of maximum length ax + b, where x
                        is the source length
  --min-len N           minimum generation length
  --match-source-len    generations should match the source length
  --no-early-stop       continue searching even after finalizing k=beam
                        hypotheses; this is more correct, but increases
                        generation time by 50%
  --unnormalized        compare unnormalized hypothesis scores
  --no-beamable-mm      don't use BeamableMM in attention layers
  --lenpen LENPEN       length penalty: <1.0 favors shorter, >1.0 favors
                        longer sentences
  --unkpen UNKPEN       unknown word penalty: <0 produces more unks, >0
                        produces fewer
  --replace-unk [REPLACE_UNK]
                        perform unknown replacement (optionally with alignment
                        dictionary)
  --sacrebleu           score with sacrebleu
  --score-reference     just score the reference translation
  --prefix-size PS      initialize generation by target prefix of given length
  --no-repeat-ngram-size N
                        ngram blocking such that this size ngram cannot be
                        repeated in the generation
  --sampling            sample hypotheses instead of using beam search
  --sampling-topk PS    sample from top K likely next words instead of all
                        words
  --temperature N       temperature for generation
  --diverse-beam-groups N
                        number of groups for Diverse Beam Search
  --diverse-beam-strength N
                        strength of diversity penalty for Diverse Beam Search
  --print-alignment     if set, uses attention feedback to compute and print
                        alignment to source tokens
  --change-ratio

"""
