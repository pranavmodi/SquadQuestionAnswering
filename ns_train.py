"""Train Bert model on SQuAD 2.0

Pre-processing code adapted from:
    > https://github.com/chrischute/squad

Author:
    Pranav Modi
"""
from ns_args import get_train_args
from json import dumps
import random
import numpy as np

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data
import torch.nn.functional as F

from ujson import load as json_load
from collections import OrderedDict
from tqdm import tqdm

from transformers import BertModel, BertForQuestionAnswering
from tensorboardX import SummaryWriter
import ns_utils as util


def main(args):
    # Set up logging and devices
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=True)
    log = util.get_logger(args.save_dir, name)
    tbx = SummaryWriter(args.save_dir)
    device, args.gpu_ids = util.get_available_devices()

    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    args.batch_size *= max(1, len(args.gpu_ids))
    print('the batch size', args.batch_size)

    # Set random seed
    log.info(f'Using random seed {args.seed}...')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Get model
    log.info('Building model...')
    model = BertForQuestionAnswering.from_pretrained('bert-base-cased')

    if args.load_path:
        log.info(f'Loading checkpoint from {args.load_path}...')
        model, step = util.load_model(model, args.load_path, args.gpu_ids)
    else:
        step = 0

    model = model.to(device)
    model.train()

    # Get saver
    saver = util.CheckpointSaver(args.save_dir,
                                 max_checkpoints=args.max_checkpoints,
                                 metric_name=args.metric_name,
                                 maximize_metric=args.maximize_metric,
                                 log=log)

    # Get optimizer and scheduler
    optimizer = optim.Adadelta(model.parameters(), args.lr,
                               weight_decay=args.l2_wd)
    scheduler = sched.LambdaLR(optimizer, lambda s: 1.)  # Constant LR

    # optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    

    # Get data loader
    log.info('Building dataset...')
    train_dataset = util.SQuAD(args.train_record_file, args.batch_size, args.use_squad_v2)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers)
    
    dev_dataset = util.SQuAD(args.dev_record_file, args.use_squad_v2)
    dev_loader = data.DataLoader(dev_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers)

    # Train
    log.info('Training...')
    steps_till_eval = args.eval_steps
    # epoch = step // len(train_dataset)
    epoch = 0
    step = 0
    while epoch != args.num_epochs:
        epoch += 1
        log.info(f'Starting epoch {epoch}...')
        print('going to evaluate right away....')
        results, pred_dict = evaluate(model, dev_loader, device,
                                      args.dev_eval_file,
                                      args.max_ans_len,
                                      args.use_squad_v2)
        
        
        with torch.enable_grad(), \
                tqdm(total=len(train_loader.dataset)) as progress_bar:
            for qas_indices, input_mask, y1, y2, id in train_loader:
                model.train()
                batch_size = qas_indices.size()[0]

                qas_indices = qas_indices.to(device)
                input_mask = input_mask.to(device)
                y1 = y1.to(device)
                y2 = y2.to(device)

                # outputs = model(qas_indices, attention_mask=input_mask,
                #                 start_positions=y1, end_positions=y2)

                outputs = model(qas_indices, start_positions=y1, end_positions=y2)
                loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

                loss.backward()
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()

                steps_till_eval -= batch_size
                print(f'Steps till eval {steps_till_eval}')
                step += batch_size

                if steps_till_eval <= 0:
                    steps_till_eval = args.eval_steps

                    # Evaluate and save checkpoint
                    #log.info(f'Evaluating at step {step}...')
                    #ema.assign(model)
                    results, pred_dict = evaluate(model, dev_loader, device,
                                                  args.dev_eval_file,
                                                  args.max_ans_len,
                                                  args.use_squad_v2)

                    print(results)
                    saver.save(step, model, results[args.metric_name], device)




def evaluate(model, data_loader, device, eval_file, max_len, use_squad_v2=True):
    nll_meter = util.AverageMeter()

    model.eval()
    pred_dict = {}
    with open(eval_file, 'r') as fh:
        gold_dict = json_load(fh)

    with torch.no_grad(), \
            tqdm(total=len(data_loader.dataset)) as progress_bar:
        for qas_indices, input_mask, y1, y2, ids in data_loader:

            qas_indices = qas_indices.to(device)
            y1 = y1.to(device)
            y2 = y2.to(device)

            batch_size = qas_indices.size(0)

            logits = model(qas_indices)
            start_logits, end_logits = logits

            start_probs = F.softmax(start_logits)
            end_probs = F.softmax(end_logits)

            loss = F.nll_loss(start_probs, y1) + F.nll_loss(end_probs, y2)
            loss_val = loss.item()

            nll_meter.update(loss_val, batch_size)

            # Get F1 and EM scores

            context_lens = [gold_dict[str(id)]['context_length'] for id in ids.tolist()]
            starts, ends = util.discretize(start_probs, end_probs, context_lens, max_len, use_squad_v2)

            # Log info
            progress_bar.update(batch_size)
            progress_bar.set_postfix(NLL=nll_meter.avg)
            progress_bar.set_postfix(NLL=nll_meter.avg)

            preds = util.convert_tokens(gold_dict,
                                           ids.tolist(),
                                           starts.tolist(),
                                           ends.tolist(),
                                           use_squad_v2)
            pred_dict.update(preds)
            break

    model.train()

    results = util.eval_dicts(gold_dict, pred_dict, use_squad_v2)
    results_list = [('NLL', nll_meter.avg),
                    ('F1', results['F1']),
                    ('EM', results['EM'])]
    if use_squad_v2:
        results_list.append(('AvNA', results['AvNA']))
    results = OrderedDict(results_list)

    return results, pred_dict


if __name__ == '__main__':
    name = 'somename'
    main(get_train_args())
