import time
import math
from tqdm import tqdm #tqdm_notebook as tqdm
import numpy as np
import torch
from torch.utils import data
import torch.nn.functional as F
from transformer import Constants
from transformer.Generator import Generator
from math_dataset import (
    VOCAB_SZ,
    MAX_QUESTION_SZ,
    MAX_ANSWER_SZ,
    np_decode_string
)
from loss import compute_performance
from checkpoints import rotating_save_checkpoint, build_checkpoint
from math_dataset import np_encode_string, question_to_position_batch_collate_fn
from dgl_transformer.dataset.graph import graph_to_device


def train_epoch(
    model, training_data, optimizer, device, graph_pool, epoch, tb=None, log_interval=100
):
    model.train()
  
    total_loss = 0
    n_char_total = 0
    n_char_correct = 0

    for batch_idx, batch in enumerate(tqdm(training_data, mininterval=2, leave=False)):        
        #batch_qs, batch_as = batch
        #gold_as = torch.tensor(batch_as[:, 1:]).to(device)
        gold_as, g = batch
        
        gold_as = gold_as.to(device)
        g = graph_to_device(g, device)
        #g = graph_pool(batch_qs, batch_as, device=device)

        optimizer.zero_grad()

        pred_as = model(g)
        
        loss, n_correct = compute_performance(pred_as, gold_as, smoothing=True)    
        loss.backward()

        # update parameters
        optimizer.step()
    
        # note keeping
        total_loss += loss.item()

        non_pad_mask = gold_as.ne(Constants.PAD)
        n_char = non_pad_mask.sum().item()
        n_char_total += n_char
        n_char_correct += n_correct
    
        if tb is not None and batch_idx % log_interval == 0:
            tb.add_scalars(
                {
                    "loss_per_char" : total_loss / n_char_total,
                    "accuracy" : n_char_correct / n_char_total,
                },
                group="train",
                sub_group="batch",
                global_step=epoch * len(training_data) + batch_idx
            )

    loss_per_char = total_loss / n_char_total
    accuracy = n_char_correct / n_char_total

    if tb is not None:
        tb.add_scalars(
            {
                "loss_per_char" : loss_per_char,
                "accuracy" : accuracy,
            },
            group="train",
            sub_group="epoch",
            global_step=epoch
        )

    return loss_per_char, accuracy
  
  
def eval_epoch(model, validation_data, device, graph_pool, epoch, tb=None, log_interval=100):
    model.eval()

    total_loss = 0
    n_char_total = 0
    n_char_correct = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(validation_data, mininterval=2, leave=False)):
            # prepare data
            #batch_qs, batch_as = batch
            #gold_as = torch.tensor(batch_as[:, 1:]).to(device)

            #g = graph_pool(batch_qs, batch_as, device=device)
            
            gold_as, g = batch
            gold_as = gold_as.to(device)
            g = graph_to_device(g, device)        

            # forward
            pred_as = model(g)
            loss, n_correct = compute_performance(pred_as, gold_as, smoothing=False)

            # note keeping
            total_loss += loss.item()

            non_pad_mask = gold_as.ne(Constants.PAD)
            n_char = non_pad_mask.sum().item()
            n_char_total += n_char
            n_char_correct += n_correct

    loss_per_char = total_loss / n_char_total
    accuracy = n_char_correct / n_char_total
        
    if tb is not None:
        tb.add_scalars(
            {
                "loss_per_char" : loss_per_char,
                "accuracy" : accuracy,
            },
            group="eval",
            sub_group="epoch",
            global_step=epoch
        )
        
    return loss_per_char, accuracy
  

def interpolate_epoch(model, interpolate_data, device, graph_pool, epoch, tb=None, log_interval=100):
    model.eval()

    total_loss = 0
    n_char_total = 0
    n_char_correct = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(interpolate_data, mininterval=2, leave=False)):
            # prepare data
            #batch_qs, batch_as = batch
            #gold_as = torch.tensor(batch_as[:, 1:]).to(device)

            #g = graph_pool(batch_qs, batch_as, device=device)
            gold_as, g = batch
            gold_as = gold_as.to(device)
            g = graph_to_device(g, device) 

            # forward
            pred_as = model(g)
            loss, n_correct = compute_performance(pred_as, gold_as, smoothing=False)

            # note keeping
            total_loss += loss.item()

            non_pad_mask = gold_as.ne(Constants.PAD)
            n_char = non_pad_mask.sum().item()
            n_char_total += n_char
            n_char_correct += n_correct

    loss_per_char = total_loss / n_char_total
    accuracy = n_char_correct / n_char_total

    if tb is not None:
        tb.add_scalars(
            {
                "loss_per_char" : loss_per_char,
                "accuracy" : accuracy,
            },
            group="interpolate",
            sub_group="epoch",
            global_step=epoch
        )
        
    return loss_per_char, accuracy
      
  
def train(exp_name, unique_id,
          model, training_data, validation_data, interpolate_data, optimizer, device, graph_pool, epochs,
          interpolate_loader=None, tb=None, log_interval=100, interpolate_interval=1,
          start_epoch=0, best_valid_accu=0.0, best_valid_loss=float('Inf'),
          best_interpolate_accu=0.0, best_interpolate_loss = float('Inf')):    
    
  for epoch_i in range(start_epoch, epochs):
    start = time.time()
    valid_loss, valid_accu = eval_epoch(model, validation_data, device, graph_pool, epoch_i, tb, log_interval)
    print('[Validation]  loss: {valid_loss},  ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
            'elapse: {elapse:3.3f}ms'.format(
                valid_loss=valid_loss, ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu,
                elapse=(time.time()-start)*1000))

    if valid_accu > best_valid_accu:
        print("Checkpointing Validation Model...")
        best_valid_accu = valid_accu
        best_valid_loss = valid_loss
        state = build_checkpoint(exp_name, unique_id, "validation", model, optimizer, best_valid_accu, best_valid_loss, epoch_i)
        rotating_save_checkpoint(state, prefix=f"{exp_name}_{unique_id}_validation", path="./checkpoints", nb=5)    
    
    if epoch_i % interpolate_interval == 0:
        start = time.time()
        interpolate_loss, interpolate_accu = interpolate_epoch(model, interpolate_data, device, graph_pool, epoch_i, tb, log_interval)
        print('[Interpolate]  loss: {interpolate_loss},  ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
                'elapse: {elapse:3.3f}ms'.format(
                    interpolate_loss=interpolate_loss, ppl=math.exp(min(interpolate_loss, 100)), accu=100*interpolate_accu,
                    elapse=(time.time()-start)*1000))

        if interpolate_accu > best_interpolate_accu:
            print("Checkpointing Interpolate Model...")
            best_interpolate_accu = interpolate_accu
            best_interpolate_loss = interpolate_loss
            state = build_checkpoint(
                exp_name, unique_id, "interpolate", model, optimizer, best_interpolate_accu, best_interpolate_loss, epoch_i)

            rotating_save_checkpoint(state, prefix=f"{exp_name}_{unique_id}_interpolate", path="./checkpoints", nb=5)    
    print('[ Epoch', epoch_i, ']')

    start = time.time()
    train_loss, train_accu = train_epoch(
        model, training_data, optimizer, device, graph_pool, epoch_i, tb, log_interval)
    print('[Training]  loss: {train_loss}, ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
          'elapse: {elapse:3.3f}ms'.format(
              train_loss=train_loss, ppl=math.exp(min(train_loss, 100)), accu=100*train_accu,
              elapse=(time.time()-start)*1000))

    
def predict_multiple(questions, model, device, graph_collate, beam_size=5,
                     max_token_seq_len=MAX_QUESTION_SZ, max_answer_seq_len=MAX_ANSWER_SZ, n_best=1,
                     alpha=0.6):
    model.eval()

    qs = list(map(lambda q: np_encode_string(q), questions))
    g = graph_collate.beam(qs, device, max_answer_seq_len, Constants.BOS, beam_size)
    g = graph_to_device(g, device)
    
    answers = model.infer(g, max_answer_seq_len, Constants.EOS, beam_size, alpha=alpha)
    res = []
    for answer in answers:
        answer = np.array(answer)
        first_eos = np.where(answer == Constants.EOS)[0][0]
        # removes first char as it is BOS and goes till EOS
        res.append(np_decode_string(answer[1:first_eos+1]))
    return res

    
    
def predict_single(question, model, device, graph_collate, beam_size=5,
                   max_token_seq_len=MAX_QUESTION_SZ, max_answer_seq_len=MAX_ANSWER_SZ, n_best=1, alpha=0.6):
    model.eval()
    
    qs = [np_encode_string(question)]
    g = graph_collate.beam(qs, device, max_answer_seq_len, Constants.BOS, beam_size)
    g = graph_to_device(g, device)
    
    answers = model.infer(g, max_answer_seq_len, Constants.EOS, beam_size, alpha=alpha)
    res = []
    for answer in answers:
        answer = np.array(answer)
        first_eos = np.where(answer == Constants.EOS)[0][0]
        # removes first char as it is BOS and goes till EOS
        res.append(np_decode_string(answer[1:first_eos+1]))
    return res
