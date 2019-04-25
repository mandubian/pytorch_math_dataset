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


def train_epoch(model, training_data, optimizer, device, epoch, tb=None, log_interval=100):
    model.train()
  
    total_loss = 0
    n_char_total = 0
    n_char_correct = 0

    for batch_idx, batch in enumerate(tqdm(training_data, mininterval=2, leave=False)):
        batch_qs, batch_qs_pos, batch_as, batch_as_pos = map(lambda x: x.to(device), batch)
        gold_as = batch_as[:, 1:]

        optimizer.zero_grad()

        pred_as = model(batch_qs, batch_qs_pos, batch_as, batch_as_pos)

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
  
  
def eval_epoch(model, validation_data, device, epoch, tb=None, log_interval=100):
    model.eval()

    total_loss = 0
    n_char_total = 0
    n_char_correct = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(validation_data, mininterval=2, leave=False)):
            # prepare data
            batch_qs, batch_qs_pos, batch_as, batch_as_pos = map(lambda x: x.to(device), batch)
            gold_as = batch_as[:, 1:]

            # forward
            pred_as = model(batch_qs, batch_qs_pos, batch_as, batch_as_pos)
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
  

def interpolate_epoch(model, interpolate_data, device, epoch, tb=None, log_interval=100):
    model.eval()

    total_loss = 0
    n_char_total = 0
    n_char_correct = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(interpolate_data, mininterval=2, leave=False)):
            # prepare data
            batch_qs, batch_qs_pos, batch_as, batch_as_pos = map(lambda x: x.to(device), batch)
            gold_as = batch_as[:, 1:]

            # forward
            pred_as = model(batch_qs, batch_qs_pos, batch_as, batch_as_pos)
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
          model, training_data, validation_data, interpolate_data, optimizer, device, epochs,
          interpolate_loader=None, tb=None, log_interval=100, interpolate_interval=1,
          start_epoch=0, best_valid_accu=0.0, best_valid_loss=float('Inf'),
          best_interpolate_accu=0.0, best_interpolate_loss = float('Inf')):    
    
  for epoch_i in range(start_epoch, epochs):
    print('[ Epoch', epoch_i, ']')

    start = time.time()
    train_loss, train_accu = train_epoch(model, training_data, optimizer, device, epoch_i, tb, log_interval)
    print('[Training]  loss: {train_loss}, ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
          'elapse: {elapse:3.3f}ms'.format(
              train_loss=train_loss, ppl=math.exp(min(train_loss, 100)), accu=100*train_accu,
              elapse=(time.time()-start)*1000))

    start = time.time()
    valid_loss, valid_accu = eval_epoch(model, validation_data, device, epoch_i, tb, log_interval)
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
        interpolate_loss, interpolate_accu = interpolate_epoch(model, interpolate_data, device, epoch_i, tb, log_interval)
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

def predict(generator, data, device, max_predictions=None):
    if max_predictions is not None:
        cur = max_predictions
    else:
        cur = len(data)
            
    resps = []
    for batch_idx, batch in enumerate(data):
        if cur == 0:
            break
        
        batch_qs, batch_qs_pos = map(lambda x: x.to(device), batch)
        all_hyp, all_scores = generator.generate_batch(batch_qs, batch_qs_pos)
        
        for i, idx_seqs in enumerate(all_hyp):
            for j, idx_seq in enumerate(idx_seqs):
                r = np_decode_string(np.array(idx_seq))
                s = all_scores[i][j].cpu().item()
                resps.append({"resp":r, "score":s})
        cur -= 1
                    
    return resps
                

def predict_dataset(dataset, model, device, callback, max_batches=None,
                    beam_size=5, max_token_seq_len=MAX_QUESTION_SZ, n_best=1,
                    batch_size=1, num_workers=1):
    
    generator = Generator(model, device, beam_size=beam_size,
                          max_token_seq_len=max_token_seq_len, n_best=n_best)

    if max_batches is not None:
        cur = max_batches
    else:
        cur = len(dataset)
            
    resps = []
    for batch_idx, batch in enumerate(dataset):
        if cur == 0:
            break
        
        batch_qs, batch_qs_pos, _, _ = map(lambda x: x.to(device), batch)
        all_hyp, all_scores = generator.generate_batch(batch_qs, batch_qs_pos)
        
        callback(batch_idx, all_hyp, all_scores)
        
        cur -= 1
    return resps
    
    
def predict_multiple(questions, model, device, beam_size=5,
                     max_token_seq_len=MAX_QUESTION_SZ, n_best=1, batch_size=1,
                     num_workers=1):

    questions = list(map(lambda q: np_encode_string(q), questions))
    questions = data.DataLoader(questions, batch_size=1, shuffle=False, num_workers=1, collate_fn=question_to_position_batch_collate_fn)
    
    generator = Generator(model, device, beam_size=beam_size, max_token_seq_len=max_token_seq_len, n_best=n_best)
        
    return predict(generator, questions, device)
    
    
def predict_single(question, model, device, beam_size=5,
                   max_token_seq_len=MAX_QUESTION_SZ, n_best=1):
    
    generator = Generator(model, device, beam_size=beam_size,
                          max_token_seq_len=max_token_seq_len, n_best=n_best)
    
    qs = [np_encode_string(question)]
    qs, qs_pos = question_to_position_batch_collate_fn(qs)
    qs, qs_pos = qs.to(device), qs_pos.to(device)
    
    all_hyp, all_scores = generator.generate_batch(qs, qs_pos)
    resp = np_decode_string(np.array(all_hyp[0][0]))
    
    resps = []
    for i, idx_seqs in enumerate(all_hyp):
        for j, idx_seq in enumerate(idx_seqs):
            r = np_decode_string(np.array(idx_seq))
            s = all_scores[i][j].cpu().item()
            resps.append({"resp":r, "score":s})
    
    return resps
