import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import rdkit
import matplotlib.pyplot as plt
import time
import math
import matplotlib.ticker as ticker
from helper import train_test_split, timeSince, visualize
from data_preprocess import Data, TorchDataset, normalize_coor, pad_coor
plt.switch_backend('agg')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# GET DATA
smint_list, coor_list, smi_list, smi_dic, longest_smi, longest_coor = Data(smi_path='./data/smi_data.csv',
                                                                           coor_path='./data/coor_data.sdf').extract()
p_coor_list = pad_coor(coor_list, longest_coor)
np_coor_list = pad_coor(normalize_coor(coor_list), longest_coor)
train_x, train_y, test_x, test_y = train_test_split(x = smint_list, y = np_coor_list, ratio=0.95)
B = 16
train_set = TorchDataset(train_x, train_y)
test_set = TorchDataset(test_x, test_y)
train_loader = DataLoader(train_set, batch_size = B, shuffle = True)
test_loader = DataLoader(test_set, batch_size = B)

def train_epoch(train_loader,test_loader, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion, tf):

    total_loss = 0
    total_test_loss = 0

    for data in train_loader:

        input_tensor, target_tensor = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden, self_attn = encoder(input_tensor)

        # Teacher Forcing 
        if tf :
          decoder_outputs, _, cross_attn = decoder(encoder_outputs, encoder_hidden, target_tensor)
        else :
          decoder_outputs, _, cross_attn = decoder(encoder_outputs, encoder_hidden)


        loss = criterion(decoder_outputs, target_tensor)
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()


    encoder.eval() 
    decoder.eval()


    with torch.no_grad() :
      for test_inputs, test_labels in test_loader :
        test_inputs = test_inputs.to(device)
        test_labels = test_labels.to(device)

        encoder_outputs, encoder_hidden, _ = encoder(test_inputs)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden)

        test_loss = criterion(decoder_outputs, test_labels)
        total_test_loss += test_loss.item()

    return total_loss / len(train_loader), total_test_loss / len(test_loader)



def train(train_loader, test_loader, encoder, decoder, n_epochs, learning_rate=0.001,
               print_every=100, plot_every=100, tf_rate = 1):
    start = time.time()
    train_loss_total = 0  # Reset every print_every
    test_loss_total = 0

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

    criterion = nn.L1Loss()

    tf = True

    for epoch in range(1, n_epochs + 1):
      if epoch > (tf_rate * n_epochs) :
        tf = False
      encoder.train()
      decoder.train()

      train_loss, test_loss = train_epoch(train_loader, test_loader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, tf)
      train_loss_total += train_loss
      test_loss_total += test_loss

      for i in range(5) :
         visualize(encoder, decoder, smi_list[i], smi_dic, longest_smi, mode="cross", path="../attention process/", name=f"{i}-cross-E{epoch}")
         visualize(encoder, decoder, smi_list[i], smi_dic, longest_smi, mode="self", path="../attention process/", name=f"{i}-self-E{epoch}")

      if epoch % print_every == 0:
          train_loss_avg = train_loss_total / print_every
          test_loss_avg = test_loss_total / print_every
          train_loss_total = 0
          test_loss_total = 0
          print('%s (%d %d%%) /// Train loss: %.4f - Test loss: %.4f' % (timeSince(start, epoch / n_epochs),
                                      epoch, epoch / n_epochs * 100, train_loss_avg, test_loss_avg))
