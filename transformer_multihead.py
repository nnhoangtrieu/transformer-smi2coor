# DIM_MODEL : Dimension (hidden size) of the model. 
#           : It should be 128, 256, 512 (I find 512 too big), and it should also be divisible by NUM_HEAD


# NUM_BLOCK : Number of encoder and decoder block connected. My experiments have shown that increasing this might
#             make the model too complex which does not lead to better result. 

# NUM_HEAD : Number of head for attention
#            


# DROPOUT : Dropout rate of nn.Dropout() layer


# FORWARD_EXTENSION : The scalar to scale up the network (Apply for only some layers, DIM_MODEL * FORWARD_EXTENSION)


# N_EPOCHS : Number of epochs


# TEACHER_FORCING_RATE : For example, if set to 0.4, first 40% epoch will be trained using teacher forcing. 
#                      + Teacher forcing: Using known target in training data to predict the next state of RNN.
#                      + NO Teacher Forcing: Using prediction as the input for predicting the next state


# ATTENTION_IMAGE_OUTPUT_PATH : Name of folder to output attention image during training. I chose 3 random SMILES to output for each train.
#                             : I have already created a folder name "attention image
#                             : The image is in the format H-R-CROSS-E. 
#                             : H1 means the attention of head number 1
#                             : R1 means the random SMILES number 1
#                             : CROSS or SELF means image of Cross Attention or Self Attention
#                             : E1 means the image of epoch number 1



# ---------------------------HYPERPARAMETER-------------------------------------- #
DIM_MODEL = 256

NUM_ENCODER_BLOCK = 1
NUM_SELF_ATTENTION_HEAD = 2

NUM_DECODER_BLOCK = 1
NUM_CROSS_ATTENTION_HEAD = 1

DROPOUT = 0.5
FORWARD_EXTENSION = 1

N_EPOCHS = 30
LEARNING_RATE = 0.001
TEACHER_FORCING_RATE = 0.0
ATTENTION_IMAGE_OUTPUT_PATH = 'attention image'
# ------------------------------------------------------------------------------- #



























import torch 
import torch.nn as nn 
import torch.nn.functional as F
import time
import random
from utils.data_preprocess import train_loader, test_loader, smi_list, smi_dic, smint_list, coor_list, np_coor_list, longest_coor, longest_smi, device
from utils.helper import visualize, timeSince, visualize2


class NN_Attention(nn.Module) :
    def __init__(self, dim_model, num_head) :
        super(NN_Attention, self).__init__()
        self.dim_model = dim_model
        self.num_head = num_head
        self.dim_head = dim_model // num_head

        self.W = nn.Linear(dim_model, dim_model)
        self.U = nn.Linear(dim_model, dim_model)
        self.V = nn.Linear(dim_model // num_head, 1)

    def forward(self, Q, K) :
        B, longest_smi, dim_model = K.size()

        W = self.W(Q) 
        U = self.U(K)
        len_W, len_U = W.size(1), U.size(1)
        
        
        W = W.reshape(B, self.num_head, len_W, self.dim_head)
        U = U.reshape(B, self.num_head, len_U, self.dim_head)

        attn_score = self.V(torch.tanh(W + U))

        attn_score = attn_score.squeeze(-1).unsqueeze(2)

        attn_distribution = F.softmax(attn_score, dim = -1)

        K = K.reshape(B, self.num_head, longest_smi, self.dim_head)

        attn = attn_distribution @ K

        attn = attn.reshape(B, 1, self.num_head * self.dim_head)

        return attn, attn_distribution
    

class DP_Attention(nn.Module) : # Dot Product Attention
    def __init__(self, dim_model, num_head) :
        super(DP_Attention, self).__init__()
        self.dim_model = dim_model
        self.num_head = num_head
        self.dim_head = dim_model // num_head

        self.Q = nn.Linear(dim_model, dim_model)
        self.K = nn.Linear(dim_model, dim_model)
        self.V = nn.Linear(dim_model, dim_model)

        self.out = nn.Linear(dim_model, dim_model)

    def forward(self, Q, K, V) :
        B = Q.size(0) # Shape Q, K, V: (B, longest_smi, dim_model)

        Q, K, V = self.Q(Q), self.K(K), self.V(V)

        len_Q, len_K, len_V = Q.size(1), K.size(1), V.size(1)

        Q = Q.reshape(B, self.num_head, len_Q, self.dim_head)
        K = K.reshape(B, self.num_head, len_K, self.dim_head)
        V = V.reshape(B, self.num_head, len_V, self.dim_head)
        
        K_T = K.transpose(2,3).contiguous()

        attn_score = Q @ K_T

        attn_score = attn_score / (self.dim_head ** 1/2)

        attn_distribution = torch.softmax(attn_score, dim = -1)

        attn = attn_distribution @ V

        attn = attn.reshape(B, len_Q, self.num_head * self.dim_head)
        
        attn = self.out(attn)

        return attn, attn_distribution


class EncoderBlock(nn.Module) :
    def __init__(self, dim_model, num_head, fe, dropout) :
        super(EncoderBlock, self).__init__()

        self.SelfAttention = DP_Attention(dim_model,num_head)
        self.LSTM = nn.LSTM(input_size=2 * dim_model, hidden_size=dim_model, batch_first=True)

    def forward(self, Q, K, V) :

        attn, self_attn = self.SelfAttention(Q, Q, Q)

        input_lstm = torch.cat((Q, attn), dim = -1)

        all_state, (last_state, _) = self.LSTM(input_lstm)

        return all_state, last_state, self_attn


class Encoder(nn.Module) :
    def __init__(self, dim_model, num_block, num_head, len_dic, fe = 1, dropout = 0.1) :
        super(Encoder, self).__init__()

        self.dim_model = dim_model
        self.Embedding = nn.Embedding(len_dic, dim_model)
        self.Dropout = nn.Dropout(dropout)
        self.EncoderBlocks = nn.ModuleList(
            EncoderBlock(dim_model, num_head, fe, dropout) for _ in range(num_block)
        )

    def forward(self, x) :
        out = self.Dropout(self.Embedding(x))

        for block in self.EncoderBlocks : 
            out, last_state, self_attn = block(out, out, out) 
        return out, last_state, self_attn


class GRU(nn.Module) :
    def __init__(self, dim_model, longest_coor, dropout, num_head = 1, output_size = 3) :
        super(GRU, self).__init__()

        self.longest_coor = longest_coor

        self.CrossAttention = NN_Attention(dim_model, num_head)

        self.GRU = nn.GRU(3 + dim_model, dim_model, batch_first=True)

        self.Linear = nn.Linear(dim_model, output_size)

        self.Dropout = nn.Dropout(dropout)

    def forward(self, e_all, e_last, target = None) :
        B = e_all.size(0)

        d_input = torch.zeros(B, 1, 3).to(device)

        d_hidden = e_last

        d_outputs, cross_attn = [], []

        for i in range(self.longest_coor) :
            d_output, d_hidden, step_attn = self.forward_step(d_input, d_hidden, e_all)

            d_outputs.append(d_output), cross_attn.append(step_attn)

            if target is not None :
                d_input = target[:, i, :].unsqueeze(1)
            else :
                d_input = d_output

        d_outputs = torch.cat(d_outputs, dim = 1)

        cross_attn = torch.cat(cross_attn, dim = 2)
        
        return d_outputs, d_hidden, cross_attn


    def forward_step(self, d_input, d_hidden, e_all) :
        Q = d_hidden.permute(1,0,2)

        d_input = self.Dropout(d_input)

        attn, attn_distribution = self.CrossAttention(Q, e_all)

        input_lstm = torch.cat((attn, d_input), dim = 2)

        output, d_hidden = self.GRU(input_lstm, d_hidden) 
        
        output = self.Linear(output)

        return output, d_hidden, attn_distribution


class DecoderBlock(nn.Module) :
    def __init__(self, dim_model, num_head, longest_coor, fe, dropout) :
        super(DecoderBlock, self).__init__()

        self.GRU = GRU(dim_model, longest_coor,dropout, num_head)

    def forward(self, e_all, e_last, target = None) :
        output, _, cross_attn = self.GRU(e_all, e_last, target)

        return output, cross_attn


class Decoder(nn.Module) :
    def __init__(self, dim_model,num_block, num_head, longest_coor, fe = 1, dropout = 0.1) :
        super(Decoder, self).__init__()

        self.DecoderBlocks = nn.ModuleList(
            [DecoderBlock(dim_model, num_head,longest_coor, fe, dropout) for _ in range(num_block)]
        )

        self.dropout = nn.Dropout(dropout)

        
    def forward(self, e_all, e_last, target = None) :
        for block in self.DecoderBlocks :
            target, cross_attn = block(e_all, e_last, target)
        
        return target, cross_attn


r1 = random.randint(1, len(smi_list))
r2 = random.randint(1, len(smi_list))
r3 = random.randint(1, len(smi_list))


def train_epoch(train_loader,test_loader, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion, tf):

    total_loss = 0
    total_test_loss = 0

    for input, target in train_loader:
        input, target = input.to(device), target.to(device)

        encoder_optimizer.zero_grad(), decoder_optimizer.zero_grad()
        
        e_all, e_last, self_attn = encoder(input)

        # Teacher Forcing
        if tf :
          prediction, cross_attn = decoder(e_all, e_last, target)
        else :
          prediction, cross_attn = decoder(e_all, e_last)


        loss = criterion(prediction, target)
        loss.backward()

        encoder_optimizer.step(), decoder_optimizer.step()
        
        total_loss += loss.item()


    encoder.eval(), decoder.eval()
    


    with torch.no_grad() :
      for input, target in test_loader :
        input, target = input.to(device), target.to(device)
        
        e_all, e_last, self_attn = encoder(input)
        prediction, cross_attn = decoder(e_all, e_last)

        test_loss = criterion(prediction, target)
        total_test_loss += test_loss.item()

    return total_loss / len(train_loader), total_test_loss / len(test_loader)


def train(train_loader, test_loader, encoder, decoder, n_epochs, learning_rate=0.001,
               print_every=1, visual_path= "", tf_rate = 1):
    start = time.time()

    train_loss_total = 0  
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

        visualize(encoder, decoder, smi_list[r1], smi_dic, longest_smi, mode="cross", path=f"{visual_path}", name=f"R1-CROSS-E{epoch}")
        visualize(encoder, decoder, smi_list[r2], smi_dic, longest_smi, mode="cross", path=f"{visual_path}", name=f"R2-CROSS-E{epoch}")
        visualize(encoder, decoder, smi_list[r3], smi_dic, longest_smi, mode="cross", path=f"{visual_path}", name=f"R3-CROSS-E{epoch}")
        visualize(encoder, decoder, smi_list[r1], smi_dic, longest_smi, mode="self", path=f"{visual_path}", name=f"R1-SELF-E{epoch}")
        visualize(encoder, decoder, smi_list[r2], smi_dic, longest_smi, mode="self", path=f"{visual_path}", name=f"R2-SELF-E{epoch}")
        visualize(encoder, decoder, smi_list[r3], smi_dic, longest_smi, mode="self", path=f"{visual_path}", name=f"R3-SELF-E{epoch}")

        if epoch % print_every == 0:
            train_loss_avg = train_loss_total / print_every
            test_loss_avg = test_loss_total / print_every
            train_loss_total = 0
            test_loss_total = 0
            print('%s (%d %d%%) /// Train loss: %.4f - Test loss: %.4f' % (timeSince(start, epoch / n_epochs),
                                        epoch, epoch / n_epochs * 100, train_loss_avg, test_loss_avg))







encoder = Encoder(dim_model=DIM_MODEL,
                  num_block=NUM_ENCODER_BLOCK,
                  num_head=NUM_SELF_ATTENTION_HEAD,
                  dropout=DROPOUT,
                  fe = FORWARD_EXTENSION,
                  len_dic=len(smi_dic)).to(device)

decoder = Decoder(dim_model=DIM_MODEL,
                  num_block=NUM_DECODER_BLOCK,
                  num_head=NUM_CROSS_ATTENTION_HEAD,
                  dropout=DROPOUT,
                  fe=FORWARD_EXTENSION,
                  longest_coor=longest_coor,
                  ).to(device)


train(train_loader, test_loader, encoder, decoder,
      n_epochs=N_EPOCHS,
      learning_rate=LEARNING_RATE,
      visual_path=ATTENTION_IMAGE_OUTPUT_PATH,
      tf_rate=TEACHER_FORCING_RATE)


