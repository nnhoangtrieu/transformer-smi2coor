import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
import math
import rdkit
from rdkit import Chem
import torch
plt.switch_backend('agg')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_test_split(x, y, ratio = 0.9) :
  ratio = int(len(x) * ratio)

  train_x = x[:ratio]
  train_y = y[:ratio]

  test_x = x[ratio:]
  test_y = y[ratio:]

  return train_x, train_y, test_x, test_y



def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)



def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))



def count_atoms(smi):
    # Parse the SMILES string
    mol = Chem.MolFromSmiles(smi)

    # Check if parsing was successful
    if mol is not None:
        # Count the number of atoms
        num_atoms = mol.GetNumAtoms()
        return num_atoms
    else:
        print("Error: Unable to parse SMILES string.")
        return None



    


def int2smi(input, invert_dic) :
    output = []
    input = input.cpu().numpy()
    
    for smiles in input :
        out = [invert_dic[atom] for atom in smiles]
        output.append(out)
    return output



def replace_duplicate_atom(smi) :
    smi = list(smi[:-1])
    smi = [smi.replace('Na', 'X')
                    .replace('Cl', 'Y')
                    .replace('Br', 'Z')
                    .replace('Ba', 'T') for smi in smi]

    return smi



def smi2int(smi, smi_dic, longest_smi) :
    smi = list(smi)
    smint = [smi_dic[atom] for atom in smi]
    smint = smint + [0] * (longest_smi - len(smint))
    smint_torch = torch.tensor(smint).view(1,-1)

    return smint_torch



def evaluate(encoder, decoder, smi, smi_dic, longest_smi) :
    encoder.eval()
    decoder.eval()
    
    
    smint = smi2int(smi, smi_dic, longest_smi)
    smint = smint.to(device)

    with torch.no_grad() :
        e_out, e_last, self_attn = encoder(smint)
        prediction, cross_attn = decoder(e_out, e_last)
    
    num_head = cross_attn.size(1)
    cross_attn, self_attn = cross_attn.squeeze(), self_attn.squeeze()
    cross_attn, self_attn = cross_attn.cpu().numpy(), self_attn.cpu().numpy()
    
    return prediction, cross_attn, self_attn, num_head



def plot_attn(matrix, smi, mode, path = "", name = "") :
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix, cmap = "viridis")
    fig.colorbar(cax)

    if mode == "cross" :
        ax.set_xticklabels([''] + smi)
    if mode == "self" :
        ax.set_xticklabels([''] + smi)
        ax.set_yticklabels([''] + smi)
    
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    
    plt.savefig(f"{path}/{name}.png")
    plt.show()
        



def visualize(encoder,
              decoder,
              smi,
              smi_dic,
              longest_smi,
              mode = "cross",
              path = "",
              name = 1) :
    
    prediction, cross_attn, self_attn, num_head = evaluate(encoder, decoder, smi, smi_dic, longest_smi)

    # attn, self_attn = attn.squeeze(), self_attn.squeeze()
    # attn, self_attn = attn.cpu().numpy(), self_attn.cpu().numpy()

    smi = replace_duplicate_atom(smi)

    coor_len = count_atoms(''.join(smi))
    smi_len = len(smi)

    if mode == "cross" :
        matrix = cross_attn[:coor_len, :smi_len]
    if mode == "self" :
        matrix = self_attn[:smi_len, :smi_len]
    
    if mode == "cross" :
        for i, head in enumerate(cross_attn) :
            matrix = head[:coor_len, :smi_len]
            plot_attn(matrix, smi, mode, path, f"H{i}-{name}")
    
    if mode == "self" :
        for i, head in enumerate(self_attn) :
            matrix = head[:smi_len, :smi_len]
            plot_attn(matrix, smi, mode, path, f"H{i}-{name}")

    # for i in range(1, num_head) :
    #     if mode == "cross" :
    #         matrix = matrix[i]
    #         matrix = cross_attn[:coor_len, :smi_len]
    #     if mode == "self" :
    #         matrix = matrix[i]
    #         matrix = self_attn[:smi_len, :smi_len]
    #     plot_attn(matrix[i], smi, mode, path, f"H{i}-{name}")
