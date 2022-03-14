import os, math
# import extract_bvh_array
import numpy
from scipy.fftpack import dct
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class MLP(torch.nn.Module):
    def __init__(self, input_size, output_size, num_hidden_layer):
        super().__init__()
        hidden_size = input_size * 2
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        #layers.append(nn.ReLU(inplace=True))
        layers.append(nn.ELU(inplace=True))
        for _ in range(num_hidden_layer):
            layers.append(nn.Linear(hidden_size, hidden_size))
            #layers.append(nn.ReLU(inplace=True))
            layers.append(nn.ELU(inplace=True))
        layers.append(nn.Linear(hidden_size, output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def weighted_mse_loss(input, target, weight):
    return torch.sum((weight * (input - target)) ** 2)

def copy_net_weights(
    net,
    net_pre):

    n0 = net_pre.layers[0].weight.shape[1]
    n1 = net_pre.layers[0].weight.shape[0]
    with torch.no_grad():
        net.layers[0].weight *= 0.01
        net.layers[0].bias *= 0.01
        net.layers[0].weight[:n1,:n0] = net_pre.layers[0].weight[:n1,:n0]
        net.layers[0].bias[:n1] = net_pre.layers[0].bias[:n1]

    if len(net.layers) < 3:
        return

    n2 = net_pre.layers[2].weight.shape[0]
    with torch.no_grad():
        net.layers[2].weight *= 0.01
        net.layers[2].bias *= 0.01
        net.layers[2].weight[:n2,:n1] = net_pre.layers[2].weight[:n2,:n1]
        net.layers[2].bias[:n2] = net_pre.layers[2].bias[:n2]

    if len(net.layers) < 5:
        return

    n3 = net_pre.layers[4].weight.shape[0]
    with torch.no_grad():
        net.layers[4].weight *= 0.01
        net.layers[4].bias *= 0.01
        net.layers[4].weight[:n3,:n2] = net_pre.layers[4].weight[:n3,:n2]
        net.layers[4].bias[:n3] = net_pre.layers[4].bias[:n3]


def model_input(cycles, nframe):
    tmp_list = []
    tmp_list.append( torch.linspace(0.0, 1.0, nframe, dtype=torch.float32) )
    for cycle in cycles:
        tmp_list.append( torch.sin(2.*math.pi*cycle*tmp_list[0]) )
        tmp_list.append( torch.cos(2.*math.pi*cycle*tmp_list[0]) )
    return torch.stack(tmp_list,dim=1)  # pt_in.reshape([*pt_in.shape,1])


def new_cycle(np_diff, np_weights):
    deviation = np_weights * (np_diff - np_diff.mean(axis=0))
    eig_val, eig_vec = numpy.linalg.eig(deviation.transpose() @ deviation)
    eig_vec = eig_vec.real
    history0 = deviation @ eig_vec[:,0]
    new_cycle = numpy.abs(dct(history0)).argmax() * 0.5
    new_cycle += 0.25 * numpy.random.randn(1)[0]
    return new_cycle

def compress(
        net,
        cycles,
        np_trg:numpy.ndarray,
        np_weights: numpy.ndarray):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pt_trg = torch.from_numpy(np_trg).float().to(device)
    pt_weight = torch.from_numpy(np_weights).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-5)

    pt_in = model_input(cycles, np_trg.shape[0]).to(device)
    # print("   shape of mlp input: ",pt_in.shape)
    loader = DataLoader(TensorDataset(pt_in, pt_trg), batch_size=100, shuffle=True)

    for iepoch in range((len(cycles)+1)*300):
        net.train()
        for sample_batched in loader:
            batch_in = sample_batched[0].to(device)
            batch_trg = sample_batched[1].to(device)
            optimizer.zero_grad()
            batch_out = net.forward(batch_in)
            loss = weighted_mse_loss(batch_out, batch_trg, weight=pt_weight)
            loss.backward()
            optimizer.step()
        net.eval()
        if iepoch % 100 == 0:
            with torch.no_grad():
                pt_out = net.forward(pt_in)
                loss = weighted_mse_loss(pt_out, pt_trg, weight=pt_weight)
                print("   ",iepoch, loss.data.item())

    with torch.no_grad():
        np_out = net.forward(pt_in).cpu().numpy()

    return np_out


    '''
        approximations.append(np_out)

        cycles.append(new_cycle)
    '''

    #path_save,_ = os.path.splitext(path)
    #path_save = path_save + "_{}.bvh".format(itr)
    #extract_bvh_array.swap_data_in_bvh(path_save, np_out, path)
    #print("trans_diff:",np_diff[:,:3].max())
    #print("angle_diff:",np_diff[:,3:].max())
    #plt.plot(history0)
    #plt.show()
    #print(numpy.abs(dct(history0)))

if __name__ == "__main__":
    path = os.path.join( os.getcwd(), "asset", "06_08.bvh")
    np_trg = extract_bvh_array.extract_bvh_array(path)
    np_trg = np_trg[1:,]
    np_weight = numpy.ones([np_trg.shape[1]])
    np_weight[:3] = 10
    apps = compress(np_trg, np_weight, 3)

