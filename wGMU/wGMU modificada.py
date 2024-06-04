#wGMU modificada

#Las librerias iguales
from torch import nn, tanh, sigmoid, relu, FloatTensor, rand, stack, optim, cuda, softmax, save, device, tensor, int64, no_grad, concat
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import shutil
from alive_progress import alive_bar
from torchsummary import summary
import warnings
warnings.filterwarnings("ignore")

#Función PAra la capa ajustada de SoftMax (usada más adelante)
def SoftmaxModified(x):
  input_softmax = x.transpose(0,1)
  function_activation = nn.Softmax(dim=1)
  output = function_activation(input_softmax)
  output = output.transpose(0,1)
  return output

#GMU adaptada para 5 modalidades
class MultiModalGMUAdapted(nn.Module):

    def __init__(self, input_size_array, hidden_size, dropoutProbability):
        super(MultiModalGMUAdapted, self).__init__()
        self.input_size_array = input_size_array
        self.modalities_number = len(input_size_array)
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropoutProbability)

        # Capas lineales para representaciones ocultas de cada modalidad
        self.h_head = nn.Linear(input_size_array[0], hidden_size, bias=False)
        self.h_ent1 = nn.Linear(input_size_array[1], hidden_size, bias=False)
        self.h_inter = nn.Linear(input_size_array[2], hidden_size, bias=False)
        self.h_ent2 = nn.Linear(input_size_array[3], hidden_size, bias=False)
        self.h_tail = nn.Linear(input_size_array[4], hidden_size, bias=False)

        # Capas lineales para las compuertas de cada modalidad
        self.z_head = nn.Linear(input_size_array[0], hidden_size, bias=False)
        self.z_ent1 = nn.Linear(input_size_array[1], hidden_size, bias=False)
        self.z_inter = nn.Linear(input_size_array[2], hidden_size, bias=False)
        self.z_ent2 = nn.Linear(input_size_array[3], hidden_size, bias=False)
        self.z_tail = nn.Linear(input_size_array[4], hidden_size, bias=False)

    def forward(self, inputModalities):
        dict_hiddens = {}
        # Cálculo de representaciones ocultas de cada modalidad
        dict_hiddens['head'] = tanh(self.dropout(self.h_head(inputModalities[0])))
        dict_hiddens['ent1'] = tanh(self.dropout(self.h_ent1(inputModalities[1])))
        dict_hiddens['inter'] = tanh(self.dropout(self.h_inter(inputModalities[2])))
        dict_hiddens['ent2'] = tanh(self.dropout(self.h_ent2(inputModalities[3])))
        dict_hiddens['tail'] = tanh(self.dropout(self.h_tail(inputModalities[4])))

        # Cálculo de compuertas de cada modalidad
        z_head_v = sigmoid(self.dropout(self.z_head(inputModalities[0])))
        z_ent1_v = sigmoid(self.dropout(self.z_ent1(inputModalities[1])))
        z_inter_v = sigmoid(self.dropout(self.z_inter(inputModalities[2])))
        z_ent2_v = sigmoid(self.dropout(self.z_ent2(inputModalities[3])))
        z_tail_v = sigmoid(self.dropout(self.z_tail(inputModalities[4])))

        # Aplicar softmax a las compuertas
        z_normalized = SoftmaxModified(stack([z_head_v, z_inter_v, z_tail_v, z_ent1_v, z_ent2_v]))
        print('GMU z', z_normalized.shape)
        print('GMU z_head', z_head_v.shape)
        print('GMU h_head', dict_hiddens['head'].shape)

        # Calcular la salida final
        final = sum([z_normalized[i] * dict_hiddens[modalidad] for i, modalidad in enumerate(["head", "inter", "tail", "ent1", "ent2"])])
        print('GMU', final.shape)
        return final, z_normalized
