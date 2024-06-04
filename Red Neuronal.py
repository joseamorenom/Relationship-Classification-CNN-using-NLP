#Red Neuronal

#Imported Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

#CNN definition
class CNN(nn.Module):
    def __init__(self, kernel_size=(3,1)):
        # input_size = (batch_size, input_channels, input_length)
        super(CNN, self).__init__()

        #Capas convolucionales
        self.conv_head = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size) #cada una sale un vecotr de 1024
        self.conv_inter = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size)
        self.conv_tail = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size)

        #Capas de activación
        self.relu1 = nn.ReLU() #para quitar linealidad
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

        #Capas de Maxpooling
        self.maxpool_1 = nn.MaxPool2d(kernel_size=(5,1), stride=1)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=(3,1),stride=1)
        self.maxpool_3 = nn.MaxPool2d(kernel_size=(9,1), stride=1)

        #Fully Connected
        self.fc1 = nn.Linear(5*1024, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

        # Definir la capa wGMU para combinar las salidas de las modalidades
        self.wGMU = MultiModalGMUAdapted([1024,1024,1024,1024,1024], hidden_size=1024, dropoutProbability=0.1)


    def forward(self, x_head,x_ent1, x_inter, x_ent2, x_tail):
      #imprimir el shape de todas esas entradas y miro si el dataset las está entregando bien
        out_head = self.conv_head(x_head)
        out_inter = self.conv_inter(x_inter)
        out_tail = self.conv_tail(x_tail)

        # aplico relu y maxpooling
        out_head = self.relu1(out_head)
        out_inter = self.relu2(out_inter)
        out_tail = self.relu3(out_tail)

        out_head = self.maxpool_1(out_head)
        out_inter = self.maxpool_2(out_inter)
        out_tail = self.maxpool_3(out_tail)

        #out_head = out_head[:,0,0,:]
        #out_inter = out_inter[:,0,0,:]
        #out_tail = out_tail[:,0,0,:]

        print("Sin recorte (head): ", out_head.size())
        print("Sin recorte (inter): ", out_inter.size())
        print("Sin recorte (tail): ", out_tail.size())

        x_ent1 = torch.tanh(x_ent1)
        x_ent2 = torch.tanh(x_ent2)

        #ajusto las dimensiones que voy a meter a la wGMU

        #Validar las dimensiones antes de ese mean. en ambas arquitecturas
        #Ambas deben ser uno en los 3 contextos. (Sin)

        out_head = torch.mean(out_head, dim=(1, 2))
        out_inter = torch.mean(out_inter, dim=(1, 2))
        out_tail = torch.mean(out_tail, dim=(1, 2))


        # Llamar al modelo wGMU para combinar las salidas de las modalidades
        out_combined, compuertas = self.wGMU([out_head, x_ent1, out_inter, x_ent2, out_tail])

        print(out_combined.shape)


        # aplico fully connected (La primera se quita porque la wGMU ya hace esta primera reducción de tamaño)
        #out_combined = self.fc1(out_combined)
        out_combined = self.fc2(out_combined)
        out_combined = self.fc3(out_combined)

        return out_combined

