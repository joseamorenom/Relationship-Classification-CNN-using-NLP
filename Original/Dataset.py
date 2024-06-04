#Dataset

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import json
import numpy as np

class processMatriz(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.inic_m = None
        self.head_m = None
        self.inter_m = None
        self.tail_m = None
        self.final_m = None
        self.relation = None

        #Funciones adicionales
        def posicionesh(num):
          if len(h_pos[num]) == 1:
                valorh = h_pos[num]
                return valorh
          else:
                valoresh = h_pos[num]
                return valoresh

        def posicionest(num):
            if len(t_pos[num]) == 1:
                  valort = t_pos[num]
                  return valort
            else:
                  valorest = t_pos[num]
                  return valorest

        def padding(tensor, tamano_deseado):
            tamano_actual = tensor.size(0)
            filas_faltantes = tamano_deseado - tamano_actual
            if filas_faltantes > 0:
                # crear un tensor de ceros con las dimensiones adecuadas
                ceros_a_agregar = torch.zeros(filas_faltantes, tensor.size(1))
                # concatenar los ceros al tensor original
                tensor = torch.cat((tensor, ceros_a_agregar), dim=0)
            return tensor

        #proceso los archivos JSON y almaceno las matrices en data
        listasrel = []
        for file_path in file_paths:
            #Cargar el archivo de Json
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
            flat_emb = data["flat_emb"]
            relation = data["relation"]
            listasrel.append(relation)
            #print('DESDE DENTROOO REL: ',relation)
            h_pos_t = data["h_pos"]
            t_pos_t = data["t_pos"]

            #Convierto lo necesario a tensores
            flat_emb_t = torch.tensor(flat_emb) #tensor de matriz directa

            #Procesamiento de la matriz principal
            Matriz3d = flat_emb_t.reshape((flat_emb_t.shape[0], 1, -1, 1024))

            #1. SubListas de elementos necesarios
            inic = torch.tensor([])
            head  = torch.tensor([])
            inter = torch.tensor([])
            tail = torch.tensor([])
            final = torch.tensor([])

            #2. Submatrices 3D
            inic_m = torch.empty(0,7,1024)
            head_m = torch.empty(0,1024)
            inter_m = torch.empty(0,5,1024)
            tail_m = torch.empty(0,1024)
            final_m = torch.empty(0,11,1024)

            #Proceso cada matriz de 100x1024 para la segmentacion
            count = 0
            for matrizsub in Matriz3d:
              inic = matrizsub[0][0:h_pos_t[count][0]][:]
              if len(h_pos_t[count]) ==1 :
                head = matrizsub[0][h_pos_t[count][0]][:]
                inter = matrizsub[0][h_pos_t[count][0]+1:t_pos_t[count][0]][:]
                if len(t_pos_t[count]) ==1 :
                  tail = matrizsub[0][t_pos_t[count][0]][:]
                  final = matrizsub[0][int(t_pos_t[count][0])+1:][:]
                else:
                  tail = matrizsub[0][t_pos_t[count][0]:t_pos_t[count][-1]+1][:]
                  final = matrizsub[0][int(t_pos_t[count][-1])+1:][:]
              else:
                head = matrizsub[0][h_pos_t[count][0]:h_pos_t[count][-1]+1][:]
                inter = matrizsub[0][h_pos_t[count][-1]+1:t_pos_t[count][0]][:]
                if len(t_pos_t[count]) ==1 :
                  tail = matrizsub[0][t_pos_t[count][0]][:]
                  final = matrizsub[0][int(t_pos_t[count][0])+1:][:]
                else:
                  tail = matrizsub[0][t_pos_t[count][0]:t_pos_t[count][-1]+1][:]
                  final = matrizsub[0][int(t_pos_t[count][-1]):][:]
              count = count+1

              #ajuste de tamaños----------------
              if head.dim() != 1:
                head = torch.mean(head, dim=0)
              if tail.dim() != 1:
                tail = torch.mean(tail, dim=0)

              if inic.size(0)>7:
                inic = inic[:7]
              elif inic.size(0)<7:
                inic = padding(inic,7)

              if inter.size(0)>5:
                inter = inter[:5]
              elif inter.size(0)<5:
                inter = padding(inter,5)

              if final.size(0)>11:
                final = final[:11]
              elif final.size(0)<11:
                final = padding(final,11)

              #Formacion de las matrices 3d de salida
              inic_m = torch.cat((inic_m, inic.unsqueeze(0)), dim=0)
              head_m = torch.cat((head_m, head.unsqueeze(0)), dim=0)
              inter_m = torch.cat((inter_m, inter.unsqueeze(0)), dim=0)
              tail_m = torch.cat((tail_m, tail.unsqueeze(0)), dim=0)
              final_m = torch.cat((final_m, final.unsqueeze(0)), dim=0)

            #Hago el reshape final para corregir la dimension faltante
            tamanoOr = inic_m.size()
            tam = tamanoOr[0]
            inic_mf = inic_m.reshape(tam, 1 , 7, 1024)
            head_mf = head_m.reshape(tam, 1024) #Aqui quitamos el 1
            inter_mf = inter_m.reshape(tam,1, 5, 1024)
            tail_mf = tail_m.reshape(tam, 1024)
            final_mf = final_m.reshape(tam,1, 11, 1024)
            

            #guardo todo al final
            if self.inic_m is None:
                self.inic_m = inic_mf
            else:
               self.inic_m = torch.cat((self.inic_m, inic_mf), dim=0)
            if self.head_m is None:
                self.head_m = head_mf
            else:
               self.head_m = torch.cat((self.head_m, head_mf), dim=0)
            if self.inter_m is None:
                self.inter_m = inter_mf
            else:
               self.inter_m = torch.cat((self.inter_m, inter_mf), dim=0)
            if self.tail_m is None:
                self.tail_m = tail_mf
            else:
               self.tail_m = torch.cat((self.tail_m, tail_mf), dim=0)
            if self.final_m is None:
                self.final_m = final_mf
            else:
               self.final_m = torch.cat((self.final_m, final_mf), dim=0)
        listasrel = [elemento for sublista in listasrel for elemento in sublista]
        matriz_resultante = np.array(listasrel).reshape(-1, 1)
        self.relation = matriz_resultante

    def __len__(self):
         #entrega el numero de oraciones procesadas
        if self.inic_m is not None:
            return self.inic_m.shape[0]
        else:
            return 0  # En caso de que no haya procesado nada aun

    def __getitem__(self, index):
        # index dice cuál oracion necesito
        elemento1 = self.inic_m[index]
        elemento2 = self.head_m[index]
        elemento3 = self.inter_m[index]
        elemento4 = self.tail_m[index]
        elemento5 = self.final_m[index]
        elemento6 = self.relation[index]
        #print("index:", index)
        #print("relation: ", self.relation, '\n')
        #print("elemento6: ", elemento6)
        return elemento1,elemento2,elemento3,elemento4,elemento5, elemento6
#Fin Dataset----------------------------------------------------------------------

