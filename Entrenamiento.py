#Entrenamiento

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import random

from tqdm import tqdm


#Funcion para encontrar NANS
def encNan(lisDat):
    #recibe un una lista de tensores
    #los convierte en una lista final y la recorre para encontrar NAN
    salida = []
    nan = 0
    cero = data[0].flatten().tolist()
    uno = data[1].flatten().tolist()
    dos = data[2].flatten().tolist()
    tres = data[3].flatten().tolist()
    cuatro = data[4].flatten().tolist()
    cinco = data[5].flatten().tolist()
    salida.extend(cero)
    salida.extend(uno)
    salida.extend(dos)
    salida.extend(tres)
    salida.extend(cuatro)
    salida.extend(cinco)
    for nume in salida:
        if math.isnan(nume):
            nan+=1
    return nan

#Defino el dispositivo para mover a la GPU
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

#semilla
seed = 42
random.seed(seed)
torch.manual_seed(seed)

# hiperparámetros
learning_rate = 0.001
epochs = 2  #debe ser 200 para un buen entrenamiento, antes habia 10
batch_size = 10

# instancio el modelo
model = CNN()

#Mando el modelo a la GPU
model = CNN().to(device)

#función de pérdida
loss_function = nn.CrossEntropyLoss()

#optimizador
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# instancio dataset
#DIRECCIONNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
full_dataset = processMatriz(["/content/Matriz2.json"])
#/home/patrones/Escritorio/Jose/Pruebas spyder/ArchivoSinNAN/nuevoTrain.json
print('La longitud del dataset completo será: ', len(full_dataset))
#Salio 10 pero esta bien porque es un archivo de 10 oraciones......


#proporción de entrenamiento y prueba
train_ratio = 0.9
test_ratio = 0.1
#opr qe no me deja moverlos?

#tamaño de los conjuntos de entrenamiento y prueba
train_size = int(train_ratio * len(full_dataset))
test_size = len(full_dataset) - train_size

print('los tamaños de los conjuntos es: \n train size: ', train_size,'\n test size: ', test_size)

#división con la semilla
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# DataLoader para entrenamiento y prueba
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) #mantener ese shuffle (evita el sobreajuste de clases)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size) #aqui no importa el shuffle
print('La longitud del dataloader de train será: ', len(train_dataloader))
print('La longitud del dataloader de test será: ', len(test_dataloader))

#listas para almacenar métricas
train_losses = []
train_accuracies = []
validation_losses = []
validation_accuracies = []
contadorent = 0

#Parámetros para early stopping
bestValLoss = 100
epochswi = 0 #without improvement
patience = 8  #epocas sin mejora antes de parar

#para el rendimiento de validacion
trueVal = []
predVal = []
capaSoftVal = nn.Softmax(dim=0)

#Contador provisional
contAcc = 0

#contadores adicionales
contTrainAc = 0
contValAc = 0

#contador para borrar
borrar = 0

print("Bien hasta antes del entreno")

# Ciclo de entrenamiento
for epoch in tqdm(range(epochs)):
    correct_train = 0
    total_loss = 0
    contadorent+=0

    contTrainAc = 0
    contValAc = 0

    #print(contadorent)

    model.train()

    print("Comienzo la epoca ------------------------------")

    for data in train_dataloader:

        #print('Dentro del ciclo pequeño')
        contAcc+=1
        #correct_train = 0
         #Detector de NAN
        numnan = encNan(data)
        if numnan!=0:
            print("NAN encontrada en la epoca: ", contadorent, ". Con un total de NAN de :", numnan)

        #datos hacia la GPU
        inputs_head, inputs_ent1, inputs_inter, inputs_ent2, inputs_tail, labels = [item.to(device) for item in data]

        #Contador rendimiento
        contTrainAc += len(labels)

        #para ver los resultados
        #borrar+=1
        #print("Se entró al entreno: ", borrar)
        #print("el contador de muestras va en: ", contTrainAc)

        #if borrar < 6:
        #    print("\n Predicted: ", predicted)
        #    print("\n labels", labels)
        #if borrar == 72:
        #    print("\n Predicted: ", predicted)
        #    print("\n labels", labels)
        optimizer.zero_grad()

        #Reajusto las dimensiones para que queden de los tamaños adecuados

        outputs = model(inputs_head, inputs_ent1, inputs_inter, inputs_ent2, inputs_tail)
        #print("outputs para el loss: ", outputs.size(), " labels: ", labels.size())
        loss = loss_function(outputs, labels[:,0])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        correct_train += (predicted == labels.view(-1)).sum().item()

        #print("el numero de aciertos es: ", correct_train)
        #print("acabo el ciclo pequeno")
        #print('Correct train : ',correct_train)
    #print('Salió del ciclo pequeño')
    #print('VALOOOOOOR DEL CONTADOR............',contAcc)
    average_train_loss = total_loss / contTrainAc
     #Revisar estas accuracies que se salen del rango. incluso con la division por dataloader
    #print('\n\n Para el train accuracy -> correct train: ',correct_train,'full len: ',len(train_dataset),'\n\n')
    accuracy_train = (correct_train / contTrainAc)
    print('Resultado total train: ', accuracy_train)

    #AÑADIDO
    #correct_train = 0
    #total_loss = 0

    train_losses.append(average_train_loss)
    train_accuracies.append(accuracy_train)

    #VALIDACION
    model.eval()
    validation_loss = 0.0
    correct_validation = 0

    with torch.no_grad():
      #El no grad sería util para lo otro?
        for data in test_dataloader:


            inputs_head, inputs_inter, inputs_tail, inputs_ent1, inputs_ent2, labels = [item.to(device) for item in data]
            outputs = model(inputs_head, inputs_inter, inputs_tail, inputs_ent1, inputs_ent2)
            #print('labels', labels)
            #print('outputs', outputs)
            #print('\n\n\n\n')

            #Contador rendimiento
            contValAc += len(labels)

            trueVal.append(labels)

            outputVal = capaSoftVal(outputs)
            rel_pred_val = outputVal.argmax()
            predVal.append(rel_pred_val)
            predicted_labels = [tensor.item() for tensor in predVal]


            loss = loss_function(outputs, labels[:,0])
            validation_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct_validation += (predicted == labels.view(-1)).sum().item()
            #correct_train += (predicted == labels).sum().item()


    average_validation_loss = validation_loss / contValAc
    #print('\n\n Para el validation accuracy -> correct validation: ',correct_validation,'full len: ',len(test_dataset),'\n\n')
    accuracy_validation =  (correct_validation / contValAc)  #Se divide por el dataloader, no dataset
    #print('Resultado total validation: ', accuracy_validation)

    validation_losses.append(average_validation_loss)
    validation_accuracies.append(accuracy_validation)

    #AÑADIDO
    validation_loss = 0
    correct_validation = 0

    #Condicionales del early stopping
    if average_validation_loss < bestValLoss:
        bestValLoss = average_validation_loss
        # Guardar el modelo actual (model.state_dict()) o crear una copia del mejor modelo
        torch.save({
                  'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'train_losses': train_losses,
                  'train_accuracies': train_accuracies,
                  'validation_losses': validation_losses,
                  'validation_accuracies': validation_accuracies
              }, '/content/modelo_entrenado.pth')
        #DIRECCIONNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
        epochswi = 0  # reinicio el contador
        #AGREGAR UN BREAK AQUÍ?
    else:
        epochswi += 1

    # compruebo la paciencia
    if epochswi >= patience:
        print('\n')
        print('Early stopping realizado en la epoca numero', epoch)
        break

