#Rendimiento

#Retorno del modelo entrenado
checkpoint = torch.load('/home/patrones/Escritorio/Jose/Pruebas spyder/ArchivoSinNAN/modelo_entrenado.pth')
#/home/patrones/Escritorio/Jose/Pruebas spyder/ArchivoSinNAN/modelo_entrenado.pth

#creo una nueva instancia del modelo
model = CNN()

#cargo el estado del modelo desde el archivo
model.load_state_dict(checkpoint['model_state_dict'])

#pongo el modelo en modo de evaluación (no entrenamiento porque ya entrene)
model.eval()

#Creo la softmax para los predicted labels
softm = nn.Softmax(dim=0)

#CICLO DE TEST

# instancio dataset
full_dataset = processMatriz(["/home/patrones/Escritorio/Jose/Pruebas spyder/ArchivoSinNAN/nuevoTest.json"])
#/home/patrones/Escritorio/Jose/Pruebas spyder/ArchivoSinNAN/nuevoTest.json
test_dataloader = DataLoader(full_dataset, batch_size=5)

true_labels=[]
predicted_labelst = []

capaSoft = nn.Softmax(dim=0)
iter = 0
for data in test_dataloader:

     #Detector de NAN
    numnanTest = encNan(data)
    if numnanTest!=0:
        print("NAN encontrada en la iteracion de test: ", iter, ". Con un total de NAN de :", numnanTest)

    iter+=iter
    
    true_labels.append(data[-1])

    inputs_head, inputs_inter, inputs_tail, inputs_ent1, inputs_ent2, labels = data
    outputs = model(inputs_head, inputs_inter, inputs_tail, inputs_ent1, inputs_ent2)
    contadorPred = 0
    for tens in outputs:
        contNantens = 0
        for finder in tens:
            if math.isnan(finder):
                contNantens+=1
        if contNantens!=0:
            print("Hay un total de: ", contNantens, " en la iteracion de tensor: ", contadorPred)
        output = capaSoft(tens)
        rel_pred = output.argmax()
        predicted_labelst.append(rel_pred)
        contadorPred+=1

predicted_labels = [tensor.item() for tensor in predicted_labelst]
true_labelsf = [valor.item() for tensor_lista in true_labels for valor in tensor_lista]


from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

true_labels = true_labelsf

#accuracy
accuracy = accuracy_score(true_labels, predicted_labels) #aqui está normalizado
print(f'Accuracy: {accuracy:.2f}')
print('\n')

#matriz de confusión
confusion = confusion_matrix(true_labels, predicted_labels)
print('Confusion Matrix:')
print(confusion)

#gráfica de la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=True, yticklabels=True)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

#F1 score
f1 = f1_score(true_labels, predicted_labels, average='weighted')
print(f'F1 Score: {f1:.2f}')