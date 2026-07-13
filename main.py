import torch as t
import pandas as pd
import torch.nn as nn
import torch.optim as optim

df = pd.read_csv("bitcoin_5ans.csv",skiprows=(1,2))


df["Variation"] = df["Close"].pct_change()
df = df.dropna()
liste_variation = df["Variation"].tolist()
X_liste = []
Y_lste =  []

seuil = 0.005

for i in range(len(liste_variation)-3):
    
    X_liste.append(liste_variation[i:i+3])
    
    var_lendeman = (liste_variation[i+3])
    
    if var_lendeman <- seuil:
        Y_lste.append(0)
    elif var_lendeman > seuil:
        Y_lste.append(2)        
    else:
        Y_lste.append(1)


X = t.tensor(X_liste,dtype=t.float32)

Y = t.tensor(Y_lste, dtype=t.long )

print(f"dataset ready ia will fit with {len(X)} days.." )

#RESEAUX DE NEURONE

model = nn.Sequential(
    nn.Linear(in_features=3,out_features=32),
    nn.ReLU(),
    nn.Linear(in_features=32,out_features=64),
    nn.ReLU(),
    nn.Linear(in_features=64,out_features=64),
    nn.ReLU(),
    nn.Linear(in_features=64,out_features=32),
    nn.ReLU(),
    nn.Linear(in_features=32,out_features=16),
    nn.ReLU(),
    nn.Linear(in_features=16,out_features=8),
    nn.ReLU(),
    nn.Linear(in_features=8,out_features=4)
)


cout = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.005)

print("training starting..")

for epoch in range(15000):
    optimizer.zero_grad()
    
    y_pred = model(X)
    
    loss = cout(y_pred,Y)
    
    loss.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        print(f"Epoch = {epoch:03d}  | loss : {loss.item():.4f}")


print("\n--- Entraînement terminé ! ---")