# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: SACHIN S

### Register Number: 212224040283

```
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("C:\\Users\\admin\\Downloads\\Exp-1.csv")
df
x = df[["Input"]].values
y = df[["Output"]].values
xt,xst,yt,yst = train_test_split(x,y,test_size=0.2,random_state=42)
scale1 = MinMaxScaler()
scale2=MinMaxScaler()
xt = scale1.fit_transform(xt)
xst = scale2.fit_transform(xst)
xt = torch.FloatTensor(xt)
xst = torch.FloatTensor(xst)
yt = torch.FloatTensor(yt)
yst = torch.FloatTensor(yst)
class neuralnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1,16),
            nn.ReLU(), 
            nn.Linear(16,8), 
            nn.ReLU(), 
            nn.Linear(8,1)
        )
    def forward(self,x):
        return self.network(x)
model = neuralnet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)
epochs = 1000
losses=[]
for i in range(epochs):
    optimizer.zero_grad()
    pred = model(xt)
    loss = criterion(pred, yt)
    loss.backward()
    optimizer.step()

    if i % 50 == 0:
        print(f"{i}/{epochs} Loss: {loss.item():.4f}")
        losses.append(loss.item())
new = scale1.transform([[16]])
new = torch.FloatTensor(new)

pred = model(new)
print(pred.item())
plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()
```

### Dataset Information

<img width="262" height="547" alt="image" src="https://github.com/user-attachments/assets/be3d1ba3-9276-4f04-98b5-65384094ce8b" />

### OUTPUT

### Training Loss Vs Iteration Plot

<img width="341" height="502" alt="image" src="https://github.com/user-attachments/assets/7596ee59-bddf-4240-85bc-f21fe4fd3e56" />
<img width="875" height="608" alt="image" src="https://github.com/user-attachments/assets/4ef2cf69-c98f-4cda-b641-d968c4b3b57c" />

### New Sample Data Prediction

<img width="291" height="65" alt="image" src="https://github.com/user-attachments/assets/e70ee797-87f4-47bb-90fe-4c6c68da2c7c" />

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
