import numpy as np
import torch
items = torch.tensor([1,10,12,2,3,5,8])
true_values = [1,4,9,11,29,33,75,9,13,8]
num_recommendations = 6

twelve_accuracy_all = []
temp_accuracy = []

temp_accuracy = []
for i in items:
    if i in true_values:
        temp_accuracy.append(1)
        print("hej")
    else:
        temp_accuracy.append(0)
if(num_recommendations <= len(true_values)):
    temp_accuracy_final = sum(temp_accuracy)/num_recommendations
else:
    temp_accuracy_final = sum((np.sort(temp_accuracy)[0:len(true_values)]))/len(true_values)

twelve_accuracy_all.append(temp_accuracy_final)

value_1 = 58758.57/12000
value_2 = 59099.06/12000

value_2 = 10345.90/12000
value_1 = 10287.15/12000

value_1 = 457.65/12000
value_2 = 440.30/12000

values = [value_1,value_2]
np.mean(values)
np.std(values)

