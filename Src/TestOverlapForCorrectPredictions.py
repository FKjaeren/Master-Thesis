#Load customers where 1 correct product prediction have been made

with open('Results/correctlypredicted_customers_with_FM.txt') as f:
    FM_customers = f.readlines()

with open('Results/correctlypredicted_customers_with_MLP.txt') as f:
    MLP_customers = f.readlines()

with open('Results/correctlypredicted_customers_with_DeepFM.txt') as f:
    DeepFM_customers = f.readlines()


res_MLP = []
res_FM = []
res_DeepFM = []
for sub in FM_customers:
    res_FM.append(float(sub.replace("\n", "")))

for sub in MLP_customers:
    res_MLP.append(float(sub.replace("\n", "")))

for sub in DeepFM_customers:
    res_DeepFM.append(float(sub.replace("\n", "")))

print(f"There were {len(res_FM)} customer for which the FM model predicted on article correctly out of 2125")
print(f"There were {len(res_MLP)} customer for which the MLP model predicted on article correctly out of 2125")
print(f"There were {len(res_DeepFM)} customer for which the MLP model predicted on article correctly out of 2125")


temp_accuracy = sum(res_FM==i for i in res_MLP)

count=0
customer_overlap = []
for customer in res_MLP:
    if(customer in res_FM):
        customer_overlap.append(customer)
        count+=1

print(f"Of the {len(res_MLP)} customers which MLP made 1 correct prediction {count} of them were also predicted correctly by FM")

count_deep_overlap=0
for customer in res_DeepFM:
    if(customer in customer_overlap):
        count_deep_overlap+=1

count_deep_overlap_MLP=0
for customer in res_DeepFM:
    if(customer in res_MLP):
        count_deep_overlap_MLP+=1

count_deep_overlap_FM=0
for customer in res_DeepFM:
    if(customer in res_FM or customer in res_MLP):
        count_deep_overlap_FM+=1

print(f"The overlap between FM and deepFM is :{count_deep_overlap_FM}")
print(f"The overlap between MLP and deepFM is :{count_deep_overlap_MLP}")
print(f"The overlap between MLP and FM is :{count_deep_overlap}")