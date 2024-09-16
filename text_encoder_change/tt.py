import pickle


with open('train_rest.pkl', 'rb') as f:
    data=pickle.load(f)
data_rest=[]
print(len(data))
print(len(data[0]))
