# -*- coding: utf-8 -*-
"""
Created on Fri May  1 22:30:35 2020

@author: jnke2
"""
import networkx as nx
import pandas as pd
import numpy as np
import random
from tqdm import tqdm

from collections import defaultdict
import matplotlib.pyplot as plt;


#from pandas.tools.plotting import parallel_coordinates




N=10 #is the total number of nodes 
L=5 #length of the walk
class node_to_vector():
    w= 3
    s= 10
    n=10
    l=0.01 
    
    
    def Return_node_vec(self, node): #This function returns the vector representation of each node
        n_index = self.node_index[node]
        v_n = self.w1[n_index] 
        
        #print()
        return v_n
    
    def __init__(self,window_size=3,s_h_lay=10,num_train=500,learn_rate=0.01): #I can initialize it differently
        
        self.window_size = window_size
        self.s_h_lay = s_h_lay      #size of the hiddden layer or the dimension of my node embedding
        self.num_train = num_train #number of training
        self.learn_rate = learn_rate
        #print (self.window_size)
        
    def generate_training_data(self, Graph_1):
       
        node_counts={}  
        for i in range (self.s_h_lay):
            node_counts[str(i)]=i+1
            
        
        #print(node_counts)
        self.v_count = len(node_counts.keys())   #can remove this
        #self.v_count is an integer
        self.node_list = list(node_counts.keys())   #get the list of nodes
        self.node_index = dict((node, i) for i, node in enumerate(self.node_list))
        self.index_node = dict((i, node) for i, node in enumerate(self.node_list)) #you can fix this to keep their indices the same
        print(self.node_index)
        training_data = []

        
        for chain in Graph_1:
            chain_len = len(chain)
            for i, node in enumerate(chain):
             
                node_vec=[0,0,0,0,0,0,0,0,0,0]
                node_index = self.node_index[chain[i]]
                node_vec[node_index] = 1
                n_target = node_vec
                #for(i in range)
                #print(n_target)
                #print(n_target)

                
                n_context = []
                for j in range(i - self.window_size, i + self.window_size+1):
                    if j != i and j <= chain_len-1 and j >= 0:
                        
                        node_vec_2=[0,0,0,0,0,0,0,0,0,0]
                        node_index_2 = self.node_index[chain[j]]
                        #print(node_index_2,"\n")
                        node_vec_2[node_index_2] = 1
                        
                        n_context.append(node_vec_2)
            
            training_data.append([n_target, n_context])
            #print(training_data)
            #print(training_data)
        return np.array(training_data) # I can just call the training function here instead of returning something
    
    
    def train(self, training_data):
        self.list=[]
        self.w1 = np.random.uniform(-1, 1, (self.v_count, self.n))
        self.w2 = np.random.uniform(-1, 1, (self.n, self.v_count))
        for i in range(self.num_train):
            self.loss = 0
            for w_t, w_c in training_data:
                y_pred, h, u = self.forward_prop(w_t)
                EI = np.sum([np.subtract(y_pred, word) for word in w_c], axis=0)
                W_1,W_2=self.backprop(EI, h, w_t)
                self.loss += -np.sum([u[word.index(1)] for word in w_c]) + len(w_c) * np.log(np.sum(np.exp(u)))  #computer a different loss function after
            
            #print(self.loss)
            self.list.append(self.loss)
                #print(self.loss)
        return self.list
                #plt.scatter(i, self.loss)
            #plt.plot(i, self.loss)
            #plt.show()
    
    def forward_prop(self, x): 
    
        h = np.dot(x, self.w1)
        u = np.dot(h, self.w2)
        
        y_c = np.exp(u - np.max(u))/(np.exp(u - np.max(u))).sum(axis=0)
        #print(u,"\n")
        #print("exponential of u is ",np.exp(u),"\n")
        #print("exponential of u-max_u is ",np.exp(u-np.max(u)),"\n")
        #print("predicted output is ",y_c)
      
        
        return y_c, h, u
    
    
    def backprop(self, e, h, x): 
        
        dl_dw2 = np.outer(h, e)
        dl_dw1 = np.outer(x, np.dot(self.w2, e.T))
        
        self.w1 = self.w1 - (self.learn_rate * dl_dw1)
        self.w2 = self.w2 - (self.learn_rate * dl_dw2)
        return self.w1, self.w2
       
        #print(size(self.w1))
        
        
    
    
    def vec_sim(self, node, top_n): # I took this code online to perform the cosine similarity
        v_w1 = self.Return_node_vec(node)
        node_sim = {}

        for i in range(self.v_count):
            v_w2 = self.w1[i]
            theta_sum = np.dot(v_w1, v_w2)
            theta_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
            theta = theta_sum / theta_den

            node = self.index_node[i]
            node_sim[node] = theta
        
        words_sorted = sorted(node_sim.items(), key=lambda kv: kv[1], reverse=True)

        for word, sim in words_sorted[:top_n]:
            print(word, sim)



df = pd.read_csv("test_draft.txt", sep = "\t")
df.head()
G = nx.from_pandas_edgelist(df, "source", "target", edge_attr=True, create_using=nx.Graph())

def get_randomwalk(node, path_length):
    
    random_walk = [node]
    
    for i in range(path_length-1):
        temp = list(G.neighbors(node))  #a list of all the neighbours of the nide
        #print("set temp ",set(temp),"set random ",set(random_walk),"\n") #avoid me to go back to the same node
        temp = list(set(temp) - set(random_walk))     #avoid me to go back to the same node
        #print(temp,"\n")
        if len(temp) == 0:
            break

        random_node = random.choice(temp)
        random_walk.append(random_node)
        #print("the random walk is ",random_walk)
        node = random_node
    #cast the integers into strings
    R = [str(random) for random in random_walk]  
    Random_walk=",".join(R)  
    return Random_walk

#nx.draw(G,with_labels=True,font_weight='bold')
#plt.savefig("simple_path.png") # save as png
#plt.show()
Graph=[]
Graph_1=[]
f=[]
for i in range(N):
    Graph.append([get_randomwalk(i,L)])

Chain=[]
for chain in Graph:
    for node in chain:
        Graph_1.append(node.split(",")) #split each node 
        
#print(Graph_1)
        #[['0', '4', '8', '3', '2'], ['1', '6', '8', '3', '4'],..]



################# Some useful variables ###################################
n2v= node_to_vector() 
cost=[]
training_data = n2v.generate_training_data(Graph_1)

cost=(n2v.train(training_data))

"""fig, ax = plt.subplots()  
ax.plot(np.arange(500), cost, 'r')  
ax.set_xlabel('Iterations')  
ax.set_ylabel('Cost')  
ax.set_title('Error vs. Training Epoch') """



#print(cost,"\n")
node = '2';
#vec = n2v.node_vec(node)

vec=[]


for i in range(10):
    vec.append(n2v.Return_node_vec(str(i)))
    
#print(vec[0])
data=[]  
data=vec

#data = [song1, song2, song3]

#calculate 2d indicators
def indic(data):
    #alternatively you can calulate any other indicators
    max = np.max(data, axis=1)
    min = np.min(data, axis=1)
    return max, min
nod=[0,1,2,3,4,5,6,7,8,9]

x,y = indic(data)

fig, ax = plt.subplots()
ax.scatter(x, y)

for i, txt in enumerate(nod):
    ax.annotate(txt, (x[i], y[i]))

#print(x," \n",y," \n",data,"\n")
#plt.scatter(x, y, marker='x')
"""for i, txt in enumerate(nod):
    ax.annotate(txt, (y[i], x[i]))

plt.show()"""

#print(node, vec)
n2v.vec_sim(node, 3)
print(Graph_1[0])

