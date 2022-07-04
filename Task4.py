import warnings
warnings.filterwarnings("ignore")


import gensim 
import pandas as pd
import numpy as np


"""Check the end of file for the LSTM network that failed"""



if __name__ == '__main__':
    import warnings
    import xgboost as xgb
    warnings.filterwarnings("ignore")
    from operator import itemgetter
    import gensim 
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.neural_network import MLPClassifier


    from collections import Counter 


    

    def norm_input(feature):
        _,m=feature.shape
        for dex in range(m):
            vect=feature[:,dex]
            vect= (vect-vect.mean())/vect.std()
            feature[:,dex]=vect


    def sort_queries(master_dict,k):
            similarity_score=master_dict.copy()
            for keys in similarity_score.keys():
                similarity_score[keys]=sorted(similarity_score[keys],key=itemgetter(1),reverse=True)[:k]
            return similarity_score


    def col_save(sorted_list):
            a=[]
            for keys in sorted_list.keys():
                for i in range(len(sorted_list[keys])):
                    a.append([keys,sorted_list[keys][i][0],sorted_list[keys][i][-1]])
            return a

    def mean_average_precision(passage,bm_dict,optimized_key,num):
            map_dict={}
            for keys in bm_dict.keys():
        
                score=0
                relevance=0
                if len(bm_dict[keys])==num:
                    for i in range(len(bm_dict[keys])):
                        if (passage[(passage["qid"]==keys)&(passage["pid"]==bm_dict[keys][i][0])].iloc[0]["relevancy"])>0:
                            relevance += (passage[(passage["qid"]==keys)&(passage["pid"]==bm_dict[keys][i][0])].iloc[0]["relevancy"])
                            score+=relevance/(i+1)
                    score=score/(optimized_key[keys])
                    map_dict.update({keys:score})
            
            map_score=0
            for keys in map_dict.keys():
                map_score+=map_dict[keys]
            
            
            map_score=map_score/len(map_dict)
            
            return map_score

    def NDCG(passage,bm_dict,optimized_key,num):
            map_dict={}
            for keys in bm_dict.keys():
        
                score=0
                relevance=0
                optimized=0
                if len(bm_dict[keys])==num:
                    for i in range(len(bm_dict[keys])):
                        if (passage[(passage["qid"]==keys)&(passage["pid"]==bm_dict[keys][i][0])].iloc[0]["relevancy"])>0:
                            gain=(2**(passage[(passage["qid"]==keys)&(passage["pid"]==bm_dict[keys][i][0])].iloc[0]["relevancy"]))-1
                            discount=np.log2(i+1+1)
                            score+=(gain/discount)
                        optimized=0   
                        if (optimized_key[keys]==1):
                            optimized+=1
                        elif (optimized_key[keys]==2):
                            optimized+=1+(1/np.log2(3))
                        elif (optimized_key[keys]==3):
                            optimized+=1+(1/np.log2(3))+(1/np.log2(4))
                        elif (optimized_key[keys]==4):
                            optimized+=1+(1/np.log2(3))+(1/np.log2(4))+(1/np.log2(5))

                    if optimized!=0:
                        NDCG=score/optimized
                    else:
                        NDCG=0
                    
                    map_dict.update({keys:NDCG})
            
            ndcg_score=0
            for keys in map_dict.keys():
                ndcg_score+=map_dict[keys]
            ndcg=ndcg_score/len(map_dict)
            
            return ndcg

    
    bm=pd.read_pickle("bm25h.pkl")
  

    bm2=pd.read_pickle("valdh.pkl")
    

    X_v=bm2[["score","tf_idf_passage_sum","tf_idf_query_sum","passage_length","query_length","tf_idf_diff","pid/qid_feature","cosine_similarity"]]
    Y_v=bm2[["relevancy"]]

    X_v=np.array(X_v)
    Y_v=np.array(Y_v)

    clf = MLPClassifier(  activation="relu", solver="adam", batch_size=200, hidden_layer_sizes=(50, 2), random_state=1, max_iter=400)

    X=bm[["score","tf_idf_passage_sum","tf_idf_query_sum","passage_length","query_length","tf_idf_diff","pid/qid_feature","cosine_similarity"]]
    Y=bm[["relevancy"]]

    X=np.array(X)
    norm_input(X)

    clf.fit(X,Y)

    
    


    norm_input(X_v)

    a=clf.predict_proba(X_v)
    a=a[:,1]

    bm2["scores"]=list(a)
    val_score={}
    
    for index in range(len(bm2)):
        if bm2["qid"][index] not in val_score.keys():
            val_score.update({bm2["qid"][index]:[]})
        val_score[bm2["qid"][index]].append([bm2["pid"][index],bm2["scores"][index]])
    def sort_queries(master_dict,k):
            similarity_score=master_dict.copy()
            for keys in similarity_score.keys():
                similarity_score[keys]=sorted(similarity_score[keys],key=itemgetter(1),reverse=True)[:k]
            return similarity_score

    def col_save(sorted_list):
            a=[]
            for keys in sorted_list.keys():
                for i in range(len(sorted_list[keys])):
                    a.append([keys,sorted_list[keys][i][0],sorted_list[keys][i][-1]])
            return a

    from operator import itemgetter
    val_100=sort_queries(val_score,100)
    val_10=sort_queries(val_score,10)
    val_3=sort_queries(val_score,3)

    df_list_3=col_save(val_3)
    df_list_10=col_save(val_10)
    df_list_100=col_save(val_100)

    vd_3=pd.DataFrame(df_list_3,columns=["qid","pid","score"])
    vd_10=pd.DataFrame(df_list_10,columns=["qid","pid","score"])
    vd_100=pd.DataFrame(df_list_100,columns=["qid","pid","score"])

    vd_3=pd.merge(vd_3,bm2[["qid","pid","relevancy"]],on=["qid","pid"],how="left")
    vd_10=pd.merge(vd_10,bm2[["qid","pid","relevancy"]],on=["qid","pid"],how="left")
    vd_100=pd.merge(vd_100,bm2[["qid","pid","relevancy"]],on=["qid","pid"],how="left")

    f=bm2.groupby(["qid"],as_index=False)["relevancy"].sum()
    
    new_dict={}
    for i in range(len(f)):
        if f["qid"][i] not in new_dict.keys():
            new_dict.update({f["qid"][i]:f["relevancy"][i]})
        

    map_3=mean_average_precision(vd_3,val_3,new_dict,3)
    map_10=mean_average_precision(vd_10,val_10,new_dict,10)
    map_100=mean_average_precision(vd_100,val_100,new_dict,100)

    ndcg_3=NDCG(vd_3,val_3,new_dict,3)
    ndcg_10=NDCG(vd_10,val_10,new_dict,10)
    ndcg_100=NDCG(vd_100,val_100,new_dict,100)

    print("The MAP at rank 3 is {}".format(map_3))
    print("The MAP at rank 10 is {}".format(map_10))
    print("The MAP at rank 100 is {}".format(map_100))

    print("The NDCG at rank 3 is {}".format(ndcg_3))
    print("The NDCG at rank 10 is {}".format(ndcg_10))
    print("The NDGC at rank 100 is {}".format(ndcg_100))

    vd_100["assignment"]="A1"
    vd_100["algorithm"]="NN"

    vd_100["rank"] = vd_100.groupby("qid")["score"].rank("first", ascending=False)
    vd_100=vd_100[["qid","assignment","pid","rank","score","algorithm"]]

    print("Saving the predictions for top 100 retrieved passages as NN txt")
    vd_100.to_csv("NN.txt",header=None,index=None, sep='\t', mode='a')


    """
    The Code for LSTM network 

    class load_data(Dataset):

    def __init__(self,inputs,labels):
        self.inputs = inputs
        self.labels = labels
        
    def __getitem__(self,index):
       return self.inputs[index], self.labels[index]
    
    def __len__(self):
        return len(self.inputs)



    class LSTM(nn.Module):
        def __init__(self, dim_input, dim_hidden, total_layers, no_labels,drop):
            super(LSTM, self).__init__()
            self.total_layers = total_layers
            self.dim_h= dim_hidden
            self.longterm= nn.LSTM(dim_input, dim_hidden, total_layers, dropout= drop)
            self.linear= nn.Linear(dim_hidden,no_labels) 
            self.sigmoid = nn.Sigmoid()

        
    def forward(self, input):
        batch_shape=input.size(0)
        cell_initial=torch.zeros(self.total_layers, batch_shape).to(device)
        hidden_inital=torch.zeros(self.total_layers, batch_shape).to(device)
        pas,_ = self.longterm(input,(hidden_inital,cell_initial))
      
        classified= self.linear(pas)
        classified= self.sigmoid(classified)
        return classified 




        import torch
        import torch.nn as nn
        import torch.optim as optim

        from torch.utils.data import Dataset




        x=torch.tensor(bm[["score","tf_idf_passage_sum","tf_idf_query_sum","passage_length","query_length","tf_idf_diff","pid/qid_feature","cosine_similarity"]].values)

        y = torch.tensor(labels).unsqueeze(1)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        load = load_data(x,y)

        train_loader = torch.utils.data.DataLoader(dataset=load, batch_size=1000, shuffle=True)

        neural = LSTM(8,1000,8,1,0.2).to(device)





        crit= torch.nn.BCELoss(size_average=True)
        opt= torch.optim.Adam(neural.parameters(), 0.01)
        ep = 10
        for i in range(ep):
            for data,target in train_loader:
                target = target.float()
                data = data.float()
                preds = neural(data)
                loss = crit(preds, target)
                opt.zero_grad()
                loss.backward()
                pt.step()
        
            preds = (preds>0.5).float()
        
            metric = (preds == target).float().mean()
            print("Total epochs are {} and the current is {}. Accuracy is {:.2f}  and the loss is  {:.2f}".format(ep,i+1, accuracy, loss))

    """