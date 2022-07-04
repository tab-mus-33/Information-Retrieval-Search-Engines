import warnings
warnings.filterwarnings("ignore")


import gensim 
import pandas as pd
import numpy as np



if __name__ == '__main__':
    from nltk.stem import WordNetLemmatizer
    from collections import Counter 
    from operator import itemgetter

    def convert_passage(text):
        lemma = WordNetLemmatizer()
       
        text_cleaned = list(set(lemma.lemmatize(w) for w in text))
        return text_cleaned

    def get_average(keyed,sentence):
        rep=[]
        normed=0
        for i in range(len(sentence)):
            try:
                rep.append((keyed[sentence[i]]/np.linalg.norm(keyed[sentence[i]])))
            except:
                pass
        a=np.array(rep)
        normed=a.mean(axis=0)
        return normed

    def rep_compile(mod,line_passage):
        dummy=[]
        for i in range(len(line_passage)):
            dummy.append(get_average(mod,line_passage[i]))
        return np.vstack(dummy)


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


    def train_test_split(X,y,percent):
        indices=int(len(X)*(1-percent))
        X_train=X[:indices,:]
        y_train=y[:indices]
        X_test=X[indices:,:]
        y_test=y[indices:]
        return X_train,y_train,X_test,y_test

    def eval_metrics(labels,outputs):
        true_positive=0
        false_positive=0
        true_negative=0
        false_negative=0
        recall=0
        precision=0
        f1=0

        for j in range(len(labels)):
            if (labels[j]==0) and (outputs[j]==0):
                true_negative+=1
            elif (labels[j]==1) and (outputs[j]==1):
                true_positive+=1
            elif (labels[j]==1) and (outputs[j]==0):
                false_negative+=1
            elif (labels[j]==0) and (outputs[j]==1):
                false_positive+=1

    
        recall=true_positive/(true_positive+false_positive)
        precision=true_positive/(true_positive+false_negative)

        return precision,recall



    def norm_input(feature):
        _,m=feature.shape
        for dex in range(m):
            vect=feature[:,dex]
            vect= (vect-vect.mean())/vect.std()
            feature[:,dex]=vect

    class model_lr:
        def output(self,var):
            s=1/(1+np.exp(-var))
            return s 
        
        def penalty(self,X_training,labels,W):
            dim=len(X_training)
            predictions=self.output(X_training@W)
            for_one=labels*np.log(predictions)
            for_zero=(1-labels)*np.log(1-predictions)
            total=-np.sum(for_one+for_zero)
            return total/dim
        
        def train_model(self, X_training, y_training, learning_rate, iterations):        
            r,p=X_training.shape
            W=np.zeros(p)
            cost_log=[]
            for j in range(iterations):
                predictions=self.output(X_training@W)
                W = W - learning_rate*(X_training.T@(predictions-y_training))/r
            
                cost=self.penalty(X_training,y_training,W)
                cost_log.append(cost)
            self.trained_w=W
        
            return cost_log
            
        def log_pred_prob(self,X_te):
            predictions=self.output(X_te@self.trained_w)
            return predictions

        def only_predictions(self,X_te):
            predictions=self.output(X_te@self.trained_w)
            class_preds=[1 if element>0.5 else 0 for element in predictions ]
            return np.array(class_preds)

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
                
                NDCG=score/optimized
                map_dict.update({keys:NDCG})
        
        ndcg_score=0
        for keys in map_dict.keys():
            ndcg_score+=map_dict[keys]
        ndcg=ndcg_score/len(map_dict)
        
        return ndcg

    def cos(p_arr, q_arr):
        cosine=[]
        for i in range(len(p_arr)):
            cosine_simalarity=np.inner(q_arr[i,:],p_arr[i,:])/(np.linalg.norm(q_arr[i,:])*np.linalg.norm(p_arr[i,:]))
            cosine.append(cosine_simalarity)
        return np.vstack(cosine)
        

    print("Reading Training Data")
    passage=pd.read_csv("train_data.tsv",sep='\t',header=0)

    print("Starting pre-processing")
    passage["passage"]=passage["passage"].apply(gensim.utils.simple_preprocess)
    passage["queries"]=passage["queries"].apply(gensim.utils.simple_preprocess)

    print("Reading Validation data")
    validation=pd.read_csv("validation_data.tsv",sep='\t',header=0)

    print("Pre-processing on validation data")
    validation["passage"]=validation["passage"].apply(gensim.utils.simple_preprocess)
    validation["queries"]=validation["queries"].apply(gensim.utils.simple_preprocess)


    only_unique_passages=passage.drop_duplicates(['pid']).reset_index()
    only_unique_query=passage.drop_duplicates(["qid"]).reset_index()
    p_val_unique=validation.drop_duplicates(['pid']).reset_index()
    q_val_unique=validation.drop_duplicates(["qid"]).reset_index()

    only_unique_passages=only_unique_passages["passage"].reset_index()
    only_unique_query=only_unique_query["queries"].reset_index()
    p_val_unique=p_val_unique["passage"].reset_index()
    q_val_unique=q_val_unique["queries"].reset_index()

    text=pd.concat([only_unique_passages, p_val_unique], axis=0)
    text2=pd.concat([only_unique_query,q_val_unique],axis=0)

    text.rename(columns={"passage":'text'},inplace=True)
    text2.rename(columns={"queries":'text'},inplace=True)

    training_data=pd.concat([text,text2],axis=0)
    
    del only_unique_passages
    del only_unique_query
    del p_val_unique
    del q_val_unique
    del text
    del text2



    import gensim
    

    model = gensim.models.Word2Vec(window=15,min_count=2,workers=20,sg=1,negative=4,sample=0.001)
    print("Training word embeddings (Word2Vec")
    model.build_vocab(training_data["text"], progress_per=1000)
    model.train(training_data["text"], total_examples=model.corpus_count, epochs=model.epochs)





    save_v = model.wv

    model.save("trained_word.model")

    del model
   
    passage["W"]=passage["relevancy"].map({1:60,0:5})
    sample=passage.sample(n=600000, weights="W")
    passage=sample

    zero, one = passage["relevancy"].value_counts()

    non_relevant = passage[passage['relevancy'] == 0]
    relevant = passage[passage['relevancy'] == 1]

    
    relevant_sampled = relevant.sample(zero, replace=True)

    chek = pd.concat([non_relevant, relevant_sampled], axis=0)


    passage=chek
    passage.reset_index(inplace=True)

    

    

    print("Getting average word embeddings")
    passage_array=rep_compile(save_v,passage["passage"])
    query_array=rep_compile(save_v,passage["queries"])

    
    print("Getting word embeddings for validation data")
    val_pas=rep_compile(save_v,validation["passage"])
    val_que=rep_compile(save_v,validation["queries"])


    cos_score_training=cos(query_array,passage_array)

    X=np.hstack((query_array,passage_array,cos_score_training))
    Y=np.array(passage["relevancy"])

    norm_input(X)
    
    print("Fitting the model")
    model=model_lr()
    loss_one=model.train_model(X,Y,0.9,400)
    
    y_preds=model.only_predictions(X)

    print("Saving the cross-entropy loss per epoch in a list using json")
    import json

    with open("loss_0.9_400.","w") as f:
        json.dump(loss_one,f)

    p,r=eval_metrics(Y,y_preds)

    print("The precision is {} and recall is {} on the training set after 400 iterations and learning rate of 0.9".format(p,r))
    print("Getting predictions")

    cos_val=cos(val_que,val_pas)

    X_val=np.hstack((val_pas,val_que,cos_val))

    norm_input(X_val)

    predictions=model.log_pred_prob(X_val)

    scores=list(predictions)
    print("Calculating Map and NDCG")
    validation["scores"]=scores

    val_score={}
    for index in range(len(validation)):
        if validation["qid"][index] not in val_score.keys():
            val_score.update({validation["qid"][index]:[]})
        val_score[validation["qid"][index]].append([validation["pid"][index],validation["scores"][index]])

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

    vd_3=pd.merge(vd_3,validation[["qid","pid","relevancy"]],on=["qid","pid"],how="left")
    vd_10=pd.merge(vd_10,validation[["qid","pid","relevancy"]],on=["qid","pid"],how="left")
    vd_100=pd.merge(vd_100,validation[["qid","pid","relevancy"]],on=["qid","pid"],how="left")

    a=validation.groupby(["qid"],as_index=False)["relevancy"].sum()
    new_dict={}
    for i in range(len(a)):
        if a["qid"][i] not in new_dict.keys():
            new_dict.update({a["qid"][i]:a["relevancy"][i]})

    
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

    vd_100["assignment"]="A2"
    vd_100["algorithm"]="LR"

    vd_100["rank"] = vd_100.groupby("qid")["score"].rank("first", ascending=False)
    vd_100=vd_100[["qid","assignment","pid","rank","score","algorithm"]]

    vd_100.to_csv("LR.txt",header=None,index=None, sep='\t', mode='a')
    
        