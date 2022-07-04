import warnings
warnings.filterwarnings("ignore")


import gensim 
import pandas as pd
import numpy as np



if __name__ == '__main__':
    import warnings
    import xgboost as xgb
    warnings.filterwarnings("ignore")
    from operator import itemgetter
    import gensim 
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    from collections import Counter 

    def query_inverted_index(check,vocabulary):
    
            id={}
            for i in range(len(vocabulary)):
                if vocabulary[i] not in id.keys():
                    id.update({ vocabulary[i]:[]})
            #First creates a list object for each word as a key, that is storing passage ids
            for index in range(len(check)):
                for word in check['queries'][index]:
                        if word in id.keys():
                            id[word].append(check['qid'][index])
                            
            #Then counts term frequency using a counter object 
            for keys in id:
                id[keys]=dict(Counter(id[keys]))
            return id
        
        
    def index_invert(check,vocabulary):
            id={}
            for i in range(len(vocabulary)):
                if vocabulary[i] not in id.keys():
                    id.update({ vocabulary[i]:[]})
            uniq=check.drop_duplicates(['pid']).reset_index()
            #First creates a list object for each word as a key, that is storing passage ids
            for index in range(len(uniq)):
                for word in uniq['passage'][index]:
                        if word in id.keys():
                            id[word].append(uniq['pid'][index])
                            
            #Then counts term frequency using a counter object 
            for keys in id:
                id[keys]=dict(Counter(id[keys]))
            return id

    def get_idf(igg):
            ifd={}
            for keys in igg:
                check={}
                for keys_sub in igg[keys]:
                    check=np.log10(1+(N/len(igg[keys])))
                ifd.update({keys:check})
            return ifd

    def return_tf(igg):
            a={}
            for keys in igg:
                for keys_sub in igg[keys]:
                    if keys_sub not in a.keys():
                        a.update({keys_sub:[[keys,igg[keys][keys_sub]]]})
                    else:
                        a[keys_sub].append([keys,igg[keys][keys_sub]])
            for keys in a:
                sum=0
                for i in range(len(a[keys])):
                    sum+= a[keys][i][-1]
                for i in range(len(a[keys])):
                    a[keys][i][-1]=a[keys][i][-1]/sum
            return a


    def term_bm_passage(igg):
            a={}
            for keys in igg:
                for keys_sub in igg[keys]:
                    if keys_sub not in a.keys():
                        a.update({keys_sub:{keys:igg[keys][keys_sub]}})
                    else:
                        a[keys_sub].update({keys:igg[keys][keys_sub]})
            return a

    def term_bm_passage_two(igg):
            a={}
            for keys in igg:
                for keys_sub in igg[keys]:
                    if keys_sub not in a.keys():
                        a.update({keys_sub:[[keys,igg[keys][keys_sub]]]})
                    else:
                        a[keys_sub].append([keys,igg[keys][keys_sub]])
            return a

    def term_bm_query(igg):
            a={}
            for keys in igg:
                for keys_sub in igg[keys]:
                    if keys_sub not in a.keys():
                        a.update({keys_sub:[[keys,igg[keys][keys_sub]]]})
                    else:
                        a[keys_sub].append([keys,igg[keys][keys_sub]])
            return a


    def tf_idf_query_func(tf,ifd):
            tfidf={}
            for keys in tf:
                for i in range(len(tf[keys])):
                    if keys not in tfidf.keys():
                        tfidf.update({keys:[[tf[keys][i][0],tf[keys][i][-1]*ifd[tf[keys][i][0]]]]})
                    else:
                        tfidf[keys].append([tf[keys][i][0],tf[keys][i][-1]*ifd[tf[keys][i][0]]])
            return tfidf


    def bm_idf_func(invert):
            dictionary={}
            for keys in invert:
                check={}
                for keys_sub in invert[keys]:
                    check=1+np.log(((N-len(invert[keys])+0.5)/(len(invert[keys])+0.5)))
                    dictionary.update({keys:check})
            return dictionary




    def map_passage(tfq,dat):
                qtp={}
                for i in tfq.keys():
                    qtp.update({i:[]})
                for index in qtp.keys():
                    temp=list(dat[dat["qid"]==index]["pid"])
                    for q in range(len(temp)):
                        qtp[index].append(temp[q])
                return qtp
        
    def document_length(frequency_table):
                length={}
                for keys in frequency_table:
                    sum=0
                    for i in range(len(frequency_table[keys])):
                        sum+= frequency_table[keys][i][-1]
                        length.update({keys:sum})
                return length
        
        
    def BM25_score(query_x_passage,query_dict,passage_dict,document_l,bm_idf,avdl):
                k1=1.2
                k2=100
                b=0.75
                bm_score={}
                for keys in query_dict.keys():
                    
                    if keys not in bm_score.keys():
                        bm_score.update({keys:[]})
                    for i in range(len(query_x_passage[keys])):
                        bm_score_per_query=0
                    
                        for q_i in range(len(query_dict[keys])):
                            if query_dict[keys][q_i][0] in passage_dict[query_x_passage[keys][i]].keys():
                                f_i=passage_dict[query_x_passage[keys][i]][query_dict[keys][q_i][0]]
                                qfi=query_dict[keys][q_i][-1]
                                idf=bm_idf[query_dict[keys][q_i][0]]
                                K=k1*(1-b)+((b*document_l[query_x_passage[keys][i]])/avdl)
                                bm_score_per_query+=idf*((k1+1)*f_i)/(K+f_i)+((k2+1)*qfi)/(k2+qfi)
                            else:
                                bm_score_per_query+=0
                        bm_score[keys].append([query_x_passage[keys][i],bm_score_per_query])
                return bm_score

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

    def tf_idf_passage(tf,ifd):
            tfidf={}
            for keys in tf:
                for i in range(len(tf[keys])):
                    if keys not in tfidf.keys():
                        tfidf.update({keys:[[tf[keys][i][0],tf[keys][i][-1]*idf[tf[keys][i][0]]]]})
                    else:
                        tfidf[keys].append([tf[keys][i][0],tf[keys][i][-1]*idf[tf[keys][i][0]]])
            return tfidf


    def vector_space(tfidf,vocab):
            vector=np.zeros(len(vocab))
            for index in range(len(tfidf)):
                #for index_array in range(len(vocab)):
                if tfidf[index][0] in vocab:
                    vector[vocab.index(tfidf[index][0])]=tfidf[index][-1]
            return vector



    def cosine_similarity(tf_idf_query,tf_idf_passage,vocab,query_mapper):
            passage={}
            for i in tf_idf_query.keys():
                if i not in passage.keys():
                    passage.update({ i:[]})
            for query_index in tf_idf_query.keys():
                query_vector=vector_space(tf_idf_query[query_index],vocab)
                for pid in range(len(query_mapper[query_index])):
                    passage_vector=vector_space(tf_idf_passage[query_mapper[query_index][pid]],vocab)
                    cosine_simalarity=np.inner(query_vector,passage_vector)/(np.linalg.norm(query_vector)*np.linalg.norm(passage_vector))
                    passage[query_index].append([query_mapper[query_index][pid],cosine_simalarity])
            return passage


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




    def cos(p_arr, q_arr):
            cosine=[]
            for i in range(len(p_arr)):
                cosine_simalarity=np.inner(q_arr[i,:],p_arr[i,:])/(np.linalg.norm(q_arr[i,:])*np.linalg.norm(p_arr[i,:]))
                cosine.append(cosine_simalarity)
            return cosine

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

    

    def hyperpm(df,training_data,holdout_data,lr_list,eta_list,g_list,depth_list,min_ch_list):
        lr=0
        etas=0
        gammas=0
        depth=0
        child=0
        max=0
        for indf in range(len(lr_list)):
            for j in range(len(eta_list)):
                for h in range(len(g_list)):
                    for al in range(len(depth_list)):
                        for hh in range(len(min_ch_list)):
                            model=xgb.train({'eta':eta_list[j],'n_estimators':90,"max_depth":depth_list[al],"gamma":g_list[h],"min_child_weight":min_ch_list[hh],'learning_rate':lr_list[indf],'objective':'rank:pairwise',"eval_metric":"ndcg@10"},training_data)
                            yp=model.predict(holdout_data)
                            ypred=list(yp)
                            df["scores"]=ypred
                            random_dict={}
                            for index in range(len(df)):
                                if df["qid"][index] not in random_dict.keys():
                                    random_dict.update({df["qid"][index]:[]})
                                random_dict[df["qid"][index]].append([df["pid"][index],df["scores"][index]])
                            df_10=sort_queries(random_dict,10)
                            list_10=col_save(df_10)
                            d_10=pd.DataFrame(list_10,columns=["qid","pid","score"])
                            d_10=pd.merge(d_10,df[["qid","pid","relevancy"]],on=["qid","pid"],how="left")
                            z=df.groupby(["qid"],as_index=False)["relevancy"].sum()
                            n_dict={}
                            for i in range(len(z)):
                                if z["qid"][i] not in n_dict.keys():
                                    n_dict.update({z["qid"][i]:z["relevancy"][i]})
                            nd_10=NDCG(d_10,df_10,n_dict,10)
                            print("The NDCG at rank 10 is {}".format(nd_10))
                            if nd_10>max:
                                max=nd_10
                                lr=lr_list[indf]
                                gammas=g_list[h]
                                etas=eta_list[j]
                                depth=depth_list[al]
                                child=min_ch_list[hh]
        params={'eta':etas,'n_estimators':90,"max_depth":depth,"gamma":gammas,"min_child_weight":child,'learning_rate':lr,'objective':'rank:pairwise',"eval_metric":"ndcg@10"}
        
        return params





   
    print("Reading Training Data")
    passage=pd.read_csv("train_data.tsv",sep='\t',header=0)

    
    qid=passage.drop_duplicates(['qid']).reset_index()
    qid=list(qid["qid"])
    qid=qid[:2000]
    passage=passage[passage["qid"].isin(qid)]


    passage.reset_index(inplace=True)
    print("Starting pre-processing")
    passage["passage"]=passage["passage"].apply(gensim.utils.simple_preprocess)
    passage["queries"]=passage["queries"].apply(gensim.utils.simple_preprocess)

    print("Importing the Model")
    model = gensim.models.Word2Vec.load("trained_word.model")
    save_v = model.wv
    
    import gc
    
    print("Getting Word Embeddings for Training set")

    passage_array=rep_compile(save_v,passage["passage"])
    query_array=rep_compile(save_v,passage["queries"])
    cos_score_training=cos(query_array,passage_array)

    print("Getting Cosine Similarity")

    passage["cosine_similarity"]=cos_score_training

    print("Clearing data and collecting garbage")
    del passage_array 
    del query_array
    del cos_score_training
    del model 
    gc.collect()

    print("Calculating other features, BM25 and TFIDF")

    

    appended_passage=[]
    for i in range(len(passage)):
        for j in range(len(passage["passage"][i])):
            appended_passage.append(passage["passage"][i][j])


    vocabulary=Counter(appended_passage)
    vocabulary=list(vocabulary.keys())




    only_unique=passage.drop_duplicates(['pid'])
    N=len(only_unique["pid"])

    inverted_index=index_invert(passage,vocabulary)


    from re import M


    query_index=query_inverted_index(passage,vocabulary)


    term_frequency=return_tf(inverted_index)

    idf=get_idf(inverted_index)

    tf_query=return_tf(query_index)
    tf_idf_query= tf_idf_query_func(tf_query,idf)
    bm_idf=bm_idf_func(inverted_index)





    query_to_passage=map_passage(tf_idf_query,passage)
    freq_pas=term_bm_passage(inverted_index)
    freq_que=term_bm_passage(query_index)
    freq_pas_list=term_bm_passage_two(inverted_index)
    freq_que_list=term_bm_passage_two(query_index)

    document_lengths=document_length(freq_pas_list)



    sum=0
    for keys in document_lengths.keys():
        sum+= document_lengths[keys]
    
    av=sum/len(document_lengths)




    bm_scores=BM25_score(query_to_passage,freq_que_list,freq_pas,document_lengths,bm_idf,av)

    df_list=col_save(bm_scores)
    bm25=pd.DataFrame(df_list,columns=["qid","pid","score"])
    bm=pd.merge(bm25,passage[["pid","qid","relevancy","cosine_similarity"]],on=["qid","pid"], how="left")


    tfidf_passage= tf_idf_passage(term_frequency,idf)
    print("Processing Term Frequency for query")
    

    def map_passage(tf_idf_query,passage_test):
            query_to_passage={}
            for i in tf_idf_query.keys():
                query_to_passage.update({i:[]})
            for index in query_to_passage.keys():
                temp=list(passage_test[passage_test["qid"]==index]["pid"])
                for q in range(len(temp)):
                    query_to_passage[index].append(temp[q])
            return query_to_passage
    
    query_to_passage=map_passage(tf_idf_query,passage)

    passage_sum={}
    for keys in tfidf_passage.keys():
        sum=0
        for i in range(len(tfidf_passage[keys])):
            sum+=tfidf_passage[keys][i][1]
            passage_sum.update({keys:sum})

    query_sum={}
    for keys in tf_idf_query.keys():
        sum=0
        for i in range(len(tf_idf_query[keys])):
            sum+=tf_idf_query[keys][i][1]
            query_sum.update({keys:sum})

    bm["tf_idf_passage_sum"]=bm["pid"].apply(lambda x: passage_sum[x])
    bm["tf_idf_query_sum"]=bm["qid"].apply(lambda x: query_sum[x])
    bm["passage_length"]=bm["pid"].apply(lambda x: len(tfidf_passage[x]))
    bm["query_length"]=bm["qid"].apply(lambda x: len(tf_idf_query[x]))
    bm["tf_idf_diff"]=np.abs(bm["tf_idf_passage_sum"]-bm["tf_idf_query_sum"])
    bm["pid/qid_feature"]=bm["tf_idf_diff"]/bm["tf_idf_passage_sum"]

    print("Saving the file for task 4")
    bm.to_pickle("bm25h.pkl")

    
    print("Creating training and holdout set for hyperparameter tuning")

    vid=bm.drop_duplicates(['qid']).reset_index()
    vnid=list(vid["qid"])
    vtid=vnid[:1200]
    vhid=vnid[1200:]
    vin=bm[bm["qid"].isin(vtid)]
    holdout=bm[bm["qid"].isin(vhid)]
    vin.reset_index(inplace=True)
    holdout.reset_index(inplace=True)

    vin["rank"] = vin.groupby("qid")["score"].rank("first", ascending=False)
    vin=vin[vin["rank"]<300]
    vinx=vin[["score","tf_idf_passage_sum","tf_idf_query_sum","passage_length","query_length","tf_idf_diff","pid/qid_feature","cosine_similarity"]]
    groups=vin.groupby("qid").size().to_frame("size")["size"].to_numpy()
    training_set=xgb.DMatrix(vinx, label=vin["relevancy"])
    training_set.set_group(groups)

    holdout["rank"] = holdout.groupby("qid")["score"].rank("first", ascending=False)
    h_x=holdout[["score","tf_idf_passage_sum","tf_idf_query_sum","passage_length","query_length","tf_idf_diff","pid/qid_feature","cosine_similarity"]]
    gr_h=holdout.groupby("qid").size().to_frame("size")["size"].to_numpy()
    holdout_set=xgb.DMatrix(h_x, label=holdout["relevancy"])
    holdout_set.set_group(gr_h)


    learning_rates=[0.99,0.01]
    eta=[0.1,0.5]
    gamma=[0.3,0.7]
    dp_list=[3,5]
    min_ch=[0.3,0.6]

    print("Starting hyperparameter tuning")

    parameters=hyperpm(holdout,training_set,holdout_set,learning_rates,eta,gamma,dp_list,min_ch)

    print("The best params are")
    print(parameters)

    print("Deleting variables and collecting garbage again")

    del holdout_set
    del holdout 
    del h_x
    del training_set
    del vin
    del vinx
    del vnid
    del appended_passage
    del vocabulary
    del only_unique
    del inverted_index
    del query_index
    del term_frequency
    del idf
    del tf_query
    del tf_idf_query
    del bm_idf
    del query_to_passage
    del freq_pas
    del freq_que
    del freq_pas_list
    del freq_que_list
    del document_lengths

    gc.collect()

    print("Tuning the Final Model")

    bm["rank"] = bm.groupby("qid")["score"].rank("first", ascending=False)
    bm=bm[bm["rank"]<300]
    bx=bm[["score","tf_idf_passage_sum","tf_idf_query_sum","passage_length","query_length","tf_idf_diff","pid/qid_feature","cosine_similarity"]]
    groups=bm.groupby("qid").size().to_frame("size")["size"].to_numpy()
    trainx=xgb.DMatrix(bx, label=bm["relevancy"])
    trainx.set_group(groups)

    model = xgb.train(parameters,trainx)

    import matplotlib.pyplot as plt
    
    print("Model Trained, plotting feature importance and saving")
    ax = xgb.plot_importance(model, color='red')
    fig = ax.figure
    fig.set_size_inches(10, 10)


    fig.savefig("feature_importancen.png")

    print("feature importance saved")

    print("Reading Validation data")
    validation=pd.read_csv("validation_data.tsv",sep='\t',header=0)

    print("Pre-processing on validation data")
    validation["passage"]=validation["passage"].apply(gensim.utils.simple_preprocess)
    validation["queries"]=validation["queries"].apply(gensim.utils.simple_preprocess)

    print("Processing Embeddings and Cosine score")
    val_p=rep_compile(save_v,validation["passage"])
    val_q=rep_compile(save_v,validation["queries"])
    cos_score_val=cos(val_q,val_p)

    validation["cosine_similarity"]=cos_score_val

    del val_p
    del val_q
    del cos_score_val
    gc.collect()

    print("Processing BM25 and other features")
    appended_passage=[]
    for i in range(len(validation)):
        for j in range(len(validation["passage"][i])):
            appended_passage.append(validation["passage"][i][j])

    vocabulary=Counter(appended_passage)
    vocabulary=list(vocabulary.keys())


    only_unique=validation.drop_duplicates(['pid'])
    N=len(only_unique["pid"])

    inverted_index=index_invert(validation,vocabulary)


    from re import M


    query_index=query_inverted_index(validation,vocabulary)


    term_frequency=return_tf(inverted_index)

    idf=get_idf(inverted_index)

    tf_query=return_tf(query_index)
    tf_idf_query= tf_idf_query_func(tf_query,idf)
    bm_idf=bm_idf_func(inverted_index)

    query_to_passage=map_passage(tf_idf_query,validation)
    freq_pas=term_bm_passage(inverted_index)
    freq_que=term_bm_passage(query_index)
    freq_pas_list=term_bm_passage_two(inverted_index)
    freq_que_list=term_bm_passage_two(query_index)




    af=document_length(freq_pas_list)


    sm=0
    for keys in af.keys():
        sm+= af[keys]
    
    avdl=sm/len(af)


    bm_scores=BM25_score(query_to_passage,freq_que_list,freq_pas,af,bm_idf,avdl)

    df_list=col_save(bm_scores)
    bm25=pd.DataFrame(df_list,columns=["qid","pid","score"])


    bm2=pd.merge(bm25,validation[["pid","qid","relevancy","cosine_similarity"]],on=["qid","pid"], how="right")


    idf=get_idf(inverted_index)
    print("Processing TFIDF for passage")
    tfidf_passage= tf_idf_passage(term_frequency,idf)
    print("Processing Term Frequency for query")


    def map_passage(tf_idf_query,passage_test):
            query_to_passage={}
            for i in tf_idf_query.keys():
                query_to_passage.update({i:[]})
            for index in query_to_passage.keys():
                temp=list(passage_test[passage_test["qid"]==index]["pid"])
                for q in range(len(temp)):
                    query_to_passage[index].append(temp[q])
            return query_to_passage

    query_to_passage=map_passage(tf_idf_query,validation)

    passage_sum={}
    for keys in tfidf_passage.keys():
        sum=0
        for i in range(len(tfidf_passage[keys])):
            sum+=tfidf_passage[keys][i][1]
            passage_sum.update({keys:sum})



    query_sum={}
    for keys in tf_idf_query.keys():
        sum=0
        for i in range(len(tf_idf_query[keys])):
            sum+=tf_idf_query[keys][i][1]
            query_sum.update({keys:sum})

    

    bm2["tf_idf_passage_sum"]=bm2["pid"].apply(lambda x: passage_sum[x])


    bm2["tf_idf_query_sum"]=bm2["qid"].apply(lambda x: query_sum[x])
    bm2["passage_length"]=bm2["pid"].apply(lambda x: len(tfidf_passage[x]))
    bm2["query_length"]=bm2["qid"].apply(lambda x: len(tf_idf_query[x]))
    bm2["tf_idf_diff"]=np.abs(bm2["tf_idf_passage_sum"]-bm2["tf_idf_query_sum"])
    bm2["pid/qid_feature"]=bm2["tf_idf_diff"]/bm2["tf_idf_passage_sum"]

    print("Processing Done, Saving Features for task 4")
    bm2.to_pickle(path="valdh.pkl")

    
    print("Now Getting the predictions")

    by=bm2[["score","tf_idf_passage_sum","tf_idf_query_sum","passage_length","query_length","tf_idf_diff","pid/qid_feature","cosine_similarity"]]
    gp=bm2.groupby("qid").size().to_frame("size")["size"].to_numpy()
    train_v=xgb.DMatrix(by)
    train_v.set_group(gp)
    predsi=model.predict(train_v)
    bm2["scores"]=list(predsi)
    bm2.reset_index(inplace=True)

    print("Processing the Scores now")
    val_score={}
    for index in range(len(bm2)):
        if bm2["qid"][index] not in val_score.keys():
            val_score.update({bm2["qid"][index]:[]})
        val_score[bm2["qid"][index]].append([bm2["pid"][index],bm2["scores"][index]])


   
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

    f=validation.groupby(["qid"],as_index=False)["relevancy"].sum()
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
    vd_100["algorithm"]="LM"

    vd_100["rank"] = vd_100.groupby("qid")["score"].rank("first", ascending=False)
    vd_100=vd_100[["qid","assignment","pid","rank","score","algorithm"]]

    print("Saving the predictions for top 100 retrieved passages as LM.txt")
    vd_100.to_csv("LM.txt",header=None,index=None, sep='\t', mode='a')
