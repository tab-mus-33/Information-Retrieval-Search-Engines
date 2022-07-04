import warnings
warnings.filterwarnings("ignore")

import gensim 
import pandas as pd
import numpy as np


from collections import Counter 
import matplotlib.pyplot as plt
from pathos.multiprocessing import ProcessingPool as Pool


if __name__ == '__main__':
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize  
    from nltk.stem import WordNetLemmatizer
    from operator import itemgetter


    def query_inverted_index(dataframe,vocabulary):
        only_unique=dataframe.drop_duplicates(['qid']).reset_index()
        for i in range(len(only_unique)):
            only_unique['queries'][i]=[lemma.lemmatize(w) for w in word_tokenize(only_unique['queries'][i].lower()) if w.isalpha() and w not in combined_corpus]
        
        inverted_dict={}
        for i in range(len(vocabulary)):
            if vocabulary[i] not in inverted_dict.keys():
                inverted_dict.update({ vocabulary[i]:[]})
        #First creates a list object for each word as a key, that is storing passage ids
        for index in range(len(only_unique)):
            for word in only_unique['queries'][index]:
                    if word in inverted_dict.keys():
                        inverted_dict[word].append(only_unique['qid'][index])
                        
        #Then counts term frequency using a counter object 
        for keys in inverted_dict:
            inverted_dict[keys]=dict(Counter(inverted_dict[keys]))
        return inverted_dict
    
    
    def index_invert(dataframe,vocabulary):
        inverted_dict={}
        for i in range(len(vocabulary)):
            if vocabulary[i] not in inverted_dict.keys():
                inverted_dict.update({ vocabulary[i]:[]})
        only_unique=dataframe.drop_duplicates(['pid']).reset_index()
        #First creates a list object for each word as a key, that is storing passage ids
        for index in range(len(only_unique)):
            for word in only_unique['passage'][index]:
                    if word in inverted_dict.keys():
                        inverted_dict[word].append(only_unique['pid'][index])
                        
        #Then counts term frequency using a counter object 
        for keys in inverted_dict:
            inverted_dict[keys]=dict(Counter(inverted_dict[keys]))
        return inverted_dict

    def get_idf(inverted_index):
        idf={}
        for keys in inverted_index:
            check={}
            for keys_sub in inverted_index[keys]:
                check=np.log10(1+(N/len(inverted_index[keys])))
            idf.update({keys:check})
        return idf

    def return_tf(inverted_index):
        a={}
        for keys in inverted_index:
            for keys_sub in inverted_index[keys]:
                if keys_sub not in a.keys():
                    a.update({keys_sub:[[keys,inverted_index[keys][keys_sub]]]})
                else:
                    a[keys_sub].append([keys,inverted_index[keys][keys_sub]])
        for keys in a:
            sum=0
            for i in range(len(a[keys])):
                sum+= a[keys][i][-1]
            for i in range(len(a[keys])):
                a[keys][i][-1]=a[keys][i][-1]/sum
        return a


    def term_bm_passage(inverted_index):
        a={}
        for keys in inverted_index:
            for keys_sub in inverted_index[keys]:
                if keys_sub not in a.keys():
                    a.update({keys_sub:{keys:inverted_index[keys][keys_sub]}})
                else:
                    a[keys_sub].update({keys:inverted_index[keys][keys_sub]})
        return a

    def term_bm_passage_two(inverted_index):
        a={}
        for keys in inverted_index:
            for keys_sub in inverted_index[keys]:
                if keys_sub not in a.keys():
                    a.update({keys_sub:[[keys,inverted_index[keys][keys_sub]]]})
                else:
                    a[keys_sub].append([keys,inverted_index[keys][keys_sub]])
        return a

    def term_bm_query(inverted_index):
        a={}
        for keys in inverted_index:
            for keys_sub in inverted_index[keys]:
                if keys_sub not in a.keys():
                    a.update({keys_sub:[[keys,inverted_index[keys][keys_sub]]]})
                else:
                    a[keys_sub].append([keys,inverted_index[keys][keys_sub]])
        return a

    def remove_stop(dictionary,stopwords):
        words=list(dictionary.keys())
        removed=[tokens for tokens in words if tokens not in stopwords]
        return removed


    def tf_idf_query_func(term_frequency,idf):
        tf_idf={}
        for keys in term_frequency:
            for i in range(len(term_frequency[keys])):
                if keys not in tf_idf.keys():
                    tf_idf.update({keys:[[term_frequency[keys][i][0],term_frequency[keys][i][-1]*idf[term_frequency[keys][i][0]]]]})
                else:
                    tf_idf[keys].append([term_frequency[keys][i][0],term_frequency[keys][i][-1]*idf[term_frequency[keys][i][0]]])
        return tf_idf


    def bm_idf_func(inverted_index):
        dictionary={}
        for keys in inverted_index:
            check={}
            for keys_sub in inverted_index[keys]:
                check=1+np.log(((N-len(inverted_index[keys])+0.5)/(len(inverted_index[keys])+0.5)))
                dictionary.update({keys:check})
        return dictionary




    def map_passage(tf_idf_query,passage_test):
            query_to_passage={}
            for i in tf_idf_query.keys():
                query_to_passage.update({i:[]})
            for index in query_to_passage.keys():
                temp=list(passage_test[passage_test["qid"]==index]["pid"])
                for q in range(len(temp)):
                    query_to_passage[index].append(temp[q])
            return query_to_passage
    
    def document_length(frequency_table):
            length={}
            for keys in frequency_table:
                sum=0
                for i in range(len(frequency_table[keys])):
                    sum+= frequency_table[keys][i][-1]
                    length.update({keys:sum})
            return length
    
    
    def BM25_score(query_x_passage,query_dict,passage_dict,document_lengths,bm_idf):
            k1=1.2
            k2=100
            b=0.75
            avdl=sum(document_lengths.values())/len(document_lengths)
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
                            K=k1*(1-b)+((b*document_lengths[query_x_passage[keys][i]])/avdl)
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
                
                NDCG=score/optimized
                map_dict.update({keys:NDCG})
        
        ndcg_score=0
        for keys in map_dict.keys():
            ndcg_score+=map_dict[keys]
        ndcg=ndcg_score/len(map_dict)
        
        return ndcg

    passage=pd.read_csv("validation_data.tsv",sep='\t',header=0)



    passage["passage"]=passage["passage"].apply(gensim.utils.simple_preprocess)
    passage["len"]=passage["passage"].apply(lambda x: len(x))
    passage=passage[passage["len"]>0]

    appended_passage=[]
    for i in range(len(passage)):
        for j in range(len(passage["passage"][i])):
            appended_passage.append(passage["passage"][i][j])

    vocabulary=Counter(appended_passage)

    from nltk.corpus import stopwords
    from spacy.lang.en.stop_words import STOP_WORDS
    from_nltk = stopwords.words('english')
    
    combined_corpus=set.union(set(from_nltk),STOP_WORDS)
    lemma = WordNetLemmatizer()
    without_stop=remove_stop(vocabulary,combined_corpus)

    np.save("vocab_dict", vocabulary) 
    vocabulary=np.load("vocab_dict.npy",allow_pickle=True).item()

    only_unique=passage.drop_duplicates(['pid']).reset_index()
    N=len(only_unique["pid"])

    inverted_index=index_invert(passage,without_stop)
    query_index=query_inverted_index(passage,without_stop)
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
    bm_scores=BM25_score(query_to_passage,freq_que_list,freq_pas,document_lengths,bm_idf)

    bm_sorted_3=sort_queries(bm_scores,3)
    bm_sorted_10=sort_queries(bm_scores,10)
    bm_sorted_100=sort_queries(bm_scores,100)
    df_list_3=col_save(bm_sorted_3)
    df_list_10=col_save(bm_sorted_10)
    df_list_100=col_save(bm_sorted_100)

    bm25_3=pd.DataFrame(df_list_3,columns=["qid","pid","score"])
    bm25_10=pd.DataFrame(df_list_10,columns=["qid","pid","score"])
    bm25_100=pd.DataFrame(df_list_100,columns=["qid","pid","score"])

    bm_25_3=pd.merge(bm25_3,passage[["qid","pid","relevancy"]],on=["qid","pid"],how="left")
    bm_25_10=pd.merge(bm25_10,passage[["qid","pid","relevancy"]],on=["qid","pid"],how="left")
    bm_25_100=pd.merge(bm25_100,passage[["qid","pid","relevancy"]],on=["qid","pid"],how="left")

    a=passage.groupby(["qid"],as_index=False)["relevancy"].sum()

    new_dict={}
    for i in range(len(a)):
        if a["qid"][i] not in new_dict.keys():
            new_dict.update({a["qid"][i]:a["relevancy"][i]})
    
    
    map_3=mean_average_precision(bm_25_3,bm_sorted_3,new_dict,3)
    map_10=mean_average_precision(bm_25_10,bm_sorted_10,new_dict,10)
    map_100=mean_average_precision(bm_25_100,bm_sorted_100,new_dict,100)

    ndcg_3=NDCG(bm_25_3,bm_sorted_3,new_dict,3)
    ndcg_10=NDCG(bm_25_10,bm_sorted_10,new_dict,10)
    ndcg_100=NDCG(bm_25_100,bm_sorted_100,new_dict,100)

    print("The MAP at rank 3 is {}".format(map_3))
    print("The MAP at rank 10 is {}".format(map_10))
    print("The MAP at rank 100 is {}".format(map_100))

    print("The NDCG at rank 3 is {}".format(ndcg_3))
    print("The NDCG at rank 10 is {}".format(ndcg_10))
    print("The NDGC at rank 100 is {}".format(ndcg_100))
