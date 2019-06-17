import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
import unicodedata
import re, numpy, io, time
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from matplotlib import pyplot
from sklearn.preprocessing import Normalizer
from multiprocessing import Pool, cpu_count
from scipy import stats
import csv
#######################################################################################################################
## This file contains function read in the data, write the result into files and functions for producing tfidf matrix.#
## And functions for analysing word features.                                                                         #
#######################################################################################################################

#step 1: read data: from csv file or database
def get_data(strarg, method='csv'):
    """
    read in data that is on the path, microsoftSQL generate csv with speration ';'
    """
    if method == 'csv':
        #if the method is csv, the string is the path of the file
        data = pd.read_csv(strarg, sep=";", header=None, names=['text', 'title', 'yrke_id', 'ocupation_id', 'ssyk'])
    elif method =='sql':
        #if the method is sql, the string is the sql statement
        import sqlalchemy as sa
        engine = sa.create_engine('mssql+pyodbc://[ServerName]\\[Path]?driver=SQL+Server')
        data = pd.read_sql(strarg, engine)

    #print(data.head())
    return data

def clean_string(stringlist):
    # the function cleans a string 
    # the function is used for handling column storing stings in dataframe
    stpwds = stopwords.words("swedish")
    # add kommun & ort into the stopwords

    stw = pd.read_csv("[Path to the csv file", sep=";", names=["namn"])
    # add more stopwords from the csv file into the standard package
    stpwds.extend(stw['namn'].str.lower().values)
    #print(stpwds[1:10])
    #map the lambda function on the input: stringlist, replace the non-alphabetical characters
    sl = [re.sub("[\W|_|\d]+", " ", s.lower()) for s in stringlist]
    #map the function on sl, and return strings
    new_sl = [" ".join(t for t in word_tokenize(s) if t not in stpwds) for s in sl]
    return new_sl

def clean_data(df):
    """
    Text data is stored in a column of the dataframe, the function deletes the stopwords, clean away the non-alphabets characters and stemming the words
    Return:
        a list of cleaned text
    """
    #clean text: delete non-alphabetswords and stemming

    #stemming is not working well, e.g. Mora->mor
    #stemmer = SnowballStemmer("swedish")
    #split the string list into  7 partitions to run parallell process
    textlist = df['text'].values
    text_split = numpy.split(textlist, [6450, 12900, 19440, 25890, 32340,38790, 45240, 51690, 58140, 64590, 71040, 77400, 77490])
    pool = Pool(14)
    #data is a list of string
    data = numpy.concatenate(pool.map(clean_string, text_split))
    pool.close()
    pool.join()
    #text = data["text"].apply(lambda x: unicodedata.normalize('NFKD', x).encode('ascii', 'ignore')).tolist()""
    #use the regular expression delte all the non-alphanumeric character and the strange characters, use or |
    """
    df['text'] = df['text'].apply(lambda x: re.sub("[\W|_|\d]+", " ", x.lower()))

    df['text']= df['text'].apply(lambda x: " ".join(w for w in word_tokenize(x) if w not in stpwds))
    #iterate the list of text split the words and delete the space and stemming the word
     #it took several minutes handlings 90739 rows
        text = df['text'].apply(lambda x: re.sub("[\W]|_|\d+", " ", x.lower())).tolist()
         new_text = []
        for t in text:
        #t is a string, split it into a list and check the stopwords and stemming, add stemmer.stem(nt) if stemming is needed
        clean_token = " ".join(nt for nt in nltk.tokenize.word_tokenize(t) if nt not in stpwds)
        new_text.append(clean_token) """

    #print(len(new_text))
    """print(new_text[2])
    print(new_text[1])
    print(new_text[0]) """
    # a list of cleaned data
    return data

def write_data(path, d, type='dict'):
    """
    The function write the data either in matrix or dict form into a file.
    Arguments:
        path: the file path 
        d: the data written into the file
        type: the default type is dict, it can be changed into matrix or arrays

    """ 
    #convert the data into dataframe
    if(type=='matrix'):
        df = pd.DataFrame(data=d)
        df.to_csv(path, sep=",")
    elif(type=='dict'):
        import json
        #write swedish text to file
        with io.open(path, 'w', encoding="iso-8859-1") as file:
            json.dump(d, file, ensure_ascii=False)
    else:
        print("please give the type of writing, default is dict, the other choice is matrix")
    
    print("finish the writing!")


def data_vector(corpus):
    """
    The function takes the text corpus and vectorize the corpus.
    return tfidf-matrix
    """
    #use_idf: default True; smooth_idf: default True
    vectorizer = TfidfVectorizer(use_idf=True)
    svd_model = TruncatedSVD(algorithm='randomized', n_iter=10)

    #svd_transformer = Pipeline([('tfidf', vectocorpus
    #svd_matrix = svd_transformer.fit_transform(corpus)
    dtm = vectorizer.fit_transform(corpus)
    #(90739, 125580)
    print(dtm.toarray().shape)
    #get the attributes' values, the weights of each word
    idf = vectorizer.idf_
    dtm_dict = dict(zip(vectorizer.get_feature_names(), idf))
    #print(vectorizer.get_feature_names())
    

    #midwife
    write_data("[PathtoFile", dtm_dict)
    #write_data("C:\\Users\\TSTDAWU\\Documents\\results\\arbetskraftbarometern\\titles_features_tfidf_matrix.csv", dtm.toarray().transpose(), type="matrix")
    #the file is too big , use pickle
    pd.DataFrame(dtm.toarray()).to_pickle("[PathtoFile")
    # analyse_features(vectorizer.get_feature_names(), idf)
    #svd_matrix = svd_model.fit_transform(dtm)

    # print(svd_matrix)
    #return svd_matrix
    #lsa = Normalizer(copy=False).fit_transform(svd_matrix)
    #result = pd.DataFrame(svd_matrix, index=["c1", "c2"], #columns=vectorizer.get_feature_names())
    #print("**********components**************")
    #print(svd_model.components_)
    #print("************explained_variance*******************")
    #print(svd_model.explained_variance_)
    #print("**************explained_variance_ratio****************")
    #print(svd_model.explained_variance_ratio_)
    #return svd_model

def words_counts_matrix(corpus, an="word",max=90000, min=50):
    """
    The function creates the matrix by counts of words
    Arguments:
    an: defined as analyzer, can choose n-gram for example
    max: defined as max_df: ignored when the word frequences is higher than the threshold
    min: defined as min_df, minimum frequences for considering in the corpus
    For more information, read:
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    """
    vectorizer = CountVectorizer(analyzer=an, max_df=max, min_df=min)
    #other parameters: max_df, min_df, max_features
    X = vectorizer.fit_transform(corpus)
    #print down the result
    feature_string = ",".join([w for w in vectorizer.get_feature_names()])
    #print the feature_names into a file
    #with io.open("H:\\BitBucket\\results\\arbetskraftbarometern\\features_counts_%i_%i.csv"%(max, min), "w", encoding = "utf-8") as f:

    with io.open("PathtoFile"%(max, min), "w", encoding="utf-8") as f:
        f.write(feature_string)
   
    return X


def analyse_features(keys, values):
    #do further analysis on the features and tfidf values
    #put first the words-keys and tfidf-values into a dataframe
    #the function is running in the data_vector function, because I cannot read the data from a json file into dataframe
    df = pd.DataFrame()
    df['words'] = keys
    df['values'] = values
    #write the statistical values of idf
    print(df.describe())
    print(df.head())
    print("-------feature analysis------------")
    #läkare=9.0
    #kindergarten=10.8
    #midwife=7.9
    rare_words = df.loc[df['values']>7.9]
    rare_words = rare_words.sort_values(by=['values'])
    nine_words = df.loc[df['values']==7.9]
    common_words = df.loc[df['values']<7.9 ]
    common_words = common_words.sort_values(by=['values'])
    print("total features: %d"%len(keys))
    print("rare_words shape %d, %d"%(rare_words.shape))
    print("example of rare_word")
    #print(rare_words)

    write_data("[PathtoFile]", rare_words, type="matrix" )
    print("common_words shape %d, %d"%(common_words.shape))
    print("examples of common_words")
    #print(common_words)
    write_data("[FiletoPath", common_words, type="matrix" )
    print("Nine words")
    print(nine_words)



if __name__ == "__main__":
    # step 1: read in the data
    ## run once retrieving data and cleaning data, save the result into a file
    #sql = r"select [Column] as text, [Column] from [Table]"
    #time_start = time.clock()

    #data = get_data(sql, method='sql')
    #print (data.head())
    #print(data.columns)
    #print(data.shape)
    #time_end_1 = time.clock()
    ## time is 18.9 s, running with the python environment definition on tsta152
    ## time is 6.4 ~ 23 s, runing with "Start without debugging"
    #print("_______data retrieving time is %f__________"% (time_end_1-time_start))
    
    ##clean_data, take a df, return a string list
    #data['text'] = clean_data(data)
    #time_end = time.clock()
    ## time is tooooo long, the process is killed by the user
    ## time is 800 ~ 917s, runing with "start without debugging"
    #print("-----------the time of text cleaning is %f----------"%(time_end-time_end_1))
    ## print(len(text))
    #print(data.head())
   
    data = pd.read_csv("FiletoPath", encoding="utf-8")
    #print("---data written into a csv file---")
    print(data.head())
    print(data.shape)
    
    #check how many words in each ads
    #data['words'] = data['text'].apply(lambda x: len(str(x).split(" ")))
    #print("--------------------")
    #print(data.describe())
    #print(data.groupby(['words']).describe())
    #"""
    #         Unnamed: 0         words
    #count  90739.000000  90739.000000
    #mean   45369.000000    170.610421
    #std    26194.237375     81.083178
    #min        0.000000      1.000000
    #25%    22684.500000    110.000000
    #50%    45369.000000    165.000000
    #75%    68053.500000    224.000000
    #max    90738.000000    888.000000
    #"""
    # model = data_vector(text)
    # step 2: make matrix based on words counts or words tfidf
    corpus = data['text'].values.astype('U')
    X = words_counts_matrix(corpus)
    #print("wirte the features into a file")
    #print(X.shape)
    # time_start = time.clock()
    #words_counter=corpus_word_count(corpus)
    #time_end = time.clock()
    # time spended 161 s
    #print("the execution time on corpus counter is %f"% (time_end-time_start))
    #unique words: 15481019
    #print("---total unique words %i---" % len(words_counter.keys()))
    #s = pd.Series(list(words_counter.values()))
    #print("series shape is %i" % s.shape)
   # print("--------describe the words counts-----")
    #print(s.describe())
    #print("------The moste common words-----------")
    #write_data("[PathtoFile]", words_counter.most_common(100))
    #print("--------The rare words---------------")
    #rare_words = {key:value for key, value in words_counter.items() if value<31}
    #write_data("[PathtoFile]", rare_words)
    #get the tfidf on corpus
    #data_vector(corpus)
    #time_end = time.clock()
    #print("finish the tfidf matrix in %f seconds"% (time_end-time_start))


    # explained_variance = model.explained_variance_ratio_.sum()
    # print(format(int(explained_variance * 100)))
