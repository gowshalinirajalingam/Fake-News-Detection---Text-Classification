
# coding: utf-8

# In[1]:


get_ipython().system('pip install xgboost')


# In[2]:


get_ipython().system('pip install bayesian-optimization')


# In[115]:


import pandas as pd 
import numpy as np 
import itertools 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB   #Naive bayes
from sklearn.svm import SVC  #SVM
from sklearn.linear_model import Perceptron     #NN
from sklearn.neural_network import MLPClassifier #NN
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
from sklearn.model_selection import cross_val_score  
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import matplotlib.pyplot as plt

from bayes_opt import BayesianOptimization


# In[117]:


dfpheme= pd.read_csv("pheme_dataset.csv")
dfcsv = pd.read_csv('fake_or_real_news.csv')
dfcsv=dfcsv.head(100)


# In[118]:


dfLIAR= pd.read_csv("LIAR.csv")
dfbuzzfeed= pd.read_csv("buzzfeed.csv")
dfFakeNewsNet= pd.read_csv("FakeNewsNet.csv")


# In[65]:


dfLIAR.head()


# In[66]:


dfbuzzfeed.head()


# In[67]:


dfFakeNewsNet.head()


# In[68]:


dfpheme.head()


# In[116]:


dftext=pd.DataFrame()

def createTrainTestDataset(df):
    df['label'] = df['label'].apply(lambda x:0 if x=='FAKE' else 
                           1 if x=='REAL' else 
                           None)
    global dftext
    dftext =dftext.append(df)
    
def createTrainTestDataset1(df):
    df['label'] = df['label'].apply(lambda x:0 if x=='FALSE' else 
                           1 if x=='TRUE' else 
                           None)
    global dftext
    dftext =dftext.append(df)
    

    
    


# In[4]:


createTrainTestDataset(dfpheme[['text','label']])
createTrainTestDataset(dfcsv[['text','label']])

createTrainTestDataset1(dfLIAR[['text','label']])
createTrainTestDataset(dfbuzzfeed[['text','label']])
createTrainTestDataset(dfFakeNewsNet[['text','label']])

print(dftext)


# In[71]:


dftext.shape


# In[72]:


dftext1=dftext
dftext1['label'] = dftext1['label'].apply(lambda x:'FAKE' if x==0 else 
                           'REAL' if x==1 else 
                           None)


# In[79]:


# dftext1
ax = dftext1['label'].value_counts().plot(kind='bar',
                                    figsize=(10,5),
                                    title="Distribution of fake and real news")
ax.set_xlabel("News Label")
ax.set_ylabel("Frequency")


# In[5]:


y=dftext.label
x = dftext.drop('label', axis=1)


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(dftext['text'], y, test_size=0.33, random_state=53)


# # building vectorizer classifiers
# 

# In[7]:


# Initialize the `count_vectorizer`   
# ngram_range=(1, 2) means include 1-grams and 2-grams  
count_vectorizer = CountVectorizer(stop_words='english',lowercase=True,ngram_range=(1, 2)) 

# Fit and transform the training data 
count_train = count_vectorizer.fit_transform(X_train) 

# Transform the test set 
count_test = count_vectorizer.transform(X_test)


# In[8]:


# Initialize the `tfidf_vectorizer` 
# The parameter of use_idf=True enables inverse-document-frequency reweighting by taking the log of the ratio of the 
# total number of documents to the number of documents contacting the term. And smooth_idf=True adds 1 to document 
# frequencies to avoid division by zero,so we can measure the frequency of a term among documents for 
# sure.

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7,use_idf=True ,smooth_idf=True) 

# Fit and transform the training data 
tfidf_train = tfidf_vectorizer.fit_transform(X_train) 

# Transform the test set 
tfidf_test = tfidf_vectorizer.transform(X_test)

tfidf_test


# # Algorithm

# In[9]:


dfftfidf= pd.DataFrame()
dffcnt= pd.DataFrame()


def modeltfidf(model,modelname):
    model.fit(tfidf_train, y_train)
    pred = model.predict(tfidf_test)
    score = metrics.accuracy_score(y_test, pred)
    all_accuracies = cross_val_score(estimator=model, X=tfidf_train, y=y_train, cv=5) 
    print("accuracy:   %0.3f" % score)
    print("cross validation score: {}".format(all_accuracies.mean()))
    print('Precision Score : ' + str(precision_score(y_test,pred)))
    print('Recall Score : ' + str(recall_score(y_test,pred)))
    print('F1 Score : ' + str(f1_score(y_test,pred)))
    
    savedf1=  pd.DataFrame({'pred':pred})
    savedf1 = savedf1.reset_index(drop=True)

    savedf2 = pd.DataFrame(y_test[::1])
    savedf2 = savedf2.reset_index(drop=True)   # drop old index column and create new index column
    
    finaldf = savedf2.join(savedf1)
    finaldf = finaldf.rename(columns ={'label':'actual'})
#     print(finaldf)
    finaldf['model'] = modelname
    
    #Predict probability
    probs = model.predict_proba(tfidf_test)       # predict probabilities
    # keep the predictions for class 1 only
    probs = probs[:, 1]
    len(probs)
    probdf = pd.DataFrame(probs)
    probdf = probdf.rename(columns={0:'prob'})
#     print(probdf)
    aaa = finaldf.join(probdf)
    
    global dfftfidf
    dfftfidf = dfftfidf.append(aaa)
    
def modelbagofwords(model,modelname):
    model.fit(count_train, y_train)
    pred = model.predict(count_test)
    score = metrics.accuracy_score(y_test, pred)
    all_accuracies = cross_val_score(estimator=model, X=count_train, y=y_train, cv=5) 

    print("accuracy:   %0.3f" % score)
    print("cross validation score: {}".format(all_accuracies.mean()))
    print('Precision Score : ' + str(precision_score(y_test,pred)))
    print('Recall Score : ' + str(recall_score(y_test,pred)))
    print('F1 Score : ' + str(f1_score(y_test,pred)))
    
    savedf1=  pd.DataFrame({'pred':pred})
    savedf1 = savedf1.reset_index(drop=True)

    savedf2 = pd.DataFrame(y_test[::1])
    savedf2 = savedf2.reset_index(drop=True)   # drop old index column and create new index column
    
    finaldf = savedf2.join(savedf1)
    finaldf = finaldf.rename(columns ={'label':'actual'})
#     print(finaldf)
    finaldf['model'] = modelname
    
    #Predict probability
    probs = model.predict_proba(count_test)       # predict probabilities
    # keep the predictions for class 1 only
    probs = probs[:, 1]
    len(probs)
    probdf = pd.DataFrame(probs)
    probdf = probdf.rename(columns={0:'prob'})
#     print(probdf)
    aaa = finaldf.join(probdf)
    
    global dffcnt
    dffcnt = dffcnt.append(aaa)
  


# ## Naive_Bayes
# -----------------------
# 
# #### The multinomial Naive Bayes classifier is suitable for classification with discrete features (e.g., word counts for text classification).

# In[21]:


nb = MultinomialNB()


# In[22]:


modeltfidf(nb,"Naive_Bayes")
print(dfftfidf)


# In[23]:


modelbagofwords(nb,"Naive_Bayes")
print(dffcnt)


# #### Hyper parameter tuning

# In[24]:


from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


# In[25]:


nb.get_params().keys()


# In[26]:


pipelineNB = Pipeline((
    ('clf',MultinomialNB()),
    ))
    
parameterNB={
        'clf__alpha': [0.022,0.025, 0.028],
#         'vect__ngram_range': [(1, 1), (1, 2),(1,3)],
#         'vect__max_df': [ 0.7,0.8,0.9,1.0],
#         'vect__min_df': [1,2],

    }
    
gridnbtfidf = GridSearchCV(pipelineNB,parameterNB,scoring='accuracy', verbose=1)
gridnbcnt = GridSearchCV(pipelineNB,parameterNB,scoring='accuracy', verbose=1)


# In[27]:


gridnbtfidf.fit(tfidf_train,y_train)


# In[28]:



gridnbcnt.fit(count_train,y_train)


# In[29]:


gridnbtfidf.best_params_


# In[30]:


gridnbcnt.best_params_


# In[31]:


gridnbtfidf.best_score_


# In[32]:


gridnbcnt.best_score_


# In[33]:


modeltfidf(MultinomialNB(alpha=0.028),"Naive_bayes_hyper_PT")
print(dfftfidf)


# In[34]:


modelbagofwords(MultinomialNB(alpha=0.022),"Naive_bayes_hyper_PT")
print(dffcnt)


# ## Logistic Regression
# -------------------------------

# In[35]:


lg = LogisticRegression()


# In[36]:


lg


# In[38]:


modeltfidf(lg,"Logistic_Regression")


# In[39]:


modelbagofwords(lg,"Logistic_Regression")


# #### Hyper parameter tuning

# In[40]:


lg.get_params().keys()


# In[41]:


pipelineLG = Pipeline((
    ('lr',LogisticRegression()),
    ))
    
parameterLG={
      'lr__penalty': ['l1', 'l2'],
      'lr__C': [1, 5, 10],
      'lr__max_iter': [20, 50, 100]
    }
    
gridlgtfidf = GridSearchCV(pipelineLG,parameterLG,scoring='accuracy', verbose=1)
gridlgcnt = GridSearchCV(pipelineLG,parameterLG,scoring='accuracy', verbose=1)




# In[42]:


gridlgtfidf.fit(tfidf_train,y_train)


# In[43]:


gridlgcnt.fit(count_train,y_train)


# In[44]:


gridlgtfidf.best_params_


# In[45]:


gridlgcnt.best_params_


# In[46]:


gridlgtfidf.best_score_


# In[47]:


gridlgcnt.best_score_


# In[49]:


modeltfidf(LogisticRegression(C=10, max_iter= 20, penalty='l2'),"LogisticRegression_hyper_PT")


# In[50]:


modelbagofwords(LogisticRegression(C=1, max_iter= 20, penalty='l2'),"LogisticRegression_hyper_PT")


# ## XGBoost
# ---------------------------

# In[10]:


xg = XGBClassifier()


# In[53]:


modeltfidf(xg,"XGBoost")


# In[54]:


modelbagofwords(xg,"XGBoost")


# #### Hyper parameter tuning
# bayesian optimization:https://medium.com/spikelab/hyperparameter-optimization-using-bayesian-optimization-f1f393dcd36d

# In[11]:


#reference:  https://www.kaggle.com/btyuhas/bayesian-optimization-with-xgboost
import xgboost as xgb


dtraintfidf = xgb.DMatrix(tfidf_train, label=y_train)
dtraincnt= xgb.DMatrix(count_train, label=y_train)
dtest = xgb.DMatrix(tfidf_test)


# In[56]:


dtraintfidf


# In[57]:


def train_model_xg_tfidf(max_depth, 
                ntrees,
                min_rows, 
                learn_rate, 
                sample_rate, 
                col_sample_rate):
    params = {
        'max_depth': int(max_depth),
        'ntrees': int(ntrees),
        'min_rows': int(min_rows),
        'learn_rate':learn_rate,
        'sample_rate':sample_rate,
        'col_sample_rate':col_sample_rate
    }
#     model = XGBClassifier(nfolds=5,**params)
#     model.fit(count_train, y_train)
    cv_result = xgb.cv(params, dtraintfidf, num_boost_round=100, nfold=3)    

    # Bayesian optimization only knows how to maximize, not minimize, so return the negative RMSE
    return -1.0 * cv_result['test-rmse-mean'].iloc[-1]


# In[12]:


def train_model_xg_cnt(max_depth, 
                ntrees,
                min_rows, 
                learn_rate, 
                sample_rate, 
                col_sample_rate):
    params = {
        'max_depth': int(max_depth),
        'ntrees': int(ntrees),
        'min_rows': int(min_rows),
        'learn_rate':learn_rate,
        'sample_rate':sample_rate,
        'col_sample_rate':col_sample_rate
    }
#     model = XGBClassifier(nfolds=5,**params)
#     model.fit(count_train, y_train)
    cv_result = xgb.cv(params, dtraincnt, num_boost_round=100, nfold=3)    

    # Bayesian optimization only knows how to maximize, not minimize, so return the negative RMSE
    return -1.0 * cv_result['test-rmse-mean'].iloc[-1]


# In[13]:


bounds = {
    'max_depth':(5,10),
    'ntrees': (100,500),
    'min_rows':(10,30),
    'learn_rate':(0.001, 0.01),
    'sample_rate':(0.5,0.8),
    'col_sample_rate':(0.5,0.8)
}


# In[60]:


optimizertfidf = BayesianOptimization(
    f=train_model_xg_tfidf,
    pbounds=bounds,
    random_state=1,
)

optimizertfidf.maximize(init_points=10, n_iter=2)


# In[14]:


optimizercnt = BayesianOptimization(
    f=train_model_xg_cnt,
    pbounds=bounds,
    random_state=1,
)

optimizercnt.maximize(init_points=10, n_iter=2)


# In[15]:


# paramtfidf = optimizertfidf.max
paramcnt = optimizercnt.max


# In[16]:


# xgmodeltfidf = xgb.train(paramtfidf, dtraintfidf, num_boost_round=100)
xgmodelcnt = xgb.train(paramcnt, dtraincnt, num_boost_round=100)


# In[64]:


# Predict on testing and training set
y_predtfidf = xgmodeltfidf.predict(dtest)
y_predtfidf_df = pd.DataFrame(data={'y':y_predtfidf}) 
y_predtfidf_df.round(0)
score = metrics.accuracy_score(y_test, y_predtfidf_df.round(0))
score


# In[17]:


# Predict on testing and training set
y_predcnt = xgmodelcnt.predict(dtest)
y_predcnt_df = pd.DataFrame(data={'y':y_predcnt}) 
y_predcnt_df.round(0)
score = metrics.accuracy_score(y_test, y_predcnt_df.round(0))
score


# In[84]:


#Predict probability
savedf1 = y_predtfidf_df.round(0).reset_index(drop=True)  #predicted value
# savedf1

savedf2 = pd.DataFrame(y_test[::1])
# savedf2
savedf2 = savedf2.reset_index(drop=True)   # drop old index column and create new index column
    
finaldf = savedf2.join(savedf1)
finaldf = finaldf.rename(columns ={'label':'actual','y':'pred'})
# #     print(finaldf)
finaldf['model'] = "XGBoost_hyper_PT"
# finaldf
# y_predtfidf_df
aaa = finaldf.join(y_predtfidf_df["y"].rename("prob"))
aaa
    
dfftfidf = dfftfidf.append(aaa)


# In[85]:


dfftfidf.to_csv("uptoxgtfidf.csv")


# In[87]:


# # Predict on testing and training set
# y_predcnt = xgmodelcnt.predict(dtest)
# y_predcnt_df = pd.DataFrame(data={'y':y_predcnt}) 
# y_predcnt_df.round(0)
# score = metrics.accuracy_score(y_test, y_predcnt_df.round(0))
# score


# ## SVM

# In[91]:


# svmm = SGDClassifier(loss='log', penalty='l2',alpha=1e-3, n_iter=5, random_state=42)
svm = SVC(probability=True)


# In[92]:


modeltfidf(svm,"SVM")


# In[92]:


modelbagofwords(svm,"SVM")


# #### Hyper parameter tuning

# In[94]:


pipelinesvc = Pipeline((
    ('clf',SVC()),
    ))
    
parametersvc={
        'clf__C' : [0.01,0.1,1.0],
#         'clf__kernel':['rbf','poly'],
        'clf__gamma' : [0.01,0.1,1.0],
#           'clf__kernel':('linear', 'rbf'),
#           'clf__C':(1,0.25,0.5,0.75),
#           'clf__gamma': (1,2,3,'auto'),
#           'clf__decision_function_shape':('ovo','ovr'),
#           'clf__shrinking':(True,False),
    }
    
   
gridsvmtfidf = GridSearchCV(pipelinesvc,parametersvc,scoring='accuracy', verbose=1)
gridsvmtfidf.fit(tfidf_train,y_train)


# In[95]:


gridsvmcnt = GridSearchCV(pipelinesvc,parametersvc,scoring='accuracy', verbose=1)
gridsvmcnt.fit(count_train,y_train)


# In[19]:


gridsvmtfidf.best_params_


# In[96]:


gridsvmcnt.best_params_


# In[97]:


gridsvmtfidf.best_score_


# In[98]:


gridsvmcnt.best_score_


# In[101]:


modeltfidf(SVC(C=1,gamma=1,probability=True),"SVM_hyper_PT")


# In[104]:


modelbagofwords(SVC(C=1,gamma=0.01,probability=True),"SVM_hyper_PT")


# ## PassiveAggressiveclassifier

# In[ ]:


pa = PassiveAggressiveClassifier(max_iter=50)


# In[ ]:


modeltfidf(pa)


# In[ ]:


modelbagofwords(pa)


#  ## Neural Network

# In[105]:


mlp = MLPClassifier(max_iter=20)


# In[106]:


modeltfidf(mlp,"Neural Network")
print(dfftfidf)


# In[6]:


modelbagofwords(mlp,"Neural Network")
print(dffcnt)


# In[143]:


print(dfftfidf)
dfftfidf.to_csv("uptonaivebayes.csv")


# #### Hyper parameter tuning

# In[146]:


pipelineNN = Pipeline((
    ('clf',MLPClassifier()),
    ))
    
parameterNN={
#     'clf__hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
#     'clf__activation': ['tanh', 'relu'],
    'clf__hidden_layer_sizes': [(50,50,50), (50,100,50)],
    'clf__alpha': [0.0001, 0.05],
    'clf__learning_rate': ['constant','adaptive'],
    'clf__random_state':[1],
    'clf__max_iter':[20],

    }
# activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
#        beta_2=0.999, early_stopping=False, epsilon=1e-08,
#        hidden_layer_sizes=(100,), learning_rate='constant',
#        learning_rate_init=0.001, max_iter=200, momentum=0.9,
#        nesterovs_momentum=True, power_t=0.5, random_state=None,
#        shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
#        verbose=False, warm_start=Fals
    
    
gridNNtfidf = GridSearchCV(pipelineNN,parameterNN,scoring='accuracy', verbose=1)


# In[147]:


gridNNtfidf.fit(tfidf_train,y_train)


# In[ ]:


gridNNcnt = GridSearchCV(pipelineNN,parameterNN,scoring='accuracy', verbose=1)
gridNNcnt.fit(count_train,y_train)


# In[ ]:


gridNNtfidf.best_params_


# In[ ]:


gridNNcnt.best_params_


# In[ ]:


gridNNtfidf.best_score_


# In[ ]:


gridNNcnt.best_score_


# In[ ]:


modeltfidf(MLPClassifier(),"Neural Network_Hyper_PT")


# In[ ]:


modelbagofwords(MLPClassifier(),"Neural Network_Hyper_PT")


# In[19]:


0.86578421654

