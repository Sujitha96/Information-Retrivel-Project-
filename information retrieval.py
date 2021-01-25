#!/usr/bin/env python
# coding: utf-8

# In[209]:


f1=open(r'C:\Users\Sujji\OneDrive\Desktop\Information retrieval ranking\Information Retrieval Ranking Using Machine (2)\Information Retrieval Ranking Using Machine\CODE/information retrieval.txt','r')
text = f1.read()
print(text)


# In[210]:


text=text.lower()


# In[175]:


import re
data=[]
for i in re.split(r'[#$]',text,flags=re.I):
    data.append(i)


# In[176]:


list_t=[]
for i in data:
    list_t.append(re.sub(r'[0-9.0-9Q:()]+','',i,flags=re.I))


# In[177]:


import pandas as pd
df = pd.DataFrame(list_t)
df


# In[178]:


df=df.drop(df.index[0])


# In[179]:


#from nltk.tokenize import sent_tokenize
#token=sent_tokenize(data)
#print(token)


# In[180]:


from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
var=set(stopwords.words('english'))
print(var)


# In[181]:


filtered_words=[]
for i in list_t:
    if i not in var:
        filtered_words.append(i)
#print(filtered_words)        


# In[182]:


filtered_words


# In[183]:


from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
from nltk.stem import PorterStemmer
lst=[]
for i in filtered_words:
    tokens=word_tokenize(i)
    #lst.append(i)
    s=' '
    for i in tokens:
        ps=PorterStemmer()
        s+=ps.stem(i)
        s+=' '
        #print(var)
    lst.append(s)

#print(str)        
        
    


# In[184]:


lst


# In[185]:


df=pd.DataFrame(lst,columns=['text'])


# In[186]:


df


# In[187]:


from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer()
tf.fit(df['text'])
text_tf = tf.transform(df['text'])


# In[188]:


text_tf


# In[189]:


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
kmeans = KMeans(n_clusters=2,init='k-means++',max_iter=300,n_init=10,random_state=0)
y=kmeans.fit_predict(text_tf)
#visualization
#print(y)


# In[190]:


y


# In[191]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(text_tf,y,test_size=0.3)
#plt.scatter(x_train[x == 0, 0], x_train[x == 0,1],s=100,c='red',label='cluster1')
#plt.scatter(x_train[x == 1, 0], x_train[x == 1,1],s=100,c='blue',label='cluster2')
#plt.scatter(x_train[x == 2, 0], x_train[x == 2,1],s=100,c='green',label='cluster3')
#plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='centroids')


# In[192]:


y


# In[193]:


import warnings
warnings.filterwarnings('ignore')
from sklearn.svm import SVC
test1 = SVC(kernel='linear')
test1.fit(x_train,y_train)
pred=test1.predict(x_test)


# In[194]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[195]:


accuracy_score(y_test,pred)


# In[196]:


#query=input('enter a query?')
query=('which cities have crip gangs, what are the latest movies,what are the priceses of electronic devices in hyderabad')
query
#predict_query=test1.predict(query)


# In[197]:


from nltk.tokenize import word_tokenize
token = word_tokenize(query)

#token=word_tokenize(query)


# In[198]:


filter_sent=[]
for w in token:
    if w not in var:
        filter_sent.append(w)
print(filter_sent)        
        


# In[199]:


from nltk.stem import PorterStemmer
ps=PorterStemmer()
j=' '
for i in filter_sent:
    j+=ps.stem(i)
    j+=' '

    


# In[200]:


j


# In[201]:


test = tf.transform([j])
test


# In[205]:


x_train.shape


# In[206]:


X1=test1.predict(test)


# In[207]:


X1


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




