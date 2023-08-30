import warnings
warnings.filterwarnings('ignore')
import streamlit as st
st.title('Mall Customer Classification')
st.image('mall.jpeg')
income=st.number_input('enter yearly income in $')
income=income/1000
spend=st.number_input('enter yearly spendings in $')
spend=spend/1000
import pickle
file1=open('scale.pkl','rb')
scale=pickle.load(file1)
file2=open('model.pkl','rb')
model=pickle.load(file2)
file3=open('plot.pkl','rb')
plot=pickle.load(file3)
file4=open('centroid.pkl','rb')
centroid=pickle.load(file4)
if st.button('Predict'):
    import numpy as np
    a=[income,spend]
    x=np.array([a])
    x=scale.transform(x)
    Y_pred=model.predict(x)[0]
    st.write('Customer Type:')
    if Y_pred ==0:
        st.write('focus/target')
    elif Y_pred==1:
        st.write('sensitive')
    elif Y_pred==2:
        st.write('ignore')
    elif Y_pred==3:
        st.write('miser')
    else:
        st.write('careless')
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    sns.scatterplot(data=pd.DataFrame(plot[0]),x='INCOME',y='SPEND',color='red',label='focus/target')
    sns.scatterplot(data=pd.DataFrame(plot[1]),x='INCOME',y='SPEND',c='blue',label='sensitive/ignore')
    sns.scatterplot(data=pd.DataFrame(plot[2]),x='INCOME',y='SPEND',c='green',label='ignore')
    sns.scatterplot(data=pd.DataFrame(plot[3]),x='INCOME',y='SPEND',c='orange',label='Miser')
    sns.scatterplot(data=pd.DataFrame(plot[4]),x='INCOME',y='SPEND',c='yellow',label='Careless')
    
    #to show centroid of each categories
    sns.scatterplot(data=centroid,x=centroid[:,0],y=centroid[:,1],c='purple',marker='*',s=200,label='centroid')
    sns.scatterplot(x=[a[0]],y=[a[1]],c='black',marker='^',s=300,label='customer')
    plt.title('Classification of customer')
    plt.ylabel('Spend (x 1000)')
    #plt.xlabel(123)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
             