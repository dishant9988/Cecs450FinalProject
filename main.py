import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import plotly as py
import plotly.graph_objs as go
import os
import datetime as dt
import streamlit as st
import plotly.express as px

directory = 'C:/Users/disha/OneDrive/Desktop/netflixviz/'
plt.rcParams['figure.dpi'] = 140
df = pd.read_csv(directory + 'netflix_titles.csv')


st.set_page_config(page_title="Netflix", page_icon=directory +"./assets/faceman.png", layout="centered")

# -------------Header Section------------------------------------------------

title = '<p style="text-align: center;font-size: 40px;font-weight: 550; "> Realtime Netflix Analysis</p>'
st.markdown(title, unsafe_allow_html=True)

# -------------Sidebar Section------------------------------------------------

detection_mode = None
# Haar-Cascade Parameters
minimum_neighbors = 4
# Minimum possible object size
min_object_size = (50, 50)
# bounding box thickness
bbox_thickness = 3
# bounding box color
bbox_color = (0, 255, 0)

with st.sidebar:
    st.image(directory+"./assets/faceman.png", width=260)

    title = '<p style="font-size: 25px;font-weight: 550;">Visualization </p>'
    st.markdown(title, unsafe_allow_html=True)

    # choose the mode for detection
    mode = st.radio("Choose Visualization Mode", ('Rating distribution by Film & TV Show',), index=0)

    mode = "Rating distribution by Film & TV Show"
    detection_mode = mode

if detection_mode == "Rating distribution by Film & TV Show":

    for i in df.columns:
        null_rate = df[i].isna().sum() / len(df) * 100 
        if null_rate > 0 :
            print("{} null rate: {}%".format(i,round(null_rate,2)))



    df['country'] = df['country'].fillna(df['country'].mode()[0])


    df['cast'].replace(np.nan, 'No Data',inplace  = True)
    df['director'].replace(np.nan, 'No Data',inplace  = True)

    # Drops

    df.dropna(inplace=True)

    # Drop Duplicates

    df.drop_duplicates(inplace= True)
    # print(df.info())


    df['count'] = 1

    # Ratting


    order = pd.DataFrame(df.groupby('rating')['count'].sum().sort_values(ascending=False).reset_index())
    # st.write(order)
    rating_order = list(order['rating'])
    # st.write(rating_order)
    mf = df.groupby('type')['rating'].value_counts().unstack().sort_index().fillna(0).astype(int)[rating_order]
    # st.write(mf)
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # fig, axd = plt.subplots()


    movie = mf.loc['Movie']
    tv = -mf.loc['TV Show']


    fig, ax = plt.subplots(1,1, figsize=(12, 6))
    ax.bar(movie.index, movie, width=0.5, color='#FFBD33', alpha=0.8, label='Movie')
    ax.bar(tv.index, tv, width=0.5, color='#000000', alpha=0.8, label='TV Show')
    #ax.set_ylim(-35, 50)

    # Annotations
    for i in tv.index:
        ax.annotate(f"{-tv[i]}", 
                       xy=(i, tv[i] - 60),
                       va = 'center', ha='center',fontweight='light', fontfamily='serif',
                       color='#4a4a4a')   

    for i in movie.index:
        ax.annotate(f"{movie[i]}", 
                       xy=(i, movie[i] + 60),
                       va = 'center', ha='center',fontweight='light', fontfamily='serif',
                       color='#4a4a4a')
        
     

    for s in ['top', 'left', 'right', 'bottom']:
        ax.spines[s].set_visible(False)

    ax.set_xticklabels(mf.columns, fontfamily='serif')
    ax.set_yticks([])    

    ax.legend().set_visible(False)
    fig.text(0.16, 1, 'Rating distribution by Film & TV Show', fontsize=15, fontweight='bold', fontfamily='serif')


    fig.text(0.755,0.924,"Movie", fontweight="bold", fontfamily='serif', fontsize=15, color='#FFBD33')
    fig.text(0.815,0.924,"|", fontweight="bold", fontfamily='serif', fontsize=15, color='black')
    fig.text(0.825,0.924,"TV Show", fontweight="bold", fontfamily='serif', fontsize=15, color='#221f1f')
    st.pyplot(fig)

    # plt.show()

    z = df.groupby(['rating']).size().reset_index(name='counts')
    # st.write(z)
    mdf = df.groupby('type')['rating'].value_counts().unstack().sort_index().fillna(0).astype(int)[rating_order]
    # st.write(mdf)
    Mv = mf.loc['Movie']
    tv = mf.loc['TV Show']
    mvdf=pd.DataFrame(Mv)
    tvdf=pd.DataFrame(tv)

    pieChart = px.pie(mvdf, values='Movie', names=mvdf.index, title='Rating distribution by Film ',color_discrete_sequence=px.colors.qualitative.Set3)
    # pieChart.show()
    st.plotly_chart(pieChart, theme="streamlit", use_container_width=True)


    pieChart = px.pie(tvdf, values='TV Show', names=tvdf.index, title='Rating distribution by TV Show',color_discrete_sequence=px.colors.qualitative.Set3)
    # pieChart.show()
    st.plotly_chart(pieChart, theme="streamlit", use_container_width=True)
