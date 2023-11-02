# ***********************************************************************************************
# -*- coding: utf-8 -*-
#
# Title: Bankruptcy Prediction
# Course & Group : PGP in AI & ML - AIML2022 Cohort 8, Group 10
# Subject: Capstone Project - PCAMZC321
# Description: This is the python script used to start the Bankruptcy Prediction Web Application
#
# Lead Instructor:
# Mr. Satyaki Dasgupta
#
# Student Names :
## Radha Surendra Kukadapwar   | 2022AIML044
## Mitul Navalkishor Desai     | 2022AIML055
## Pawan Kumawat               | 2022AIML043
## Srikanth Pathapalli         | 2022AIML017
#
# ***********************************************************************************************
#
# Steps to start this GUI
# 1) Pre-requisite : Make sure to install streamlit python module using "pip install streamlit"
# 2) Open anaconda command prompt or anaconda power shell
# 3) Change the current working directory to the path where this Bankruptcy_Predictor.py file is located
# 4) Launch the GUI using "streamlit run Bankruptcy_Predictor.py" command
#
# ***********************************************************************************************

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components
import io

import os
from os.path import exists
import matplotlib.image  as mpimg
import base64
import plotly.figure_factory as figFctry

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import RFECV, mutual_info_classif

from sklearn                   import preprocessing
from sklearn.preprocessing     import MinMaxScaler, StandardScaler 
from sklearn.linear_model      import LogisticRegression
from sklearn.tree              import DecisionTreeClassifier
from sklearn.ensemble          import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.ensemble          import BaggingClassifier, ExtraTreesClassifier, IsolationForest 
from sklearn.neighbors         import KNeighborsClassifier 
from xgboost                   import XGBClassifier
from lightgbm                  import LGBMClassifier

from matplotlib.lines import Line2D
from scipy import stats
from sklearn.model_selection import train_test_split 

#---------------------------------------------------
st.set_page_config(layout="wide")
#---------------------------------------------------

def OnFileUpload(uploaded_file):
    with st.spinner('Please wait...Reading Input File...'):
        data = pd.read_csv(uploaded_file) 
    return data

def getBasicStatistics(data, tab_name):

    #tab_name.success('INPUT DATA AT A GLANCE...')
    
    col1, col2, col3, col4 = tab_name.columns([0.25, 0.25, 0.25, 0.25]) 

    con1 = col1.container()
    con1.markdown("<div class='myContainer'><div class='txth2'>SAMPLES<br>" + str(data.shape[0]) + "</div></div>", unsafe_allow_html=True)  
    
    con2 = col2.container()
    con2.markdown("<div class='myContainer'><div class='txth2'>FEATURES<br>" + str(data.shape[1]) + "</div></div>", unsafe_allow_html=True)
    
    con3 = col3.container()
    con3.markdown("<div class='myContainer'><div class='txth2'>MISSING VALUES?<br>" + str(data.isnull().values.any()) +
                  "</div></div>", unsafe_allow_html=True)
    
    con4 = col4.container()
    con4.markdown("<div class='myContainer'><div class='txth2'>DUPLICATES?<br>" + str(data.duplicated().values.any()) +
                  "</div></div>", unsafe_allow_html=True) 
    
    tab_name.markdown("<br>", unsafe_allow_html=True)
    tab_name.info("The first 5 rows")
    tab_name.write(data.head(5))
    
    tab_name.info("The last 5 rows")
    tab_name.write(data.tail(5))  
    
    tab_name.info("Statistical Distribution Of Data")
    tab_name.write(data.describe())

def myDataframeInfo(df):
    
    buffer = io.StringIO()
    df.info(buf=buffer)    
    return buffer.getvalue()  

def predictMyModel(modelFileName, my_red_df, plt):   
    
    my_file_name = './PredictionModels/' + modelFileName
    
    X_features = my_red_df.drop(columns='Bankrupt?')
    y_true = my_red_df['Bankrupt?']
    
    pklModel = pickle.load(open(my_file_name, 'rb'))
    
    y_pred = pklModel.predict(X_features)
    
    #=================================

    acrcy_scr = accuracy_score(y_true, y_pred)
    clsf_rep = classification_report(y_true, y_pred, output_dict=True)
    conf_mtrx = confusion_matrix(y_true, y_pred)
    df_clsf_rep = pd.DataFrame(clsf_rep).transpose()
    
    #=================================
    
    f_conf_mtrx, ax = plt.subplots(figsize=(5, 5)) 
    sns.heatmap(conf_mtrx, square=True, annot=True, fmt="d", cmap="RdYlGn")   

    return y_pred, acrcy_scr, df_clsf_rep, f_conf_mtrx
    
def scaleData(X_features, type):
    # Input is X_features
    if(type == 'MinMax'):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(X_features)
        X_features_scaled = pd.DataFrame(scaled_data, columns=X_features.columns)

    elif(type == 'Standard'):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(X_features)
        X_features_scaled = pd.DataFrame(scaled_data, columns=X_features.columns) 
    
    return X_features_scaled


def t20miClassif(df):
    # Input dataset is df_reduced_cols | Top 20 features using Mutual Info Classification
    features_20_best_viz = pd.DataFrame(columns=['fName', 'fScore'])
    
    X_viz = df.drop(columns='Bankrupt?')
    y_viz = df['Bankrupt?']  
    
    feature_scores_viz = mutual_info_classif(X_viz, y_viz, random_state=0)
    
    for score, f_name in sorted(zip(feature_scores_viz, X_viz.columns), reverse=True): 
        features_20_best_viz = features_20_best_viz.append({'fName': f_name, 'fScore': score}, ignore_index = True)
    
    # plot feature importance
    fig_mi, axs = plt.subplots(figsize = (15,15))
    barColor = ["blue" if i < (features_20_best_viz.fScore.iloc[19]) else "green" for i in features_20_best_viz.fScore] 
    sns.barplot(x=features_20_best_viz.fScore, y=features_20_best_viz.fName, data=features_20_best_viz, palette=barColor, ax=axs) 
    custom_lines = [Line2D([0], [0], color='green', lw=4),
                    Line2D([0], [0], color='blue', lw=4)]
    legend = axs.legend(custom_lines, ['20 Best Features', 'Remaining Features'], loc='center right', shadow=True, fontsize='large')
    legend.get_frame().set_facecolor('#FFFFFF')
    
    plt.title('20 Best Features - Mutual Info Classifier',fontsize=15)
    plt.xlabel('Feature Score',fontsize=12)
    plt.ylabel('Column Name',fontsize=12)  
    
    # Top 20 column names based on features importance using logistic regression
    X_viz20 = X_viz[features_20_best_viz.fName[:20]]
    return (X_viz20.join(y_viz)), fig_mi

def split_df(X_features, y_target, testSize, scaleFlag):
    if(scaleFlag=='Yes'):
        # Scale X features in both train & test
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(X_features)
        X_features = pd.DataFrame(scaled_data, columns=X_features.columns) 
        
    X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=testSize, random_state=42) 
    
    return X_train, X_test, y_train, y_test
    
def drawBoxPlot(df, plt, cust_palette, clr_line):
    # Boxplot for all the features     
    fig_boxPlt, axes = plt.subplots(figsize = (20,40))
    ax = sns.boxplot(data = df, orient="h", palette=np.flip(cust_palette)) 
    ax.set(xscale="log")

    # Lines & outlier colors customizations
    box_patches = [patch for patch in ax.patches if type(patch) == matplotlib.patches.PathPatch]
    if len(box_patches) == 0:
        # in matplotlib older than 3.5, the boxes are stored in ax2.artists
        box_patches = ax.artists
        
    num_patches = len(box_patches)
    lines_per_boxplot = len(ax.lines) // num_patches
    
    for i, patch in enumerate(box_patches): 
        patch.set_edgecolor(clr_line)  
        
        for line in ax.lines[i * lines_per_boxplot: (i + 1) * lines_per_boxplot]: 
            line.set_color(clr_line)
            line.set_mfc(clr_line)  # facecolor of fliers
            line.set_mec(clr_line)  # edgecolor of fliers
    
    return fig_boxPlt


def drawBoxPlotColumnWise(df, plt):
    # Boxplot for each attribute with target column  
    d1 = df.copy()
    axs = list()
    
    fig_bxpltclm, axs = plt.subplots(16, 5, figsize = (22,120))
    fig_bxpltclm.suptitle('Boxplot for each attribute with Target Column', fontsize=25, y=0.89)
    columns = list(d1.columns)
    k = 0
    for i in range(len(axs)):
        for j in range(len(axs[1])):
            if (k < len(columns)):
                sns.boxplot(x='Bankrupt?', y=columns[k], data=d1, ax=axs[i][j], palette={0:'#388E3C', 1:'#FFA000'})
                k = k + 1
                
    return fig_bxpltclm
    
def drawKDEplot(df, plt, cust_palette): 
    # Density Distribution of data in dataset with reduced columns  
    fig_kdePlt, axes = plt.subplots(nrows=13, ncols=6, figsize=(25,55))
    fig_kdePlt.suptitle('Density Distribution using KDE plot for all columns', fontsize=20, y=0.90)
    
    for i, ax in enumerate(axes.flatten()): 
        if(i<df.columns.size): 
            plt.subplot(13, 6, i+1)
            sns.kdeplot(df[df.columns[i]], fill = True, warn_singular=False, color='#FF6F00')
            
    return fig_kdePlt
    
def drawPairPlot(df, plt, clr_lst):
    # Input dataset is df_plot_20 - Pair Plot For Top 20 Columns     
    fig, axs = plt.subplots(figsize = (20, 20)) 
    
    pair_grid = sns.pairplot(data=df, diag_kind='kde', 
                     height=1.9, aspect=1.1, markers=['o', 'o'], 
                     hue='Bankrupt?', palette={0:clr_lst[0], 1:clr_lst[1]}, 
                     corner=True) 
    # pair_grid.fig.suptitle('Pair Plot For Top 20 Columns', fontsize=50)
    sns.move_legend(pair_grid, "upper right", bbox_to_anchor=(1, 1), title_fontsize=20, fontsize=20, facecolor='lightgrey') 
    
    return pair_grid
    
def drawHexBinPlot(df, plt):
    # Input dataset is df_plot_20 - Hexbin Plot For Top 20 Columns Before Scaling 
    
    indx=0
    for i in range(0, df.columns.size, 2): 
        if(i < df.columns.size-1):
            g = sns.jointplot(data=df, x=df.columns[i], y=df.columns[i+1], 
                              height=4, ratio=3, kind='hex', color="#FFA000",  
                              marginal_kws=dict(bins=20, color='#FFA000'),
                              joint_kws=dict(gridsize=15)
                             ) 
            g.savefig(os.getcwd() + '\\Images\\hex_plot_' + str(indx) + '.png', format='png', dpi=600)         
            plt.close(g.fig)
            indx += 1 
     
    fig_hxBin, axarr = plt.subplots(4, 3, sharex=False, sharey=False, figsize=(12, 12))
    fig_hxBin.suptitle('Hexbin Plot For Top 20 Columns', fontsize=15)
    index=0
    for p in range(4):
        for q in range(3):  
            if(index <= int(df.columns.size/2)-1): 
                axarr[p, q].imshow(mpimg.imread(os.getcwd() + '\\Images\\hex_plot_' + str(index) + '.png'), origin='upper')
                index += 1 
    
    # turn off x and y axis
    [ax.set_axis_off() for ax in axarr.ravel()]
    
    plt.tight_layout()
    #plt.show()
    return fig_hxBin

def dropDfColumns(df, col_type):
    
    if(col_type == 'Correlation'):
        # Correlated Columns list from master file
        cols_drop = [
            'ROA(B) before interest and depreciation after tax',
            'ROA(C) before interest and depreciation before interest',
            'Realized Sales Gross Margin', 'Gross Profit to Sales',
            'Continuous interest rate (after tax)', 'Pre-tax net Interest Rate',
            'After-tax net Interest Rate', 'Net Value Per Share (B)',
            'Net Value Per Share (C)', 'Per Share Net profit before tax (Yuan ¥)',
            'Persistent EPS in the Last Four Seasons',
            'After-tax Net Profit Growth Rate', 'Debt ratio %', 'Borrowing dependency',
            'Current Liabilities/Equity', 'Current Liability to Equity',
            'Working capitcal Turnover Rate', 'Liability-Assets Flag', 'Net Income Flag'
        ]
    elif(col_type == 'VIF'):
        # VIF Columns list from master file
        cols_drop = ["Debt ratio %", "Current Liabilities/Liability", "Current Liability to Assets", 
                            "Current Liabilities/Equity", "Operating Profit Rate", "Operating Gross Margin", 
                            "Net Value Per Share (A)", "After-tax net Interest Rate", "Gross Profit to Sales", 
                            "Operating profit/Paid-in capital", "Net Value Per Share (C)", "Liability to Equity", 
                            "Continuous interest rate (after tax)", "After-tax Net Profit Growth Rate", 
                            "ROA(B) before interest and depreciation after tax", "Current Liability to Equity", 
                            "Current Assets/Total Assets", "ROA(A) before interest and % after tax", "Borrowing dependency", 
                            "Cash Flow to Sales", "Net profit before tax/Paid-in capital",
                            "Persistent EPS in the Last Four Seasons", "Working Capital/Equity", "Net Income to Total Assets"] 

    # Dropping columns
    df_reduced_cols = df.drop(cols_drop, axis=1)
    return df_reduced_cols
 
#@st.cache_data
#@st.cache_resource
def load_css(file_name = "./Resources/style.css"):
    with open(file_name) as f:
        css = f'<style>{f.read()}</style>'
    return css

def main(): 
    
    # Loading Style Sheet 
    myCss = load_css()
    st.markdown(myCss, unsafe_allow_html=True) 

    # Creating Header
    main_bg=base64.b64encode(open('./Resources/BITS_Pilani_Logo.png', "rb").read()).decode()

    st.markdown("<div id='myHeader'><br>" + 
                 "<h2 style=text-align:center>BANKRUPTCY PREDICTION</h2>" + 
                 "<h6 style=text-align:center>PGP IN AI & ML - 2022 | Cohort 8 | Group 10 | Capstone Project - PCAMZC321</h6>" + 
                 "<br></div>", unsafe_allow_html=True) 
    st.markdown(
         f"""
         <style>
         #myHeader {{ background: url(data:image/png;base64,{main_bg}); 
         background-size: 100px;
         background-repeat: no-repeat;
         background-position: right center;
         margin-right: 200px;}}
         </style>
         """,
         unsafe_allow_html=True
     )
     
    #============================================================================
    
    # Tabbed Views
    tab_ovr, tab_pre, tab_viz, tab_prediction, tab_cmp = st.tabs(["OVERVIEW", "PREPROCESSING", "VISUALIZATIONS", "PREDICTIONS", "COMPARISION"])         
     
    # Sidebar - Header
    st.sidebar.header("Bankruptcy Prediction") 
    #======================= Declaring Required Variables ======================= 
    prePrcs_plots = ['Box Plot', 'Box Plot Column Wise', 'Hexbin Plot', 'KDE Plot', 'Pair Plot']
    
    avl_dfs = ['Original', 'Correlation Dropped', 'Mutual Information', 'VIF']
    avl_models = ['ADA Boost', 'Decision Tree', 'Random Forest', 'Voting Classifier', 'Stacking Classifier'] 

    metricsInfo = pd.DataFrame(columns=['Model', 'Approach', 'DS Type', 'Sampling', 'RecallOne', 'RecallZero', 
                                        'PrecisionOne', 'PrecisionZero', 'F1ScoreOne', 'F1ScoreZero']) 
    #============================================================================
    data = None
    avl_dataframes = None
    avl_buildModels = None
    
    cbx_demo_file = st.sidebar.checkbox('Use DEMO file')
    st.markdown(f'''
                    <style> div [data-testid="stCheckbox"] label span.st-dz {{ margin-top: 15px; }} </style>
                ''', unsafe_allow_html=True)
    # Use demo file  
    if cbx_demo_file: 
        data = pd.read_csv('./Resources/data_7030_predict.csv')
        # data.columns = data.columns.str.strip()
    else:         
        # Sidebar - Upload Data File
        uploaded_file = st.sidebar.file_uploader("Upload Your .csv File", type='csv')
        if ( uploaded_file is not None ):
            data = OnFileUpload(uploaded_file)
            # data.columns = data.columns.str.strip() 
        else: return   

    data.columns = data.columns.str.strip() 

    # Sidebar -  multi-select widgets 

    if st.sidebar.checkbox('All Datasets'): 
        avl_dataframes   = st.sidebar.multiselect('Select Input Dataset', avl_dfs, avl_dfs)
    else: 
        avl_dataframes   = st.sidebar.multiselect('Select Input Dataset', avl_dfs, avl_dfs[0])

    if st.sidebar.checkbox('All Models'): 
        avl_buildModels  = st.sidebar.multiselect('Select Model', avl_models, avl_models)
    else: 
        avl_buildModels  = st.sidebar.multiselect('Select Model', avl_models, avl_models[0])
    #======================================================================================
    
    # getBasicStatistics(data, tab_ovr) 
    col1, col2, col3, col4 = tab_ovr.columns([0.25, 0.25, 0.25, 0.25]) 

    con1 = col1.container()
    con1.markdown("<div class='myContainer'><div class='txth2'>SAMPLES<br>" + str(data.shape[0]) + "</div></div>", unsafe_allow_html=True)
    
    con2 = col2.container()
    con2.markdown("<div class='myContainer'><div class='txth2'>FEATURES<br>" + str(data.shape[1]) + "</div></div>", unsafe_allow_html=True)
    
    con3 = col3.container()
    con3.markdown("<div class='myContainer'><div class='txth2'>MISSING VALUES?<br>" + str(data.isnull().values.any()) +
                  "</div></div>", unsafe_allow_html=True)
    
    con4 = col4.container()
    con4.markdown("<div class='myContainer'><div class='txth2'>DUPLICATES?<br>" + str(data.duplicated().values.any()) +
                  "</div></div>", unsafe_allow_html=True) 

    #===========================================================================
    df_features = data.iloc[:, 1:] 
    dfScaler_data = MinMaxScaler(feature_range=(0, 1)).fit_transform(df_features)
    dfScaled_new = pd.DataFrame(dfScaler_data, columns=df_features.columns) 
    dfScaled_new = pd.DataFrame(data.iloc[:, 0]).join(dfScaled_new)
 
    tab_ovr.markdown("<br>", unsafe_allow_html=True) 
    
    x1 = dfScaled_new[dfScaled_new.columns[1]][dfScaled_new['Bankrupt?'] == 0]
    x2 = dfScaled_new[dfScaled_new.columns[1]][dfScaled_new['Bankrupt?'] == 1]
    
    # Group data together
    hist_data = [x1, x2]
    
    group_labels = ['Healthy', 'Bankrupt'] 
    # ==================================================================
    # ===================== COLOR PALETTE ==============================
    cust_palette = ['#D32F2F', '#FFA000', '#FFEB3B', '#388E3C', '#00796B', '#039BE5', '#1976D2', '#303F9F', '#7B1FA2']
    # ===================== DARK COLOR SCHEME ==========================
    clr_primary = '#FFA500'
    clr_bkgrnd  = '#001E3C'
    clr_sBkGrnd = '#152D4D'
    clr_text    = '#E0E0E0'

    clr_elemnt1 = '#165C90'
    clr_elemnt2 = '#33496F'
    clr_elemnt3 = '#1DE9B6'
    clr_elemnt4 = '#FFFFFF'

    clr_elemnt5 = '#8BADC8'
    clr_elemnt6 = '#6D99BA'
    clr_elemnt7 = '#5085AC'
    clr_elemnt8 = '#33709E'
    clr_elemnt9 = '#165C90'

    clr_plt_bar = [clr_elemnt5, clr_elemnt6, clr_elemnt7, clr_elemnt8, clr_elemnt9]
    # ==================================================================
    plt.rcParams['figure.facecolor']= clr_sBkGrnd
    plt.rcParams['figure.edgecolor']= clr_sBkGrnd
    
    plt.rcParams['axes.facecolor']= clr_sBkGrnd
    plt.rcParams['axes.edgecolor']= clr_text
    plt.rcParams['axes.labelcolor']= clr_text
    plt.rcParams['axes.spines.left']= True
    plt.rcParams['axes.spines.bottom']= True
    plt.rcParams['axes.spines.top']= False
    plt.rcParams['axes.spines.right']= False
    plt.rcParams['axes.grid']=False
    
    plt.rcParams['patch.edgecolor']= clr_sBkGrnd
    plt.rcParams['patch.force_edgecolor']=True
    
    plt.rcParams['xtick.color']= clr_text
    plt.rcParams['xtick.labelcolor']= clr_text
    plt.rcParams['ytick.color']= clr_text
    plt.rcParams['ytick.labelcolor']= clr_text

    # ================== Plot with Class Distribution ==================
    df_zList = pd.DataFrame(columns=['Idx', 'Bankrupt', 'Healthy'])

    d_b = (dfScaled_new['Bankrupt?'] == 1).sum(axis=0)
    d_h = (dfScaled_new['Bankrupt?'] == 0).sum(axis=0)
    df_zList = df_zList.append({'Idx': 'Class Labels','Bankrupt': d_b, 'Healthy': d_h}, ignore_index=True)     
    
    f_classDist, ax_barPlt = plt.subplots(figsize=(18, 0.7))  
    ax_barPlt.set_yticks([]) 
    plt.title('Target Class Distribution',fontsize=15, color= clr_text) 
    
    plt1 = sns.barplot(y = 'Idx', x = 'Healthy', data = df_zList, label = 'Healthy', color = clr_elemnt1)
    plt2 = sns.barplot(y = 'Idx', x ='Bankrupt', data = df_zList, label = 'Bankrupt', color = clr_primary)    
    
    plt.legend (ncol = 1, loc = "center right", frameon = True, bbox_to_anchor=(0, 0, 1.09, 1.09), 
                facecolor=clr_sBkGrnd, edgecolor=clr_sBkGrnd, labelcolor=clr_elemnt4, fontsize='medium') 
    plt.xlabel('Input Records/Samples')
    plt.ylabel('') 
    
    # =================================================================
    #  DistPlot
    f_distPlot = figFctry.create_distplot(hist_data, group_labels, bin_size=[.1, .25], colors=[clr_elemnt1, clr_primary]) 
    f_distPlot.update_layout(paper_bgcolor=clr_sBkGrnd, plot_bgcolor=clr_sBkGrnd, width=600, height=330)
    f_distPlot.update_xaxes(linecolor=clr_text, gridcolor=clr_elemnt2)
    f_distPlot.update_yaxes(linecolor=clr_text, gridcolor=clr_elemnt2)
    
    # =================================================================
    
    f_mnSdMd, ax = plt.subplots(figsize=(18, 10)) 

    df_statsDescribe = pd.DataFrame(dfScaled_new.describe()) 
    
    sns.lineplot(x=range(0, 96), y=df_statsDescribe.iloc[1, :], color=clr_elemnt3, linewidth = 5) # Mean
    sns.lineplot(x=range(0, 96), y=df_statsDescribe.iloc[2, :], color=clr_primary, linewidth = 2) # SD
    sns.lineplot(x=range(0, 96), y=df_statsDescribe.iloc[5, :], color=clr_elemnt1, linewidth = 4) # Median
    ax.legend(['Mean', 'Std Dev', 'Median'], facecolor=clr_sBkGrnd, edgecolor=clr_sBkGrnd, labelcolor=clr_elemnt4, fontsize='x-large') 
    
    # ====================== PLOTS ====================================
    # Target Class Distribution - Horizontal Bar
    fig_con1 = tab_ovr.container() 
    fig_con1.pyplot(f_classDist, use_container_width=True) 
    
    col_fig1, col_fig2 = tab_ovr.columns([0.50, 0.50]) 
    fig_con2 = col_fig1.container()
    fig_con3 = col_fig2.container()
    
    # Plotly - DistPlot 
    fig_con2.plotly_chart(f_distPlot, use_container_width=True, facecolor=clr_sBkGrnd)   

    # Mean | SD | Median chart 
    fig_con3.pyplot(f_mnSdMd, use_container_width=True)  

    #==========================================================
    col_sel  = 'Equity to Liability'
    col_sel2 = 'Net worth/Assets'
    
    # Visualizations for all cols
    mySelections = tab_ovr.container() 
    vizOptn = mySelections.selectbox('Select A Feature To Visualize...', dfScaled_new.columns[1:])
    
    col_sel = vizOptn
    
    fc4_mCont = tab_ovr.container()
    
    fc4_col1, fc4_col2 = fc4_mCont.columns([0.50, 0.50])
    fc4_col3, fc4_col4 = fc4_col2.columns([0.60, 0.40])
    
    x_stat = ['25%', '50%', '75%', 'Mean', 'Std Dev']
    y_stat = [df_statsDescribe[col_sel]['25%'], df_statsDescribe[col_sel]['50%'], df_statsDescribe[col_sel]['75%'],
              df_statsDescribe[col_sel]['mean'], df_statsDescribe[col_sel]['std']] 
    
    # ========================= HexBin =========================
    fc4_c1_cont = fc4_col1.container() 
    
    fig_fc4_c1_cont, ax_fc4_c1_cont = plt.subplots(figsize=(15, 15))

    norm = matplotlib.colors.Normalize(-1,1) 

    myCmapColors = [[norm(-1.0), clr_sBkGrnd],
                      [norm(-0.7), clr_elemnt9],
                      [norm(-0.3), clr_elemnt8],
                      [norm( 0.1), clr_elemnt7],
                      [norm( 0.3), clr_elemnt6],
                      [norm( 0.7), clr_elemnt5],
                      [norm( 1.0), clr_elemnt9]] 

    myCmap = matplotlib.colors.LinearSegmentedColormap.from_list("", myCmapColors)
    

    fig_fc4_c1_cont = sns.jointplot(kind='hex', data=dfScaled_new, x=col_sel, y=col_sel2,
                                        height=12, ratio=5, cmap=myCmap, joint_kws=dict(gridsize=40),
                                        marginal_kws=dict(bins=40, color=clr_elemnt3), extent=[0, 1.2, 0, 1.2]) 
    
    fc4_c1_cont.pyplot(fig_fc4_c1_cont, use_container_width=True)
 
    # ========================= Scatter =========================
    fc4_c2_rw1_cont = fc4_col3.container()
    fig_fc4_c2_rw1_cont, ax_fc4_c2_rw1_cont = plt.subplots(figsize=(10, 5))

    # plt.scatter(x=dfScaled_new[col_sel], y=dfScaled_new['Bankrupt?'], color=clr_primary)

    alphas = [0.9, 0.9]    # [0.8, 0.4] 
    colors = [clr_elemnt3, clr_primary]
    markers= ['o', 'X']
    
    i_h = dfScaled_new['Bankrupt?'] == 0
    i_b = dfScaled_new['Bankrupt?'] == 1 
    
    plt.scatter(x=dfScaled_new.loc[i_h, col_sel], y=dfScaled_new.loc[i_h, 'Bankrupt?'], s=50,
                    alpha=alphas[0], color=colors[0], marker=markers[0])

    plt.scatter(x=dfScaled_new.loc[i_b, col_sel], y=dfScaled_new.loc[i_b, 'Bankrupt?'], s=50,
                    alpha=alphas[1], color=colors[1], marker=markers[1])
    
    fc4_c2_rw1_cont.pyplot(fig_fc4_c2_rw1_cont, use_container_width=True)
     
    # ========================= KDE plot =========================
    fc4_c2_rw2_cont = fc4_col3.container()
    fig_fc4_c2_rw2_cont, ax_fc4_c2_rw2_cont = plt.subplots(figsize=(10, 5))
    
    sns.kdeplot(dfScaled_new[col_sel], fill = True, warn_singular=False, color=clr_primary, ax=ax_fc4_c2_rw2_cont)

    fc4_c2_rw2_cont.pyplot(fig_fc4_c2_rw2_cont, use_container_width=True)
     
    # ========================= Grouped Bar Plot ========================= 
    fc4_c2_rw3_cont = fc4_col2.container() 
    fig_fc4_c2_rw3_cont, ax_fc4_c2_rw3_cont = plt.subplots(figsize=(15, 4.5)) 

    sns.barplot(x = x_stat, y = y_stat, ax=ax_fc4_c2_rw3_cont, palette=clr_plt_bar)  
    plt.axhline(y=df_statsDescribe[col_sel]['max'], color=clr_primary, linestyle='-.', linewidth=4)
    
    fc4_c2_rw3_cont.pyplot(fig_fc4_c2_rw3_cont, use_container_width=True)
    
    # ========================= Box Plot =========================
    fc4_c4_rw1_cont = fc4_col4.container()
    fig_fc4_c4_rw1_cont, ax_fc4_c4_rw1_cont = plt.subplots(figsize=(4, 7.6))

    sns.boxplot(x='Bankrupt?', y=col_sel, data=dfScaled_new, ax=ax_fc4_c4_rw1_cont, palette={0:clr_elemnt1, 1:clr_primary})
    
    box_patches = [patch for patch in ax_fc4_c4_rw1_cont.patches if type(patch) == matplotlib.patches.PathPatch]
    if len(box_patches) == 0:
        # in matplotlib older than 3.5, the boxes are stored in ax2.artists
        box_patches = ax_fc4_c4_rw1_cont.artists
        
    num_patches = len(box_patches)
    lines_per_boxplot = len(ax_fc4_c4_rw1_cont.lines) // num_patches
    
    for i, patch in enumerate(box_patches): 
        patch.set_edgecolor(clr_text)  
        
        for line in ax_fc4_c4_rw1_cont.lines[i * lines_per_boxplot: (i + 1) * lines_per_boxplot]: 
            line.set_color(clr_text)
            line.set_mfc(clr_text)  # facecolor of fliers
            line.set_mec(clr_text)  # edgecolor of fliers
    
    fc4_c4_rw1_cont.pyplot(fig_fc4_c4_rw1_cont, use_container_width=True) 

    # To read Feature_Definitions.xlsx and display Short_Description for selected feature
    df_featureDesc = pd.read_excel('./Resources/Feature_Definitions.xlsx')
    
    # tab_ovr.info(df_featureDesc[df_featureDesc['Feature'] == 'Operating Gross Margin'].iloc[0, 2]) 
    tab_ovr.info(df_featureDesc[df_featureDesc['Feature'] == vizOptn].iloc[0, 2]) 
    #============================================================================
    df_original = data.copy()   
    
    if (len(avl_dataframes) and len(avl_buildModels)):
        if st.sidebar.button('Predict', type="secondary"):
            
            df_original = data.copy()     
            metricsInfo = pd.DataFrame(columns=['Model', 'Approach', 'DS Type', 'Sampling',
                                                'RecallOne', 'RecallZero',
                                                'PrecisionOne', 'PrecisionZero',
                                                'F1ScoreOne', 'F1ScoreZero', 'Accuracy'])
            
            # CORRELATION  
            df_reduced_cols = dropDfColumns(df_original, 'Correlation') 
            expander_dataCorrelation = tab_pre.expander('Dataset Obtained After Dropping Correlated Columns', True) 
            
            corrCol1, corrCol2, corrCol3 = expander_dataCorrelation.columns([0.40, 0.20, 0.40], gap="small")
            corrCol1.markdown('<h3 style=text-align:center>TOP ROWS FROM DATASET</center>', unsafe_allow_html=True) 
            corrCol1.dataframe(df_reduced_cols.head(8).style.background_gradient(cmap = "YlGnBu")) 
            
            corrCol2.markdown('<br><br><br><h2 style=text-align:center>Samples<br>' + str(df_reduced_cols.shape[0]) + '</center>', unsafe_allow_html=True) 
            corrCol2.markdown('<h2 style=text-align:center>Features<br>' + str(df_reduced_cols.shape[1]) + '</center>', unsafe_allow_html=True)

            corrCol3.markdown('<h4 style=text-align:center>BASIC STATISTICS</center>', unsafe_allow_html=True)
            corrCol3.dataframe(df_reduced_cols.describe().style.background_gradient(cmap = "YlGnBu"))
            
            # Mutual Info Classifier -> List of Top 20 columns to RETAIN, fetched from master code v0.9
            mi_cols_needed = ["Net Income to Stockholder's Equity", "Interest Expense Ratio", "Net worth/Assets",
                        "Interest Coverage Ratio (Interest expense to EBIT)", "Equity to Liability",
                        "Liability to Equity", "Degree of Financial Leverage (DFL)", 
                        "Net profit before tax/Paid-in capital", "ROA(A) before interest and % after tax", 
                        "Net Income to Total Assets", "Quick Ratio", "Retained Earnings to Total Assets", 
                        "Total income/Total expense", "Net Value Per Share (A)", "Net Value Growth Rate", 
                        "Current Ratio", "Non-industry income and expenditure/revenue", 
                        "Current Liability to Current Assets", "Working Capital to Total Assets", 
                        "Cash flow rate", "Bankrupt?"]
                        
            df_mutual_info_classif = df_original[mi_cols_needed]
            expander_mutualInfo = tab_pre.expander('Top 20 Columns - Feature Extracted using Mutual Info Classifier', False)              
            
            miCol1, miCol2, miCol3 = expander_mutualInfo.columns([0.40, 0.20, 0.40], gap="small")
            miCol1.markdown('<h3 style=text-align:center>TOP ROWS FROM DATASET</center>', unsafe_allow_html=True) 
            miCol1.dataframe(df_mutual_info_classif.head(8).style.background_gradient(cmap = "YlGnBu")) 
            
            miCol2.markdown('<br><br><br><h2 style=text-align:center>Samples<br>' + str(df_mutual_info_classif.shape[0]) + 
                            '</center>', unsafe_allow_html=True) 
            miCol2.markdown('<h2 style=text-align:center>Features<br>' + str(df_mutual_info_classif.shape[1]) + '</center>', unsafe_allow_html=True)

            miCol3.markdown('<h4 style=text-align:center>BASIC STATISTICS</center>', unsafe_allow_html=True)
            miCol3.dataframe(df_mutual_info_classif.describe().style.background_gradient(cmap = "YlGnBu"))

            df_vif_cols = dropDfColumns(df_original, 'VIF')
            expander_vif = tab_pre.expander('VIF Dataset obtained after dropping columns using Variance Inflation Factor', False) 

            vifCol1, vifCol2, vifCol3 = expander_vif.columns([0.40, 0.20, 0.40], gap="small")
            vifCol1.markdown('<h3 style=text-align:center>TOP ROWS FROM DATASET</center>', unsafe_allow_html=True) 
            vifCol1.dataframe(df_vif_cols.head(8).style.background_gradient(cmap = "YlGnBu")) 
            
            vifCol2.markdown('<br><br><br><h2 style=text-align:center>Samples<br>' + str(df_vif_cols.shape[0]) + '</center>', unsafe_allow_html=True) 
            vifCol2.markdown('<h2 style=text-align:center>Features<br>' + str(df_vif_cols.shape[1]) + '</center>', unsafe_allow_html=True)

            vifCol3.markdown('<h4 style=text-align:center>BASIC STATISTICS</center>', unsafe_allow_html=True)
            vifCol3.dataframe(df_vif_cols.describe().style.background_gradient(cmap = "YlGnBu")) 

            exp_viz1 = tab_viz.expander('Box Plot', expanded=False)
            with st.spinner('Loading Pair Plot...Please Wait...'):
                exp_viz1.pyplot(drawBoxPlot(df_reduced_cols, plt, cust_palette, clr_text))
 
            exp_viz2 = tab_viz.expander('Pair Plot', expanded=True)
            with st.spinner('Loading Pair Plot...Please Wait...'):
                exp_viz2.pyplot(drawPairPlot(df_mutual_info_classif, plt, [clr_elemnt3, clr_primary]))
                                            
            #====== Preprocessing Completed ====== 
            for mdlSelected in avl_buildModels: 
                for dstSelected in avl_dataframes: 
                    # Get Dataset as per user selection 
                    if(dstSelected == 'Original'):
                        # Applying Standard Scaling on X_features
                        df_selected = df_original.copy()
                        X_features  = df_selected.iloc[:, 1:]
                        y_target    = df_selected.iloc[:, 0]
                        X_features  = scaleData(X_features, 'Standard')
                        df_selected = X_features.join(y_target) 
                        
                    elif(dstSelected == 'Correlation Dropped'):
                        # Applying Standard Scaling on X_features
                        df_selected = df_reduced_cols.copy() 
                        X_features  = df_selected.iloc[:, 1:]
                        y_target    = df_selected.iloc[:, 0]
                        X_features  = scaleData(X_features, 'Standard')
                        df_selected = X_features.join(y_target) 
                        
                    elif(dstSelected == 'Mutual Information'):
                        # Applying Standard Scaling on X_features
                        df_selected = df_mutual_info_classif.copy()
                        X_features  = df_selected.iloc[:, :-1]
                        y_target    = df_selected.iloc[:, -1]
                        X_features  = scaleData(X_features, 'Standard')
                        df_selected = X_features.join(y_target) 
                        
                    elif(dstSelected == 'VIF'):
                        # Applying Standard Scaling on X_features
                        df_selected = df_vif_cols.copy() 
                        X_features  = df_selected.iloc[:, 1:]
                        y_target    = df_selected.iloc[:, 0]
                        X_features  = scaleData(X_features, 'Standard')
                        df_selected = X_features.join(y_target) 
                        
                    else:
                        df_selected = None 
                        st.write(dstSelected + ' not from given list')   
            
                    # Model Prediction 
                    if(mdlSelected == 'Voting Classifier' or mdlSelected == 'Stacking Classifier'):
                        mdlFileName = mdlSelected.replace(' ', '_') + '_Ensemble_' + dstSelected.replace(' ', '_') + '.sav'
                    else:
                        mdlFileName = mdlSelected.replace(' ', '_') + '_Hyper_Parameter_Tuned_' + dstSelected.replace(' ', '_') + '.sav'
                        
                    y_my_pred, acrcy_scr, df_clsf_rep, f_clsf_rep = predictMyModel(mdlFileName, df_selected, plt)
                    keyMetrics = {'Model': mdlSelected, 'Approach': 'HP', 'DS Type': dstSelected, 'Sampling': 'None', 
                                      'RecallZero': df_clsf_rep.iloc[0, 1], 'RecallOne': df_clsf_rep.iloc[1, 1],
                                      'PrecisionZero': df_clsf_rep.iloc[0, 0], 'PrecisionOne': df_clsf_rep.iloc[1, 0],
                                      'F1ScoreZero': df_clsf_rep.iloc[0, 2], 'F1ScoreOne': df_clsf_rep.iloc[1, 2],
                                      'Accuracy': acrcy_scr
                    }
                    
                    metricsInfo = metricsInfo.append(keyMetrics, ignore_index=True) 
                    # tab_cmp
                    #==========================================================
                    exp_preprcs = tab_prediction.expander(mdlSelected + ' | ' + dstSelected + ' Dataset | Accuracy : ' + str(acrcy_scr), expanded=False)
                    
                    subCol0, subCol1, subCol2, subCol3 = exp_preprcs.columns([0.10, 0.30, 0.15, 0.45], gap="small")

                    container0 = subCol0.container()
                    container0.markdown("<br><br>", unsafe_allow_html=True) 
                    
                    container1 = subCol1.container()
                    container1.pyplot(f_clsf_rep, use_container_width=True) 
                    
                    container2 = subCol2.container()
                    container2.markdown("<br><br>", unsafe_allow_html=True) 
                
                    container3 = subCol3.container()
                    container3.markdown("<br><br>", unsafe_allow_html=True) 
                    container3.dataframe(df_clsf_rep) 

                    # Predictions in a dataframe
                    df_inp = pd.DataFrame(df_selected.iloc[:, -1], columns=['Bankrupt?']).join(pd.DataFrame(df_selected.iloc[:, :-1]))
                    df_predictions = pd.DataFrame(y_my_pred, columns=['Prediction']).join(df_inp) 
                    exp_preprcs.markdown("<br>", unsafe_allow_html=True) 
                    exp_preprcs.info('Correct Predictions')
                    exp_preprcs.dataframe(df_predictions[df_predictions['Prediction'] == df_predictions['Bankrupt?']])
                    exp_preprcs.markdown("<br>", unsafe_allow_html=True) 
                    exp_preprcs.info('In-correct Predictions')
                    exp_preprcs.dataframe(df_predictions[df_predictions['Prediction'] != df_predictions['Bankrupt?']])
                    #==========================================================
 
            # Write consolidated results to COMPARISION Tab after all combinations are executed
            metricsInfo = metricsInfo.sort_values(by = ['RecallOne', 'RecallZero'], ascending = [False, False])
            tab_cmp.write(metricsInfo) 
            
if __name__ == '__main__':
    main()

# end of file