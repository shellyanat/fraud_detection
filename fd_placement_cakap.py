import streamlit as st
# For performing any kind of Mathematical Operations
import numpy as np
# For dealing with DataFrames
import pandas as pd

# For Data Visualization
import matplotlib.pyplot as plt

# For Data Visualization
import seaborn as sns

# For Download Result
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb

import plotly.graph_objects as go

from PIL import Image
import plotly.express as px

#page setting

#page setting
st.set_page_config(page_title='Teacher Fraud Detection CAKAP',layout="wide")

st.markdown('''
<style>
/*center metric label*/
[data-testid="stMetricLabel"] > div:nth-child(1) {
    justify-content: center;
}

/*center metric value*/
[data-testid="stMetricValue"] > div:nth-child(1) {
    justify-content: center;
}
</style>
''', unsafe_allow_html=True)

st.markdown("""
<style>
div[data-testid="metric-container"] {
   background-color: rgba(28, 131, 225, 0.1);
   border: 1px solid rgba(28, 131, 225, 0.1);
   padding: 2% 2% 2% 8%;
   border-radius: 5px;
   color: rgb(30, 103, 119);
   overflow-wrap: break-word;
}

/* breakline for metric text         */
div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div {
   overflow-wrap: break-word;
   white-space: break-spaces;
   color: blue;
}
</style>
"""
, unsafe_allow_html=True)


image = Image.open('Cakap_Logo.png')

st.image(image, output_format='PNG', width=480)
'''
## Fraud Detection by Data Analyst Team
'''
st.markdown("---")
a1,a2,a3 = st.columns([2,4,2])
with a2:
    st.write("Business Process Flow")
    image = Image.open('flow.drawio.png')

    st.image(image, output_format='PNG', width=480)

'''
### Fraud Detection

What Is Fraud Detection?

Fraud detection is a collection of processes and techniques designed to identify, monitor, and prevent fraud. In the online business world, fraud, scams, and bad agents are damaging in a number of ways.
Companies have to put steps in place to ensure that fraud is detected and stopped before it affects business.

This dataset is a data placement test where the teacher has the possibility to commit fraud together with students. 
as mentioned previously, so we collect data providing several metrics that could possibly indicate fraud
'''



st.markdown("---")
# techinal instruction
'''
### Instructions
'''

with st.expander("Instructions To Use This Streamlit"):
    st.markdown('<div style="text-align: justify;">1. Import The Placement Dataset (xlsx file) </div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: justify;">2. Import The Activity Dataset (xlsx file) </div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: justify;">3. If there are any error after import the data, please pay attention to the columns. It is important for the dataset to have the same columns as the example dataset.</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: justify;">4. If you want to download the clean dataset, click on download dataset button </div>', unsafe_allow_html=True)

# upload file
st.markdown("---")
'''
### Import Data

##### Upload Placement Dataset Here (XLSX)
'''
'''
Placement Dataset, it consists of 13 row and 37.454 columns of each Placement Test session record, including: student_id, agent_id, teacher_id, held_at, buy_premium_after_pt, student_presence.
'''
uploaded_file = st.file_uploader("Upload Placement File (XLSX)")
if uploaded_file is not None:
    dataframe1 = pd.read_excel(uploaded_file)
    #st.write(dataframe)

df_p1 = dataframe1.copy()

'''
##### Upload Activity Dataset Here (XLSX)
'''
'''
Activity Dataset, it consists of 12 row and 37.452 columns of various activity products from each recorded student, including: user_id, attend_private, entry_club, feed_like, etc.
'''
uploaded_file = st.file_uploader("Upload Activity File (XLSX)")
if uploaded_file is not None:
    dataframe2 = pd.read_excel(uploaded_file)
    #st.write(dataframe)

df_a1 = dataframe2.copy()

st.markdown("---")
##### ---- P R O C E S S I N G ----

## ----- PLACEMENT ------

#### M I S S I N G  V A L U E S
# drop missing values on agent_id column
# the reasons is later we will define the fraud based on the phone call from agent was held or not
# so if there's no agent_id, it will be hard to indicate the students as a fraud 
df_p1 = df_p1.drop(df_p1[(df_p1['agent_id'].isna() == True) & (df_p1['call_before_pt'].isna() == True) & (df_p1['call_after_pt'].isna() == True)].index)


#### D A T A  T Y P E

#change column with dtype float to int
df_p1['student_id'] = df_p1['student_id'].astype(int)
df_p1['total_call_before_pt'] = df_p1['total_call_before_pt'].astype(int)
df_p1['total_call_after_pt'] = df_p1['total_call_after_pt'].astype(int)
df_p1['teacher_id'] = df_p1['teacher_id'].astype(int)
df_p1['agent_id'] = df_p1['agent_id'].astype(int)

#### M A P P I N G -- new columns for help

# call history
#if the agent call the student it will return as True Value
#if the agent didn't call the student (NaN) it will return as False
df_p1['call_before_pt_yes'] = np.where(df_p1['call_before_pt'].isna() == False, True, False)
df_p1['call_after_pt_yes'] = np.where(df_p1['call_after_pt'].isna() == False, True, False)


#mapping held_at
df_p1['not_held'] = df_p1['held_at'].isna()
df_p1['held_at_1'] = df_p1['held_at'].str.contains(r'1',na=False)
df_p1['held_at_2'] = df_p1['held_at'].str.contains(r'2',na=False)
df_p1['held_at_3'] = df_p1['held_at'].str.contains(r'3',na=False)
df_p1['held_at_4'] = df_p1['held_at'].str.contains(r'4',na=False)
#total student held the agent's phone call
df_p1['total_held'] = df_p1[['held_at_1', 'held_at_2','held_at_3','held_at_4']].sum(axis=1)


#mapping buy premium after pt (boolean) and student_presence
df_p1['buy_premium_y'] =  df_p1['buy_premium_after_pt'].str.contains(r'y',na=False)
df_p1['buy_premium_n'] =  df_p1['buy_premium_after_pt'].str.contains(r'n',na=False)
df_p1['student_presence_y'] = df_p1['student_presence'].str.contains(r'Y',na=False)
df_p1['student_presence_n'] = df_p1['student_presence'].str.contains(r'N',na=False)

#change astype int (0 - 1)
df_p1['call_before_pt_yes'] = df_p1['call_before_pt_yes'].astype(int)
df_p1['call_after_pt_yes'] = df_p1['call_after_pt_yes'].astype(int)
df_p1['not_held'] = df_p1['not_held'].astype(int)
df_p1['held_at_1'] = df_p1['held_at_1'].astype(int)
df_p1['held_at_2'] = df_p1['held_at_2'].astype(int)
df_p1['held_at_3'] = df_p1['held_at_3'].astype(int)
df_p1['held_at_4'] = df_p1['held_at_4'].astype(int)
df_p1['buy_premium_y'] = df_p1['buy_premium_y'].astype(int)
df_p1['buy_premium_n'] = df_p1['buy_premium_n'].astype(int)
df_p1['student_presence_y'] = df_p1['student_presence_y'].astype(int)
df_p1['student_presence_n'] = df_p1['student_presence_n'].astype(int)

#### D R O P  U N I M P O R T A N T  C O L U M N S
df_p2 = df_p1.copy()
df_p2 = df_p2.drop(['call_before_pt','call_after_pt','held_at','student_presence'], axis = 1)


## ---- ACTIVITY ------

### D R O P  U N I M P O R T A N T  C O L U M N S

# the upskill_playback columns filled with zero
df_a1 = df_a1.drop(['upskill_playback'], axis=1)

#### D A T A  T Y P E

#we want to see students with no activity
#it's eaasier if we differentiate the value as 0 (no activity) and 1 (there's activity)
df_a1['attend_private']       = df_a1['attend_private'].astype(bool).astype(int)
df_a1['attend_semi_private']  = df_a1['attend_semi_private'].astype(bool).astype(int)
df_a1['attend_corpo_group']   = df_a1['attend_corpo_group'].astype(bool).astype(int)
df_a1['attend_corpo_priv']    = df_a1['attend_corpo_priv'].astype(bool).astype(int)
df_a1['finish_chat']          = df_a1['finish_chat'].astype(bool).astype(int)
df_a1['entry_club']           = df_a1['entry_club'].astype(bool).astype(int)
df_a1['entry_upskill']        = df_a1['entry_upskill'].astype(bool).astype(int)
df_a1['selfpaced_join']       = df_a1['selfpaced_join'].astype(bool).astype(int)
df_a1['video_progress']       = df_a1['video_progress'].astype(bool).astype(int)
df_a1['feed_like']            = df_a1['feed_like'].astype(bool).astype(int)

#### T O T A L  A C T I V I T Y
#see the total activity of each student
#without the attend_private column
df_a2 = df_a1.copy()
df_a2['other_act'] = df_a2.iloc[: ,2:].sum(axis = 1)

#### S I M P I F L Y  T H E  D A T A F R A M E

df_a3 = df_a2.copy()
df_a3['other_act'] = df_a3['other_act'].astype(bool).astype(int)
df_a3['total_act'] = df_a3.loc[:,['attend_private','other_act']].sum(axis=1)
df_a3['act_is_zero'] = np.where(df_a3['total_act'] == 0, True, False)
df_a3['act_is_zero'] = df_a3['act_is_zero'].astype(int)
df_a3 = df_a3.drop(['attend_semi_private',
       'attend_corpo_group', 'attend_corpo_priv', 'finish_chat', 'entry_club',
       'entry_upskill', 'selfpaced_join', 'video_progress', 'feed_like'],axis=1)

## ------ JOIN DATAFRAME ------
#### J O I N

# the dataframe: placement and activity
df_p_a = pd.merge(df_p2, df_a3, how = 'outer', left_on = 'student_id', right_on = 'id_user')

#### D R O P

#missing values
df_p_a = df_p_a.dropna()  

#unimportant columns
df_p_a = df_p_a.drop(['id_user'], axis=1)

#### S I M P I F L Y  D A T A F R A M E
df_p_a2 = df_p_a.copy()
df_p_a2 = df_p_a2.drop(['held_at_1', 'held_at_2', 'held_at_3','held_at_4'],axis=1)


## ---- GROUP BY TEACHER ------
#### D R O P
#unimportant columns
df_teacher = df_p_a2.copy()
df_teacher = df_teacher.drop(['student_id','agent_id','total_call_before_pt', 'total_call_after_pt','call_before_pt_yes', 'call_after_pt_yes','class_date','submit_report_date'], axis=1)

#### G R O U P I N G
#all columns by teacher
df_teacher = df_teacher.groupby(df_teacher['teacher_id']).sum().reset_index()

if 'study_name' in df_p_a2.columns:
    df_sn = df_p_a2.drop(df_p_a2.columns.difference(['teacher_id','study_name']), axis=1)
    df_sn = df_sn.drop_duplicates()
    df_teacher = pd.merge(df_teacher, df_sn, how = 'left', on = ['teacher_id'])
else:
    df_teacher = df_teacher
df_teacher.head()

#### N E W  C O L U M N S
#to help

#total student
df_teacher['total_student'] = (df_teacher['buy_premium_y']+df_teacher['buy_premium_n'])

#student with zero activity percentage (per teacher)
df_teacher['act_zero_percentage'] = df_teacher['act_is_zero']/df_teacher['total_student']

#student that doesn't held agent call percentage (per teacher)
df_teacher['not_held_percentage'] = df_teacher['not_held']/df_teacher['total_student']

#student doesn't presence (per teacher)
df_teacher['student_presence_n_percentage'] = df_teacher['student_presence_n']/df_teacher['total_student']

#student that doesn't buy premium package percentage (per teacher)
df_teacher['bp_n_percentage'] = df_teacher['buy_premium_n']/df_teacher['total_student']

#### D A T A  T Y P E
#change column with dtype float to int
df_teacher['teacher_id'] = df_teacher['teacher_id'].astype(int)
df_teacher['total_student'] = df_teacher['total_student'].astype(int)
df_teacher['buy_premium_y'] = df_teacher['buy_premium_y'].astype(int)
df_teacher['buy_premium_n'] = df_teacher['buy_premium_n'].astype(int)


#### F R A U D  P O S S I B I L I T Y

#we trying to create the fraud value based on indication condition
def f(row):
    #var1 = 0 #the fraud value of total student
    if row['total_student'] >= df_teacher.total_student.quantile(0.75):
      var1 = 1
    else:
      var1 = 0

    #var2 = 0 #the fraud value of not buy premium percentage
    if row['bp_n_percentage'] >= df_teacher.bp_n_percentage.quantile(0.5):
      var2 = 1
    elif row['bp_n_percentage'] >= df_teacher.bp_n_percentage.quantile(0.25) and (row['bp_n_percentage'] < df_teacher.bp_n_percentage.quantile(0.5)):
      var2 = 0.5
    else:
      var2 = 0

    #var3 = 0 #the fraud value of not held percentage
    if row['not_held_percentage'] >= df_teacher.not_held_percentage.quantile(0.75):
      var3 = 1
    #elif row['not_held_percentage'] >= df_teacher.not_held_percentage.quantile(0.25) and (row['not_held_percentage'] < df_teacher.not_held_percentage.quantile(0.5)):
      #var2 = 0.5
    else:
      var3 = 0

    #var4 = 0 #the fraud value of activity zero percentage
    if row['act_zero_percentage'] >= df_teacher.act_zero_percentage.quantile(0.9):
      var4 = 0.5
    else:
      var4 = 0

    total = var1 + var2 + var3 + var4
    if total >= 3:
      val = 'High'
    elif total >= 2:
      val = 'Medium'
    else:
      val = 'Low'
    return val

df_teacher['fraud_possibility'] = df_teacher.apply(f, axis=1)



## Pie chart
df_fp_count  = df_teacher['fraud_possibility'].value_counts().rename_axis('fraud_possibility').reset_index(name='counts')
labels = df_fp_count['fraud_possibility']
sizes = df_fp_count['counts']
explode = (0, 0, 0.1)  # only "explode" the 3rd slice




## show teacher with high fraud possibility
df_fp = df_teacher.copy()
df_fp = df_fp.drop(df_fp.columns.difference(['teacher_id','study_name','total_student','bp_n_percentage','fraud_possibility']), axis=1)


High = df_fp[df_fp['fraud_possibility']== 'High'].reset_index(drop=True)
Medium = df_fp[df_fp['fraud_possibility']== 'Medium'].reset_index(drop=True)
Low = df_fp[df_fp['fraud_possibility']== 'Low'].reset_index(drop=True)

High.sort_values('total_student', axis = 0, ascending = False,
                 inplace = True)
High = High.reset_index(drop=True)

Medium.sort_values('total_student', axis = 0, ascending = False,
                 inplace = True)
Medium = Medium.reset_index(drop=True)

Low.sort_values('total_student', axis = 0, ascending = False,
                 inplace = True)
Low = Low.reset_index(drop=True)


'''
### Data Profile
'''
col_1, col_2 = st.columns(2)
with col_1:
     st.metric('Total Teacher', df_teacher['teacher_id'].nunique())
with col_2:
     st.metric('Total Student', df_teacher['total_student'].sum())                   

col_5, col_6 = st.columns(2)
with col_5:
    st.metric('Total Student Buy Premium', df_teacher['buy_premium_y'].sum())
with col_6:
    st.metric('Total Student Not Buy Premium', df_teacher['buy_premium_n'].sum())

col7, col8 = st.columns(2)
with col7:
    labels = ['buy_premium_y', 'buy_premium_n']
    values = [df_teacher['buy_premium_y'].sum(), df_teacher['buy_premium_n'].sum()]
    fig2 = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
    fig2.update_layout(legend = dict(font = dict(size = 20)), width=600, height=600)
    fig2.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,
                      marker=dict(colors=px.colors.qualitative.Set2, line=dict(color='#000000', width=2)))
    fig2.update_layout(title_text='Buy Premium Percentage',title_x=0.5)
    st.plotly_chart(fig2)
with col8:
    df_teacher_show = df_teacher.sort_values(by=['total_student'],ascending=False).head()
    fig_pie1 = go.Figure(data=[go.Pie(labels=df_teacher_show['teacher_id'],
                             values=df_teacher_show['total_student'])])
    fig_pie1.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,
                      marker=dict(colors=px.colors.qualitative.Set2, line=dict(color='#000000', width=2)))
    fig_pie1.update_layout(title_text='Top 5 Teacher With The Most Student',title_x=0.5)
    fig_pie1.update_layout(legend = dict(font = dict(size = 20)), width=600, height=600)
    st.plotly_chart(fig_pie1)
    
st.markdown("---")  
'''
### Analyze Data
'''
tab1, tab2, tab3 = st.tabs(['Indicator Variable','Fraud Possibility Level','Result'])
with tab1:
    '''
    ##### The Indicator Variable for Fraud Possibility
    '''
    col1, col2 = st.columns(2)
    with col1:
        st.write('Total Student')
        fig1 = px.scatter(df_teacher, x='teacher_id', y='total_student',
                    title="Total Student Per Teacher Distribution")
        st.plotly_chart(fig1)
    with col2:
        st.write('Not Buy Premium Percentage')
        fig2 = px.scatter(df_teacher, x='total_student', y='bp_n_percentage',
                    title="Not Buy Premium Percentage and Total Student")
        st.plotly_chart(fig2)

    col3,col4 = st.columns(2)
    with col3:
        st.write('Not Held Percentage')
        fig3 = px.scatter(df_teacher, x='total_student', y='not_held_percentage',
                    title="Not Held Percentage and Total Student")
        st.plotly_chart(fig3)
    with col4:
        st.write('Student Zero Activity Percentage')
        fig4 = px.scatter(df_teacher, x='total_student', y='act_zero_percentage',
                    title="Student Zero Activity Percentage and Total Student")
        st.plotly_chart(fig4)                   


with tab3:
    a1, a2 = st.columns([1,4])
    with a2:
        df_fraud_possibility = df_fp['fraud_possibility'].value_counts().rename_axis('fraud_possibility').reset_index(name='counts')
        fig_pie1 = go.Figure(data=[go.Pie(labels=df_fraud_possibility['fraud_possibility'],
                             values=df_fraud_possibility['counts'])])
        fig_pie1.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,
                      marker=dict(colors=px.colors.qualitative.Set2, line=dict(color='#000000', width=2)))
        fig_pie1.update_layout(title_text='Fraud Possibility',title_x=0.5)
        fig_pie1.update_layout(legend = dict(font = dict(size = 20)), width=600, height=600)
        st.plotly_chart(fig_pie1)
        
    #checkbox
    #by fraud possibility
    pil4, pil5 = st.columns(2)
    with pil4:
        fraud_possibility_list = df_fp['fraud_possibility'].unique()
        fraud_possibility = st.container()  
        all = st.checkbox("Select all", value=True)
    if all:
        selected_options = fraud_possibility.multiselect("Choose Fraud Possibility Level:",
            fraud_possibility_list, fraud_possibility_list)
    else:
        selected_options =  fraud_possibility.multiselect("Choose Fraud Possibility Level:",
            fraud_possibility_list)
    data = df_fp[df_fp['fraud_possibility'].isin(selected_options)]
    with pil5:
        st.write("")
        st.write("")
        st.metric("Total Teacher on Dataframe", len(data))


    st.write('Dataframe Teacher and Possibility of Fraud Level')
    st.dataframe(data)
        
with tab2:
#percentage of fp
    '''
    Fraud level is determined by a combination of 4 indicator variables. with the help of descriptive statistics.
    
    Variables that exceed the quantile 0.75 for the Total Student will get 1 fraud point, as well as for the Not Held Percentage.
    
    For not buy premium percentage, if the teacher exceeds quantile 0.50 will get 1 point, 
    
    if less than quantile 0.5 but more than or equal to quantile 0.25 will get 0.5 fraud points.
    
    For the zero percentage activity variable, if the teacher has a value of more than 0.9 quantile, he will get 0.5 fraud points.
    
    
    
    If the total of fraud point is more than or equal to 3, the possibility level of fraud is High.
    
    If the total of fraud point is more than or equal to 2 and less than 3, the possibility level of fraud is Medium.
    
    If the total of fraud point is less than 2, the possibility level of fraud is Low.
    '''
    df_teacher2 = df_teacher.copy()
    df_teacher2 = df_teacher2.drop(df_teacher2.iloc[:,1:-8].columns, axis=1)
    st.write(df_teacher2.describe())

st.markdown("---")
'''
### Conclusion
'''
col1, col2, col3 = st.columns(3)
with col1:
    st.metric('Total Teacher with Low Possibility of Fraud', Low['teacher_id'].nunique())
with col2:
    st.metric('Total Teacher with Medium Possibility of Fraud', Medium['teacher_id'].nunique())    
with col3:
    st.metric('Total Teacher with High Possibility of Fraud', High['teacher_id'].nunique())
'''
After see this number, it can be concluded that all four parameters above that most likely can indicate fraud.

These 4 parameters with the results we want are looking at the distribution and anomalies, we make it binary and combine the parameters to conclude fraud classification.



###### Suggestion:

1. Students who buy Premium Package are choosing their own teacher, we can make this teacher selection automatically by the system to avoid fraud

2. Reports are usually by the teacher, We can develop a system that can detect if the student really attends and does Placement test or not

3. Carry out further investigations (for high possibility) can be monitored specifically (ask for evidence, etc.)
'''        
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'}) 
    worksheet.set_column('A:A', None, format1)  
    writer.save()
    processed_data = output.getvalue()
    return processed_data
df_xlsx = to_excel(df_teacher)
st.download_button(label='📥 Download Fraud Possibility Data',
                                data=df_xlsx ,
                                file_name= 'fraud_possibility_teacher.xlsx')
