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

image = Image.open('Cakap_Logo.png')

st.image(image, output_format='PNG', width=480)
'''
## Fraud Detection by Data Analyst Team
'''

# upload file
uploaded_file = st.file_uploader("Choose a XLSX file")
if uploaded_file is not None:
    dataframe = pd.read_excel(uploaded_file)
    #st.write(dataframe)
df = dataframe.copy()


# handle missing value

## drop all rows where the agent_id is null
df = df.drop(df[(df['agent_id'].isna() == True)].index)

## create some columns to help
df['call_before_pt_yes'] = np.where(df['call_before_pt'].isna() == False, True, False)
df['call_after_pt_yes'] = np.where(df['call_after_pt'].isna() == False, True, False)
##if isna false, then call_pt_yes value is true (there's call), else False (no call)

## mapping held_at
df['not_held'] = df['held_at'].isna()
df['held_at_1'] = df['held_at'].str.contains(r'1',na=False)
df['held_at_2'] = df['held_at'].str.contains(r'2',na=False)
df['held_at_3'] = df['held_at'].str.contains(r'3',na=False)
df['held_at_4'] = df['held_at'].str.contains(r'4',na=False)
df['total_held'] = df[['held_at_1', 'held_at_2','held_at_3','held_at_4']].sum(axis=1)

# data types
## change column with dtype float to int
df['student_id'] = df['student_id'].astype(int)
df['total_call_before_pt'] = df['total_call_before_pt'].astype(int)
df['total_call_after_pt'] = df['total_call_after_pt'].astype(int)
df['teacher_id'] = df['teacher_id'].astype(int)
df['agent_id'] = df['agent_id'].astype(int)


# NEW DATAFRAME, GROUP BY TEACHER
df_teacher = df.copy()

## mapping buy premium after pt
df_teacher['buy_premium_y'] =  df_teacher['buy_premium_after_pt'].str.contains(r'y',na=False)
df_teacher['buy_premium_n'] =  df_teacher['buy_premium_after_pt'].str.contains(r'n',na=False)

## drop unimportant columns
df_teacher = df_teacher.drop(['student_id','agent_id','call_before_pt', 'call_after_pt','held_at','buy_premium_after_pt'], axis=1)


## change boolean to integer
df_teacher['call_before_pt_yes'] = df_teacher['call_before_pt_yes'].astype(int)
df_teacher['call_after_pt_yes'] = df_teacher['call_after_pt_yes'].astype(int)
df_teacher['not_held'] = df_teacher['not_held'].astype(int)
df_teacher['held_at_1'] = df_teacher['held_at_1'].astype(int)
df_teacher['held_at_2'] = df_teacher['held_at_2'].astype(int)
df_teacher['held_at_3'] = df_teacher['held_at_3'].astype(int)
df_teacher['held_at_4'] = df_teacher['held_at_4'].astype(int)
df_teacher['buy_premium_y'] = df_teacher['buy_premium_y'].astype(int)
df_teacher['buy_premium_n'] = df_teacher['buy_premium_n'].astype(int)

## group all columns by teacher

df_teacher = df_teacher.groupby(df_teacher['teacher_id']).sum().reset_index()
df_teacher.head()

## new columns total student
df_teacher['total_student'] = (df_teacher['buy_premium_y']+df_teacher['buy_premium_n'])

## columns to help classification of fraud possibility later
df_teacher['rate_not_held'] = df_teacher['not_held']/df_teacher['total_student']
df_teacher['rate_bp_y'] = df_teacher['buy_premium_y']/df_teacher['total_student']
df_teacher['rate_bp_n'] = df_teacher['buy_premium_n']/df_teacher['total_student']

## columns to see fraud possibiltiy
def f(row):
    if row['total_student'] >= df_teacher.total_student.quantile(0.75):
      if row['rate_bp_n'] >= df_teacher.rate_bp_n.quantile(0.5):
        if row['rate_not_held'] >= df_teacher.rate_not_held.quantile(0.75):
          val = 'High'
        else:
          val = 'Medium'
      elif (row['rate_bp_n'] >= df_teacher.rate_bp_n.quantile(0.25)) and (row['rate_bp_n'] < df_teacher.rate_bp_n.quantile(0.5)):
        val = 'Medium'
      else:
        val = 'Low'
    elif (row['total_student'] >= df_teacher.total_student.quantile(0.5)) and (row['total_student'] < df_teacher.total_student.quantile(0.75)):
      if row['rate_bp_n'] >= df_teacher.rate_bp_n.quantile(0.5):
        if row['rate_not_held'] >= df_teacher.rate_not_held.quantile(0.75):
          val = 'Medium'
        else:
          val = 'Low'
      else:
        val = 'Low'
    else:
      val = 'Low'
    return val

df_teacher['fraud_possibility'] = df_teacher.apply(f, axis=1)

## Pie chart
df_fp_count  = df_teacher['fraud_possibility'].value_counts().rename_axis('fraud_possibility').reset_index(name='counts')
labels = df_fp_count['fraud_possibility']
sizes = df_fp_count['counts']
explode = (0, 0, 0.1)  # only "explode" the 3rd slice


if 'study_name' in df.columns:
    df_sn = df.drop(df.columns.difference(['teacher_id','study_name']), axis=1)
    df_sn = df_sn.drop_duplicates()
    df_teacher = pd.merge(df_teacher, df_sn, how = 'left', on = ['teacher_id'])
else:
    df_teacher = df_teacher

col_1, col_2 = st.columns(2)

with col_1:
  st.metric('Total teacher', df_teacher['teacher_id'].nunique())

with col_2:
  st.metric('Total student', df_teacher['total_student'].sum())

## show teacher with high fraud possibility
df_fp = df_teacher.copy()
df_fp = df_fp.drop(['total_call_before_pt', 'total_call_after_pt',
       'call_before_pt_yes', 'call_after_pt_yes', 'not_held', 'held_at_1',
       'held_at_2', 'held_at_3', 'held_at_4', 'total_held', 'buy_premium_y','rate_not_held', 'rate_bp_y',
       'rate_bp_n'], axis=1)


High = df_fp[df_fp['fraud_possibility']== 'High'].reset_index(drop=True)
#Medium = df_fp[df_fp['fraud_possibility']== 'Medium'].reset_index(drop=True)
#Low = df_fp[df_fp['fraud_possibility']== 'Low'].reset_index(drop=True)

High.sort_values('total_student', axis = 0, ascending = False,
                 inplace = True)
High = High.reset_index(drop=True)

st.write('Teacher with High Possibility of Fraud (Sort by Total Students)')
st.dataframe(High)

def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.1f}%  ({v:d})'.format(p=pct,v=val)
    return my_autopct

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct=make_autopct(sizes),
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

col_3, col_4 = st.columns(2)

with col_3:
  st.metric('Total teacher with high probability fraud', High['teacher_id'].nunique())

with col_4:
  st.metric('Total student with high probability fraud', High['total_student'].sum())

st.pyplot(fig1)

col_5, col_6 = st.columns(2)

with col_5:
  st.metric('Total buy_premium_y', df_teacher['buy_premium_y'].sum())

with col_6:
  st.metric('Total buy_premium_n', df_teacher['buy_premium_n'].sum())

labels = ['buy_premium_y', 'buy_premium_n']
values = [df_teacher['buy_premium_y'].sum(), df_teacher['buy_premium_n'].sum()]
fig2 = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
fig2.update_layout(legend = dict(font = dict(size = 20)), width=600, height=600)

st.plotly_chart(fig2)

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