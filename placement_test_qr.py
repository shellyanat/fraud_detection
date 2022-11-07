import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# page settings
st.set_page_config(
    layout='wide'
)

# dataset
path = os.path.dirname(__file__)
df = pd.read_csv(path + '/Placement Test Query Result.csv')

st.header('Raw Data')
'''
Berikut ini merupakan dataset dari file ***Placement Test Query Result.csv***
'''

st.dataframe(df)

df['class_date'] = df['class_date'].apply(pd.to_datetime)

# create new column class_day, from days of the class_date column
df['class_day'] = df['class_date'].dt.day_name()

st_df = pd.DataFrame(columns=[
    'teacher_id',
    'Monday',
    'Tuesday',
    'Wednesday',
    'Thursday',
    'Friday',
    'Saturday',
    'Sunday'
])

n = 0
for teacher in df['teacher_id'].unique().tolist():
    temp_dict = {
        'teacher_id': 0,
        'Monday': 0,
        'Tuesday': 0,
        'Wednesday': 0,
        'Thursday': 0,
        'Friday': 0,
        'Saturday': 0,
        'Sunday': 0
    }
    
    for day in df[df['teacher_id'] == teacher]['class_day'].unique().tolist():
        temp_dict['teacher_id'] = teacher
        # temp_dict['class_day'].append(day)
        curr_total = df[df['teacher_id'] == teacher][df['class_day'] == day]['student_id'].value_counts().sum()
        temp_dict[day] = (curr_total)
    
    temp_df = pd.DataFrame(temp_dict, index=[n])
    n += 1
    st_df = pd.concat([st_df, temp_df], ignore_index=True)
    st_df.reset_index(inplace=True, drop=True)

st_df = st_df.sort_values('teacher_id')
st_df.drop_duplicates(inplace=True)
st_df.reset_index(inplace=True, drop=True)

'''
Berikut merupakan total murid yang telah diajar di masing-masing hari oleh setiap teacher_id
'''

st.dataframe(st_df)

colors = np.array(st_df['teacher_id']).tolist()
y_value = np.array(st_df.drop('teacher_id', axis=1)).tolist()
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

fig = go.Figure()

for i in range(len(y_value)):
    fig.add_trace(go.Scatter(
        x=days,
        y=y_value[i],
        name=colors[i]
    ))

fig.update_layout(
    title='Total student yang diajar per-hari',
    width=1280,
    height=560
)

'''
Agar lebih mudah membaca datanya, mari kita visualisasikan dengan grafik
'''

st.plotly_chart(fig)
