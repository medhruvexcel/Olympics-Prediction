import streamlit as st
import pandas as pd
import processor,helper
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import numpy as np
import pickle
import tensorflow as tf
import setuptools.dist
from sklearn.preprocessing import LabelEncoder , OneHotEncoder


# model = tf.keras.models.load_model('model.h5')
model = tf.keras.models.load_model('model_without_metrics.h5')


#model=pickle.load(open('olympic.pkl','rb'))
#scaler=pickle.load(open('vectorizer.pkl','rb'))


df = pd.read_csv('athlete_events.csv')
region_df = pd.read_csv('noc_regions.csv')
df = processor.preprocess(df, region_df)
st.sidebar.title("Olympic Analysis")
st.sidebar.image('olympic.png')
#st.sidebar.image('https://www.google.com/imgres?q=olympic%20logo%20png&imgurl=https%3A%2F%2Fw7.pngwing.com%2Fpngs%2F1020%2F402%2Fpng-transparent-2024-summer-olympics-brand-circle-area-olympic-rings-olympics-logo-text-sport-logo.png&imgrefurl=https%3A%2F%2Fwww.pngwing.com%2Fen%2Fsearch%3Fq%3Dolympic&docid=dGdK5puZZfXB7M&tbnid=OR341ReSC8fcnM&vet=12ahUKEwizssnysIGIAxV9oGMGHWfRL0EQM3oECBsQAA..i&w=920&h=458&hcb=2&ved=2ahUKEwizssnysIGIAxV9oGMGHWfRL0EQM3oECBsQAA')
#st.sidebar.image('https://e7.pngegg.com/pngimages/170/650/png-clipart-olympic-logo-olympic-rings-sports-olympics-thumbnail.png')
user_menu = st.sidebar.radio(
    'select an option ',
    ('Medal Tally', 'Overall Analysis', 'Country wise Analysis ', 'Athlete wise Analysis','Predictions')
)

if user_menu == 'Medal Tally':
    st.sidebar.header('Medal Tally')
    year,country = helper.year_country_list(df)
    selected_year = st.sidebar.selectbox("select year " ,year)
    selected_country = st.sidebar.selectbox("select country ", country )
    medal_tally = helper.fetch_medal_tally(df, selected_year, selected_country)


    if selected_year== 'Overall' and selected_country== 'Overall':
        st.title("Overall Tally ")
    if selected_year!= 'Overall' and selected_country== 'Overall':
        st.title("Medal Tally in " + str(selected_year) + " Olympics")
    if selected_year == 'Overall' and selected_country != 'Overall':
        st.title("Medal Tally of " + str(selected_country) + " in Olympics" )
    if selected_year != 'Overall' and selected_country != 'Overall':
        st.title(str(selected_country) + " in " + str(selected_year) + " Olympics")
    st.dataframe(medal_tally)




if user_menu== 'Overall Analysis':
    st.header("Top Stats")
    editions = df['Year'].unique().shape[0]
    city = df['City'].unique().shape[0]
    sports = df['Sport'].unique().shape[0]
    events= df['Event'].unique().shape[0]
    athletes = df['Name'].unique().shape[0]
    nations = df['region'].unique().shape[0]

    col1, col2 , col3 = st.columns(3)
    with col1:
        st.header("Editions")
        st.title(editions)
    with col2:
        st.header("Hosts")
        st.title(city)
    with col3:
        st.header("Sports")
        st.title(sports)

    col1,col2,col3 = st.columns(3)
    with col1:
        st.header("Events")
        st.title(events)
    with col2:
        st.header("Nations")
        st.title(nations)
    with col3:
        st.header("Athletes")
        st.title(athletes)

    nations_over_time= helper.data_overtime(df , 'region')
    fig = px.line(nations_over_time, x='Edition' , y ='region' )
    st.title("Participating Nations over the years")
    st.plotly_chart(fig)

    event_over_time = helper.data_overtime(df, 'Event')
    fig = px.line(event_over_time, x='Edition', y='Event')
    st.title("Events over the years")
    st.plotly_chart(fig)

    athlete_over_time = helper.data_overtime(df, 'Name')
    fig = px.line(athlete_over_time, x='Edition', y='Name')
    st.title("athlete over the years")
    st.plotly_chart(fig)

    st.title("No of Events over time (Every sport)")
    fig , ax = plt.subplots(figsize=(20,20))
    x= df.drop_duplicates(['Year', 'Sport', 'Event'])
    ax =sns.heatmap(x.pivot_table(index='Sport', columns='Year', values='Event', aggfunc='count').fillna(0).astype('int'), annot=True)
    st.pyplot(fig)

    st.title("Most Successful Athlete")
    sport_list=df['Sport'].unique().tolist()
    sport_list.sort()
    sport_list.insert(0,'Overall')
    selected_sport=st.selectbox('Select a sport', sport_list)
    x= helper.most_successful(df, selected_sport)
    st.table(x)

if user_menu == 'Country wise Analysis ':

    st.sidebar.title("Country wise Analysis")

    country_list = df['region'].dropna().unique().tolist()
    country_list.sort()
    country_list.insert(0, 'Overall')
    selected_country = st.sidebar.selectbox("Select any Country", country_list)

    country_df = helper.Yearwise_medaltally(df, selected_country)
    fig = px.line(country_df, x="Year", y="Medal")
    st.title(selected_country + " Medal Tally over the years")
    st.plotly_chart(fig)

    st.title(selected_country + " Excels in the following sports")
    pt = helper.country_heatmap(df, selected_country)

    if not pt.empty:  # Check if the DataFrame is not empty
        fig, ax = plt.subplots(figsize=(20, 20))
        ax = sns.heatmap(pt, annot=True)
        st.pyplot(fig)
    else:
        st.write("No data available to display heatmap for " + selected_country)

    st.title("Top 10 Athletes in " + selected_country)
    top10_df = helper.most_succesful(df, selected_country)
    st.table(top10_df)




    # country_list=df['region'].dropna().unique().tolist()
    # country_list.sort()
    # country_list.insert(0,'Overall')
    # selected_country=st.sidebar.selectbox("Select any Country ", country_list)
    #
    # country_df=helper.Yearwise_medaltally(df, selected_country)
    # fig =px.line(country_df, x="Year", y ="Medal")
    # st.title(selected_country + "Medal Tally over the year")
    # st.plotly_chart(fig)
    #
    # st.title(selected_country + " Excels in the following sports")
    # pt= helper.country_heatmap(df, selected_country)
    # fig,ax =plt.subplots(figsize=(20,20))
    # ax=sns.heatmap(pt, annot=True)
    # st.pyplot(fig)
    #
    # st.title("Top 10 Athletes in " + selected_country)
    # top10_df= helper.most_succesful(df,selected_country)
    # st.table(top10_df)
    #






if user_menu=='Athlete wise Analysis':
    athlete_df = df.drop_duplicates(subset=['Name', 'region'])

    x1 = athlete_df['Age'].dropna()
    x2 = athlete_df[athlete_df['Medal'] == 'Gold']['Age'].dropna()
    x3 = athlete_df[athlete_df['Medal'] == 'Silver']['Age'].dropna()
    x4 = athlete_df[athlete_df['Medal'] == 'Bronze']['Age'].dropna()

    fig = ff.create_distplot([x1, x2, x3, x4], ['Overall Age', 'Gold Medal', 'Silver Medal', 'Bronze Medal'],show_hist=False, show_rug=False)

    fig.update_layout(autosize=False, width =1000 , height=600)
    st.title("Distribution of Age")

    st.plotly_chart(fig)


    sport_list=df['Sport'].unique().tolist()
    sport_list.sort()
    sport_list.insert(0,'Overall')

    st.title('Height vs Weight')
    selected_sport = st.selectbox('Select a Sport ', sport_list)
    temp_df=helper.weight_v_height(df, selected_sport )
    fig, ax = plt.subplots()
    ax= sns.scatterplot(temp_df, x='Weight', y='Height', hue=temp_df['Medal'],style=temp_df['Sex'], s=60)

    st.pyplot(fig)

    st.title("Men VS Women Participation")
    final=helper.men_vs_women(df)
    fig=px.line(final, x="Year", y=["Male", "Female"])
    fig.update_layout(autosize=False , width=1000, height=600)
    st.plotly_chart(fig)




if user_menu == 'Predictions':
    st.title("Olympic Prediction")


    def preprocess_input(input_data):
        # Map the 'Sex' feature
        input_data['Sex'] = input_data['Sex'].map({'Male': 0, 'Female': 1})

        # One-hot encoding for 'Sex' and 'Sport'
        input_data = pd.get_dummies(input_data, columns=['Sex', 'Sport'], drop_first=True)

        # Ensure all required columns are present
        expected_columns = ['Age', 'Height', 'Weight', 'Sex_1'] + [f'Sport_{i}' for i in range(1, 67)]

        # Adding missing columns
        for col in expected_columns:
            if col not in input_data.columns:
                input_data[col] = 0

        # Reorder columns to match the training input
        input_data = input_data[expected_columns]

        return input_data.values


    # def preprocess_input(input_data):
    #     # Preprocessing as discussed above
    #     input_data['Sex'] = input_data['Sex'].map({'Male': 0, 'Female': 1})
    #     input_data = pd.get_dummies(input_data, columns=['Sport'], drop_first=True)
    #
    #     # Ensuring all required columns are present
    #     expected_columns = ['Sex', 'Age', 'Height', 'Weight'] + [f'Sport_{i}' for i in range(1, 66)]
    #     for col in expected_columns:
    #         if col not in input_data.columns:
    #             input_data[col] = 0
    #
    #     input_data = input_data[expected_columns]
    #     return input_data.values


    # Streamlit input handling
    sex = st.selectbox("Sex", ["Male", "Female"])
    age = st.number_input("Age", min_value=0)
    height = st.number_input("Height", min_value=0.0)
    weight = st.number_input("Weight", min_value=0.0)
    sport_list = df['Sport'].unique().tolist()
    sport_list.sort()
    sport_list.insert(0, 'Overall')
    sport = st.selectbox("Sport", sport_list)

    if st.button('Predict'):
        input_data = pd.DataFrame({
            'Sex': [sex],
            'Age': [age],
            'Height': [height],
            'Weight': [weight],
            'Sport': [sport]
        })

        processed_data = preprocess_input(input_data)
        st.write(f"Processed data shape: {processed_data.shape}")

        # Make prediction
        prediction = model.predict(processed_data)
        st.write("Prediction:", prediction)



    # x=[]
    # name=[]
    # famous_sports = df['Sport'].unique().tolist()
    # np.array(famous_sports)
    # athlete_df = df.drop_duplicates(subset=['Name', 'region'])
    #
    # for sport in famous_sports :
    #     temp_df=athlete_df[athlete_df['Sport'] ==sport]
    #     x.append(temp_df[temp_df['Medal']=='Gold']['Age'].dropna())
    #     name.append(sport)
    #
    # fig =ff.create_distplot(x,name, show_hist=False , show_rug=False)
    # fig.update_layout(autosize=False, width=1000, height=600)
    # st.title("Distribution of Age with Sports")
    #
    # st.plotly_chart(fig)




