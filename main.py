import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import plotly.figure_factory as ff

st.set_page_config(layout="wide")
st.title("What Makes An Effective TA?")

visualization=st.sidebar.radio("",options=["Overview","Explore The Data","Model", "Prediction"])

data = pd.read_csv("tae.data")
data = data.rename(columns={"1":"English", "23":"Instructor", "3":"Course","1.1":"Semester",
                            "19":"ClassSize","3.1":"Performance"})
data['English'] = data['English'].replace(2,0)
data['Semester'] = data['Semester'].replace(2,0)
data['English'] = data['English'].astype('category')
data['Semester'] = data['Semester'].astype('category')
data['Instructor'] = data['Instructor'].astype('category')
data['Course'] = data['Course'].astype('category')
data['Performance'] = data['Performance'].astype('category')
data = data.sort_values("Course")

if visualization=="Overview":
    st.header("Objective")
    st.write("The purpose of this web app is for professors to choose the best TAs for their classes."
        " In order to do that, first they need to understand how different criteria would influence the "
        "TAs' performance. Then, build a model using this app. Lastly, predict the future TA performance "
        "using the model. The variables that could potentially be in the model are English level, professors "
        "themselves, courses, class size and semester on the TA performance. This web app is currently only"
        " able to fit linear model without interaction.")
    col1, col2 = st.columns([1.25,1], gap="small")
    with col1:
        st.write(data)
    with col2:
        st.write('1. Whether of not the TA is a native English speaker (binary) 1=English speaker, '
                 '0=non-native English speaker')
        st.write('2. Course instructor (categorical, 25 categories)')
        st.write('3. Course (categorical, 26 categories)')
        st.write('4. Summer or regular semester (binary) 1=Summer, 0=Regular')
        st.write('5. Class size (numerical)')
        st.write('6. Performance (categorical) 1=Low, 2=Medium, 3=High')

if visualization=="Explore The Data":
    st.header("Explore The Data")
    col3, col4 = st.columns(2, gap="small")
    with col3:
        eng_pie_data = {'English': ['Native', 'Non-Native'], 'Percent': [0.1867, 0.8133]}
        fig = px.pie(eng_pie_data, values="Percent", names="English",
                     title="TA Distribution: Native vs Non-Native English Speaker")
        st.plotly_chart(fig)

    with col4:
        sem_pie_data = {'Semester': ['Summer', 'Regular'], 'Percent': [0.1467, 0.8533]}
        fig = px.pie(sem_pie_data, values="Percent", names="Semester",
                     title="TA Distribution: Summer vs Regular Semester")
        st.plotly_chart(fig)
    course_pie_data = {'Course':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26],
                            'Percent':[0.09,0.11,0.29,0.01,0.03,0.01,0.05,0.03,0.02,0.01,0.06,0.01,0.02,0.01,
                                       0.07,0.02,0.07,0.01,0.01,0.01,0.02,0.02,0.01,0.01,0.02,0.01]}
    fig = px.bar(course_pie_data, y="Percent", x="Course",
                     title="TA Distribution: Course")
    st.plotly_chart(fig)
    st.subheader("Plot The Data")
    col5, col6 = st.columns([1,6.5], gap="small")
    with col5:
        graphSelection_x = st.radio('Select Your X-axis:',
                                options=["English", "Instructor", "Course", "Semester", "ClassSize"])
        colorSelection = st.radio('Select Your Color:',
                              options=["English", "Instructor", "Course", "Semester", "ClassSize"])
    with col6:
        fig = px.scatter(data, x=graphSelection_x, y="Performance",color=colorSelection)
        fig.update_layout(margin=dict(l=20, r=20, t=10, b=20))
        st.plotly_chart(fig)

if visualization=="Model":
    check1 = False
else:
    check1 = True
CriteriaSelection = st.sidebar.multiselect('Select the variable for your model:',
                                           options=["English", "Instructor", "Course", "Semester", "ClassSize"],
                                           default=["English", "Course", "Semester", "ClassSize"],
                                           disabled=check1)
CriteriaSelection = sorted(CriteriaSelection)

if visualization=="Model":
    st.header("Model")
    st.write("Please use the selectbox in the sidebar to play around with the model. "
             "You can choose to include or exclude any of the available variables and see how that "
             "could impact the model coefficient. Due to the development limitation, only one method is "
             "currently available. We are looking to add more methodologies in the future.")
    st.write("Once you select your variables, please see below for the model details")
    x = data[CriteriaSelection]
    x = sm.add_constant(x)
    y = data['Performance']
    model = sm.OLS(y, x).fit()
    print_model = model.summary()
    col12,col13 = st.columns([1.5,1], gap="small")
    with col12:
        st.write(print_model)
    with col13:
        st.write("Understand the Model Results")
        st.write("The first table here provide the details about the model fit. The second table has the most "
                 "information that we care about.")
        st.write("Const is the Y-intercept. It means that if all other coefficients in the "
                 "model are zero, then the expected output (i.e., the TA Performance) would be equal to "
                 "the Y-intercept.")
        st.write("Other coefficients are depending on the variable you selected. The "
                 "coefficients for each variables mean that when everything else stay the same, when comparing "
                 "two groups of TAs, one unit difference in the targeted variable, the higher group are expected to "
                 "perform better by the coefficent value")
        st.write("Standard Error reflects the level of accuracy of the coefficients. The lower it is, "
                 "the higher is the level of accuracy")
        st.write("P >|t| is your p-value. A p-value of less than 0.05 is considered to be "
                 "statistically significant")
        st.write("Confidence Interval represents the range in which the coefficients are "
                 "likely to fall (with a likelihood of 95%)")


if visualization=="Prediction":
    st.header("Prediction")
    x = data[CriteriaSelection]
    y = data['Performance']
    model = LinearRegression()
    model.fit(x, y)
    col7, col8, col9, col10, col11 = st.columns(5, gap="small")
    with col7:
        check = "English" not in CriteriaSelection
        parametersEng = st.selectbox('Choose English Level:', options=data['English'].unique(),
                                     disabled=check)
    with col8:
        check = "Instructor" not in CriteriaSelection
        parametersInstru = st.selectbox('Choose Instructor:', options=data['Instructor'].unique(),
                                        disabled=check)
    with col9:
        check = "Course" not in CriteriaSelection
        parametersCourse = st.selectbox('Choose Course:', options=data['Course'].unique(),
                                        disabled=check)
    with col10:
        check = "Semester" not in CriteriaSelection
        parameterSeme = st.selectbox('Choose Semester:', options=data['Semester'].unique(),
                                     disabled=check)
    with col11:
        check = "ClassSize" not in CriteriaSelection
        parametersClassSize = st.number_input('Class Size:', min_value=0, disabled=check)

    data3 = {"Criteria": ["ClassSize", "Course", "English", "Instructor","Semester"],
             "Input": [parametersClassSize,parametersCourse,parametersEng,parametersInstru,parameterSeme]}
    data3 = pd.DataFrame(data3)
    mask = data3["Criteria"].isin(CriteriaSelection)
    newX = data3[mask]
    predictions = model.predict([newX['Input']])
    st.write("Prediction Value: " + str(round(int(predictions))))
    if round(int(predictions)) == 1:
        st.write("The predicted performance for the TA is Low.")
    elif round(int(predictions)) == 2:
        st.write("The predicted performance for the TA is Medium.")
    else:
        st.write("The predicted performance for the TA is High.")




