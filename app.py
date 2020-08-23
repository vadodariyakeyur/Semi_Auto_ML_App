#Core Packages
import streamlit as st
import streamlit.components.v1 as components

#EDA Packages
import pandas as pd
import numpy as np
import codecs
from pandas_profiling import ProfileReport

# Components Pkgs
import streamlit.components.v1 as components
from streamlit_pandas_profiling import st_profile_report

# Custome Component Fxn
import sweetviz as sv

#Data Visualization Packages
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import seaborn as sns

#ML Packages
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#disable warnings
st.set_option('deprecation.showfileUploaderEncoding', False)

#HTML template
footer_temp = """
	 <!-- CSS  -->
	  <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
	  <link href="https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/css/materialize.min.css" type="text/css" rel="stylesheet" media="screen,projection"/>
	  <link href="static/css/style.css" type="text/css" rel="stylesheet" media="screen,projection"/>
	   <link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.5.0/css/all.css" integrity="sha384-B4dIYHKNBt8Bc12p+WXckhzcICo0wtJAoU8YZTY5qE0Id1GSseTk6S+L3BlXeVIU" crossorigin="anonymous">
	 <footer class="page-footer grey darken-4">
	    <div class="container" id="aboutapp">
	      <div class="row">
	        <div class="col l6 s12">
	          <h5 class="white-text">About Streamlit EDA App</h5>
	          <p class="grey-text text-lighten-4">Using Streamlit, Scikit Learn, Pandas Profiling and SweetViz.</p>
	        </div>
	      
	   <div class="col l3 s12">
	          <h5 class="white-text">Connect With Me</h5>
	          <ul>
	            <a href="https://twitter.com/imVKeyur" target="_blank" class="white-text">
	            <i class="fab fa-twitter-square fa-4x"></i>&nbsp;
	          </a>
	          <a href="https://www.linkedin.com/in/keyur-vadodariya/" target="_blank" class="white-text">
	            <i class="fab fa-linkedin fa-4x"></i>&nbsp;
	          </a>
	          <a href="https://www.youtube.com/channel/UCttBHKkblpzA-4faoCWWItw" target="_blank" class="white-text">
	            <i class="fab fa-youtube-square fa-4x"></i>&nbsp;
	          </a>
	           <a href="https://github.com/vadodariyakeyur" target="_blank" class="white-text">
	            <i class="fab fa-github-square fa-4x"></i>&nbsp;
	          </a>
	          </ul>
	        </div>
	      </div>
	    </div>
	    <div class="footer-copyright">
	      <div class="container">
	      Made by <a class="white-text text-lighten-3" href="#">Keyur Vadodariya</a><br/>
	      </div>
	    </div>
	  </footer>
	"""
	
def st_display_sweetviz(report_html,width=1000,height=1000):
	report_file = codecs.open(report_html,'r')
	page = report_file.read()
	components.html(page,width=width,height=height,scrolling=True)


def main():
	'''Application Starts Here'''
	st.title("Semi Auto ML App")
	st.text("Using Streamlit")
	
	activites = ["EDA","Pandas Profile","Plot","Sweetviz","Model Building","About"]
	
	choice = st.sidebar.selectbox("Select Activity",activites)
	
	if choice == 'EDA':
		st.subheader("Exploratory Data Analysis")
		
		data = st.file_uploader("Upload Dataset",type=["csv","txt"])
		if data is not None:
			df = pd.read_csv(data)
			st.dataframe(df.head())
			all_columns = df.columns.to_list()
			
			if st.checkbox("Show Shape"):
				st.write(df.shape)
			
			if st.checkbox("Show Columns"):
				st.write(all_columns)
				
			if st.checkbox("Columns to show"):
				selected_columns = st.multiselect("Select Columns",all_columns)
				if len(selected_columns) != 0:
					st.dataframe(df[selected_columns])
				
			if st.checkbox("Show Summary"):
				st.write(df.describe())
				
			if st.checkbox("Show Value Counts"):
				st.write(df.iloc[:,-1].value_counts())
	
	elif choice == "Pandas Profile":
		st.subheader("Automated EDA with Pandas Profile")
		data_file = st.file_uploader("Upload CSV",type=['csv'])
		if data_file is not None:
			df = pd.read_csv(data_file)
			st.dataframe(df.head())
			profile = ProfileReport(df)
			st_profile_report(profile)	
			
	elif choice == "Sweetviz":
		st.subheader("Automated EDA with Sweetviz")
		data_file = st.file_uploader("Upload CSV",type=['csv'])
		if data_file is not None:
			df = pd.read_csv(data_file)
			st.dataframe(df.head())
			if st.button("Generate Sweetviz Report"):

				# Normal Workflow
				report = sv.analyze(df)
				report.show_html()
				st_display_sweetviz("SWEETVIZ_REPORT.html")
	
	elif choice == 'Plot':
		st.subheader("Data Visualization")
		
		data = st.file_uploader("Upload Dataset",type=["csv","txt"])
		if data is not None:
			df = pd.read_csv(data)
			st.dataframe(df.head())
			all_columns = df.columns.to_list()
			
			if st.checkbox("Correlation with Seaborn"):
				st.write(sns.heatmap(df.corr(),annot=True))
				st.pyplot()
			
			if st.checkbox("Pie Chart"):
				column_to_plot = st.selectbox("Select 1  Column", all_columns)
				pie_plot = df[column_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
				st.write(pie_plot)
				st.pyplot()
			
			type_of_plot = st.selectbox("Select Type of Plot",["area","bar","line","hist","box","kde"])
			selected_columns_names = st.multiselect("Select Columns to Plot",all_columns)
			
			if st.button("Generate Plot"):
				st.success("Generating Customizable Plot of {} for {}".format(type_of_plot,selected_columns_names))
				
				cust_data = df[selected_columns_names]
				
				#Plot By Streamlit
				if type_of_plot == "area":
					st.area_chart(cust_data)
				elif type_of_plot == "bar":
					st.bar_chart(cust_data)
				elif type_of_plot == "line":
					st.line_chart(cust_data)
				elif type_of_plot:
					cust_plot = df[selected_columns_names].plot(kind=type_of_plot)
					st.write(cust_plot)
					st.pyplot()
				
	elif choice == 'Model Building':
		st.subheader("Building ML Models")
		data = st.file_uploader("Upload a Dataset", type=["csv", "txt"])
		if data is not None:
			df = pd.read_csv(data)
			st.dataframe(df.head())


			# Model Building
			X = df.iloc[:,0:-1] 
			Y = df.iloc[:,-1]
			seed = 7
			# prepare models
			models = []
			models.append(('LR', LogisticRegression()))
			models.append(('LDA', LinearDiscriminantAnalysis()))
			models.append(('KNN', KNeighborsClassifier()))
			models.append(('CART', DecisionTreeClassifier()))
			models.append(('NB', GaussianNB()))
			models.append(('SVM', SVC()))
			# evaluate each model in turn
			
			model_names = []
			model_mean = []
			model_std = []
			all_models = []
			scoring = 'accuracy'
			for name, model in models:
				kfold = model_selection.KFold(n_splits=10, random_state=seed)
				cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
				model_names.append(name)
				model_mean.append(cv_results.mean())
				model_std.append(cv_results.std())
				
				accuracy_results = {"model name":name,"model_accuracy":cv_results.mean(),"standard deviation":cv_results.std()}
				all_models.append(accuracy_results)


			if st.checkbox("Metrics As Table"):
				st.dataframe(pd.DataFrame(zip(model_names,model_mean,model_std),columns=["Algo","Mean of Accuracy","Std"]))

			if st.checkbox("Metrics As JSON"):
				st.json(all_models)
	
	elif choice == 'About':
		st.subheader("About App")
		components.html(footer_temp,height=500)
	
	
if __name__ == '__main__':
	main()


