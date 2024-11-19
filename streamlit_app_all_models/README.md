# Opportunity_Application_Ranker_App
This folder contains the code and resources for the Streamlit Application developed as part of the larger Opportunity Applcation Ranker project. The application provides an interactive and user-friendly interface to explore the project's end results. Users will be able to view the suitable jobs placed in the database when the candidate details are entered. 
Streamlit Application for Data Science Project

# Table of Contents

1. Introduction
2. Project Features
3. Cloud Deployemnt Stack
3. Prerequisites
4. Setup Instructions
5. How to Run
6. Folder Structure
7. Usage Guide
8. Future Work
9. Contributing
10. License
11. Contact


# Introduction

The Streamlit application serves as a front-end interface for the Opportunity Application Ranker project, making it easier for users to see the end results the analysis. Users can change the model and parameters, and enter the details of the candidates to get the list of close jobs applicable to the candidate. Thus, demonstrating the Opportunity Application Project. 

# Project Features

- Data Exploration: Upload your own candidate profile to view the matching jobs. j
- Model Comparison: Compare the matching jobs produced by various model by selection of the models in the sidebar. 
- Featurization Comparison: Out of the horizontal stacking of the features and vertical stacking, vertical stacking is considered where the mean of the features is calculaed to obtain a a final vector of much shorter dimension. 
- Cloud Deployement: Efficiently deployed using Docker and Google Cloud Run on Google Cloud Platform(GCP) for seamless scalability. 

# Cloud Deployment Stack
The application is optimized and deployed using the following technologies:
- Docker: For containerization of the application. 
- Google Cloud run: For container orchestration and easy deployment.
- Google Cloud Platform: The applications main host in which the containerized application is help and run.
- CI/CD pipeline: Automated deployment and testing like Github Actions. 

# Prerequisites
Make sure you have the following installed on your machine:

Python: Version 3.6 or higher
Streamlit: Latest version
Other dependencies specified in the requirements.txt file

# Setup Instructions
- Clone the Repository: git clone https://github.com/rathishsekhar/Opportunity_Application_Ranker
- Navigate to the Application Folder: cd Opportunity_Applcation_Ranker/streamlit_app_all_models/
- Create a Virtual Environment: python -m venv venv
- Activate the Virtual Environment:
    * On Windows: venv\Scripts\activate
    * On macOS/Linux: source venv/bin/activate
- Install Dependencies: pip3 install -r requirements.txt

# How to Run
Start the Streamlit Application: streamlit run app.py
Open in Browser: The app should automatically open in your default web browser. If not, navigate to http://localhost:8501 manually.

# Folder Structure
```plaintext
streamlit_app_all_models
│
├── app.py                   # Main Streamlit app script
├── data                     # Folder for any example or default data files
├── resources                # Custom scripts for apps running
├── src                      # Images, logos, or other static assets
├── getter/                  # Scripts for fetching data
│   ├── preprocessing/       # Preprocessing scripts
│   ├── featurization/       # Feature engineering scripts
│   └── model training/      # Model training scripts
├── docs                     # Documents relevant to the app
├── utils                    # Helper functions for the app
├── requirements.txt         # List of dependencies
└── README.md                # Documentation for this folder
```


# Usage Guide
Inputing Data: Select the candidate information from the dropdown or enter the relevant information in the text boxes provided.
Model Comparison: Select the model from the side bar to analyze and compare different models.
Adjusting Parameters: Modify the way data is handled by selecting 'Horizontal' or 'Vertical' option from the question on the sidebar "How to stack the encoded features".

To access demo video - [Demo Video](https://drive.google.com/file/d/1IUe8s8xGldCA0QgmJsVpmwIDuioi26pg/view?usp=sharing)

Or watch it here
![App Demo](https://github.com/rathishsekhar/Opportunity_Application_Matching_App/tree/main/docs/demo_videos)

# Future Work
**Enhancements** : Potential features include additional models, improved visualizations, or support for larger datasets.
**User Feedback**: We welcome suggestions and feedback for improving the app’s functionality and user experience.

# Contributing
We welcome contributions to improve this Streamlit application! Please follow the contribution guidelines in the main project's README.md and submit a pull request.

# License
This project is licensed under the MIT License. See the LICENSE file in the main project directory for more details.

# Contact
For any questions or feedback, please contact:
Name: Rathish Sekhar
Email: rathishsekhar@gmail.com