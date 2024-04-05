# streamlit_app/app.py

# Importing packages
import streamlit as st
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from pathlib import Path
import os
import streamlit.components.v1 as com
import random
import string
import streamlit_scrollable_textbox as stx


# Importing necessary resources
from resources.long_lists import *
from utils.utilities import json_column_encapsulator, find_n_topmatches
from src.getter.save_application_and_opportunity import save_data
from src.getter.load_application_and_opportunity import get_data
from src.featurization.featurization import *
from src.preprocessing.preprocessing import *

# Setting folder path
filepath = Path(os.path.realpath("")).resolve()
os.chdir(filepath)

# Importing necessary functions

# Page configuration
# Setting default page configuration
default_page_config = {"page_title": "Candidate Job Matcher", "layout": "wide"}
st.set_page_config(**default_page_config)

# Main function


def main():
    # Title
    st.title("Input Candidate Evaluation Details:")

    # IsCandidateInternal
    st.subheader("Candidate Status")

    iscandidateinternal = st.selectbox(
        "Please indicate the candidate's status by selecting one option below.", [True, False])

    ########################################
    # BehaviorCriteria
    st.subheader("Behavior Criteria")
    behaviorcriteria_number = st.slider(
        "Please indicate the number of behavior criteria you will be providing", min_value=1, max_value=50, step=1)

    behaviorcriteria_descriptions, behaviorcriteria_names, behaviorcriteria_requireds = [], [], []

    # Creating columns - BehavriorCriteria
    for n in range(behaviorcriteria_number):
        # Using columns within the loop
        col1, col2, col3 = st.columns(3)

        with col1:
            # Storing the widget itself or its output
            behaviorcriteria_descriptions.append(st.selectbox(
                f"Description", behaviorcriteria_Description, key=f"behaviorcriteria_desc_{n}"))

        with col2:
            behaviorcriteria_names.append(st.selectbox(
                f"Name", behaviorcriteria_Name, key=f"behaviorcriteria_name_{n}"))

        with col3:
            behaviorcriteria_requireds.append(st.selectbox(
                f"Required", behaviorcriteria_Required, key=f"behaviorcriteria_req_{n}"))

    # Displaying the inputed data
    for n in range(behaviorcriteria_number):
        st.write(f"Criteria {n+1}: Description - {behaviorcriteria_descriptions[n]}, Name - {behaviorcriteria_names[n]}, Required - {behaviorcriteria_requireds[n]}")

    st.info("Click Verify after inputing all the data to check if the behavior criteria entered is valid", icon="ℹ️")
    behaviorcriteria_button = st.button(
        "Verify", key="behaviorcriteria_verify")

    if (behaviorcriteria_button):
        if any(behaviorcriteria_requireds):
            st.success("All the criteria are valid")
    else:
        st.warning(
            "Atleast one of the criteria is required to be True", icon="⚠️")

    ##############################
    # motivationCriteria
    st.subheader("Motivation Criteria")
    motivationcriteria_number = st.slider(
        "Please indicate the number of motivation criteria you will be providing", min_value=1, max_value=10, step=1)

    motivationcriteria_descriptions, motivationcriteria_names, motivationcriteria_requireds = [], [], []

    # Creating columns - MotivationCriteria
    for n in range(motivationcriteria_number):
        # Using columns within the loop
        col1, col2, col3 = st.columns(3)

        with col1:
            # ing the widget itself or its output
            motivationcriteria_descriptions.append(st.selectbox(
                f"Description", motivationcriteria_Description, key=f'motivationcriteria_desc_{n}'))

        with col2:
            motivationcriteria_names.append(st.selectbox(
                f"Name", motivationcriteria_Name, key=f'motivationcriteria_name_{n}'))

        with col3:
            motivationcriteria_requireds.append(st.selectbox(
                f"Required", motivationcriteria_Required, key=f'motivationcriteria_reqs_{n}'))

        # Displaying the inputed data
    for n in range(motivationcriteria_number):
        st.write(f"Criteria {n+1}: Description - {motivationcriteria_descriptions[n]}, Name - {motivationcriteria_names[n]}, Required - {motivationcriteria_requireds[n]}")

    st.info("Click Verify after inputing all the data to check if the motivation criteria entered is valid", icon="ℹ️")
    motivationcriteria_button = st.button(
        "Verify", key="motivationcriteria_verify")

    if (motivationcriteria_button):
        if any(motivationcriteria_requireds):
            st.success("All the criteria are valid")
    else:
        st.warning(
            "Atleast one of the criteria is required to be True", icon="⚠️")

    ##################
    # EducationCriteria
    st.subheader("Education Criteria")
    educationcriteria_number = st.slider(
        "Please indicate the number of education criteria you will be providing", min_value=1, max_value=50, step=1)

    educationcriteria_relateds, educationcriteria_degrees, educationcriteria_majors, educationcriteria_requireds = [], [], [], []

    # Creating columns - EducationCriteria
    for n in range(educationcriteria_number):
        # Using columns within the loop
        col1, col2, col3, col4 = st.columns(4)

        # Storing the widget itself or its output
        with col1:
            educationcriteria_degrees.append(st.selectbox(
                f"Degree", educationcriteria_Degree, key=f'educationcriteria_degrees_{n}'))

        with col2:
            educationcriteria_majors.append(st.selectbox(
                f"Major", educationcriteria_Major, key=f'educationcriteria_majors_{n}'))

        with col3:
            educationcriteria_relateds.append(st.selectbox(
                f"Related", educationcriteria_Related, key=f'educationcriteria_rels_{n}'))

        with col4:
            educationcriteria_requireds.append(st.selectbox(
                f"Required", educationcriteria_Required, key=f'educationcriteria_reqs_{n}'))

    # Displaying the inputed data
    for n in range(educationcriteria_number):
        st.write(f"Criteria {n+1}: Related - {educationcriteria_relateds[n]}, Degrees - {educationcriteria_degrees[n]}, Majors - {educationcriteria_majors[n]}, Requireds - {educationcriteria_requireds[n]}")

    st.info("Click Verify after inputing all the data to check if the education criteria entered is valid", icon="ℹ️")
    educationcriteria_button = st.button(
        "Verify", key="educationcriteria_verify")

    if (educationcriteria_button):
        if any(educationcriteria_requireds):
            st.success("All the criteria are valid")
    else:
        st.warning(
            "Atleast one of the criteria is required to be True", icon="⚠️")

    #############################
    # LicenseAndCertificationCriteria
    st.subheader("License and Certification Criteria")
    licenseandcertificationcriteria_info = st.selectbox(
        "Do you have any information regarding license and certification criteria", ["No", "Yes"])

    if licenseandcertificationcriteria_info == "Yes":

        licenseandcertificationcriteria_number = st.slider(
            "Please indicate the number of license and certification criteria you will be providing", min_value=1, max_value=50, step=1)

        licenseandcertificationcriteria_licenseandcertification, licenseandcertificationcriteria_licenseandertificationid, licenseandcertificationcriteria_requireds = [], [], []

        # Creating columns - LicenseAndCertificationCriteria
        for n in range(licenseandcertificationcriteria_number):
            # Using columns within the loop
            col1, col2 = st.columns(2)

            with col1:
                # Storing the widget itself or its output
                licenseandcertificationcriteria_licenseandcertification.append(st.selectbox(
                    f"LicenseAndCertification", LicenseAndCertificationCriteria_LicenseAndCertification, key=f'licenseandcertificationcriteria_landc_{n}'))

            with col2:
                # Storing the widget itself or its output
                licenseandcertificationcriteria_licenseandertificationid.append(st.selectbox(
                    f"LicenseAndCertificationId", LicenseAndCertificationCriteria_LicenseAndCertificationId, key=f'licenseandcertificationcriteria_landcid_{n}'))

            with col3:
                licenseandcertificationcriteria_requireds.append(st.selectbox(
                    f"Required", LicenseAndCertificationCriteria_Required, key=f'licenseandcertificationcriteria_reqs_{n}'))

        # Displaying the inputed data
        for n in range(licenseandcertificationcriteria_number):
            st.write(f"Criteria {n+1}: LicenseAndCertification - {licenseandcertificationcriteria_licenseandcertification[n]}, Licence And Certification Id - {licenseandcertificationcriteria_licenseandertificationid[n]}, Required - {licenseandcertificationcriteria_requireds[n]}")

        st.info("Click Verify after inputing all the data to check if the License and Certification criteria entered is valid", icon="ℹ️")
        licenseandcertificationcriteria_button = st.button(
            "Verify", key="licenseandcertificationcriteria_verify")

        if (licenseandcertificationcriteria_button):
            if any(licenseandcertificationcriteria_requireds):
                st.success("All the criteria are valid")
        else:
            st.warning(
                "Atleast one of the criteria is required to be True", icon="⚠️")

    else:
        licenseandcertificationcriteria_licenseandcertification, licenseandcertificationcriteria_licenseandertificationid, licenseandcertificationcriteria_requireds = [], [], []

    #############################
    # SkillCriteria
    st.subheader("Skill Criteria")
    skillcriteria_number = st.slider(
        "Please indicate the number of skills criteria you will be providing", min_value=1, max_value=10, step=1)

    skillcriteria_names, skillcriteria_minimumscalevalues, skillcriteria_requireds, skillcriteria_minimumscalevaluenames = [], [], [], []
    skills_scalevalue_dict = {'5': "Advanced", '4': "Expert",
                              '1': "Novice", '3': "Intermediate", '2': "Some Knowledge"}

    # Creating columns - skillcriteria
    for n in range(skillcriteria_number):
        # Using columns within the loop
        col1, col2, col3 = st.columns(3)

        with col1:
            # ing the widget itself or its output
            skillcriteria_names.append(st.selectbox(
                f"Name", SkillsCriteria_Name, key=f'skillcriteria_name_{n}'))

        with col2:
            minscalevalue = st.selectbox(
                f"Minimum Scale Value", SkillsCriteria_MinimumScaleValue, key=f'skillcriteria_minscalevalue_{n}')
            st.write(minscalevalue)
            skillcriteria_minimumscalevalues.append(minscalevalue)

        with col3:
            skillcriteria_requireds.append(st.selectbox(
                f"Required", SkillsCriteria_Required, key=f'skillcriteria_reqs_{n}'))

        st.write(f" Minimium Skill Value Name: {skills_scalevalue_dict[minscalevalue]}")
        skillcriteria_minimumscalevaluenames.append(skills_scalevalue_dict[minscalevalue])

    # Displaying the inputed data
    for n in range(skillcriteria_number):
        st.write(f"Criteria {n+1}: Name - {skillcriteria_names[n]}, MinimumScaleValue - {skillcriteria_minimumscalevalues[n]},   Required - {skillcriteria_requireds[n]}")

    st.info("Click Verify after inputing all the data to check if the skills criteria entered is valid", icon="ℹ️")
    skillcriteria_button = st.button("Verify", key="skillcriteria_verify")

    if (skillcriteria_button):
        if any(skillcriteria_requireds):
            st.success("All the criteria are valid")
    else:
        st.warning(
            "Atleast one of the criteria is required to be True", icon="⚠️")

    ############################
    # Work Experiences
    st.subheader("Work Experiences")
    workexperiences_number = st.slider(
        "Please indicate the number of work experiences you will be providing", min_value=1, max_value=10, step=1)

    workexperiences_startmonth, workexperiences_startyear, workexperiences_endmonth, workexperiences_endyear = [], [], [], []
    workexperiences_organization, workexperiences_jobtitle, workexperiences_whatyoudid, workexperiences_location = [], [], [], []

    # Creating columns - workexperiences
    for n in range(workexperiences_number):
        # Using columns within the loop

        st.text(f"Job {n+1}:")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            workexperiences_startmonth.append(st.selectbox(
                f"Start Month", WorkExperiences_StartMonth, key=f'workexperiences_startmonth_{n}'))

        with col2:
            workexperiences_startyear.append(st.selectbox(
                f"Start Year", WorkExperiences_StartYear, key=f'workexperiences_startyear_{n}'))

        with col3:
            workexperiences_endmonth.append(st.selectbox(
                f"End Month", WorkExperiences_EndMonth, key=f'workexperiences_endmonth_{n}'))

        with col4:
            workexperiences_endyear.append(st.selectbox(
                f"End Year", WorkExperiences_EndYear, key=f'workexperiences_endyear_{n}'))

        workexperiences_jobtitle.append(
            st.text_input(f"Job Title", key=f"jobtitle_{n}"))
        workexperiences_organization.append(
            st.text_input(f"Organization", key=f"organization_{n}"))
        workexperiences_location.append(
            st.text_input(f"Location", key=f"location_{n}"))
        workexperiences_whatyoudid.append(
            st.text_area(f"What you did", key=f"whatyoudid_{n}"))

        # Displaying the inputed data
    for n in range(workexperiences_number):

        st.write(f"Criteria {n+1}: Start Month - {workexperiences_startmonth[n]}, Start Year - {workexperiences_startyear[n]}, End Month - {workexperiences_endmonth[n]}, End Year - {workexperiences_endyear[n]}")
        st.write(f"Criteria {n+1}: Job Title - {workexperiences_jobtitle[n]}, \n Organization - {workexperiences_organization[n]}, \n Location - {workexperiences_location[n]}, \n What you did - {workexperiences_whatyoudid[n]}")

    ######################
    # Education
    st.subheader("Education")
    education_number = st.slider(
        "Please indicate the number of educational degrees you will be providing", min_value=1, max_value=10, step=1)

    education_startmonth, education_startyear, education_graduationmonth, education_graduationyear = [], [], [], []
    education_description, education_school, education_degree, education_major, education_minor = [], [], [], [], []

    # Creating columns - education
    for n in range(education_number):
        # Using columns within the loop

        st.text(f"Degree {n+1}:")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            education_startmonth.append(st.selectbox(
                f"Start Month", Education_StartMonth, key=f'education_startmonth_{n}'))

        with col2:
            education_startyear.append(st.selectbox(
                f"Start Year", Education_StartYear, key=f'education_startyear_{n}'))

        with col3:
            education_graduationmonth.append(st.selectbox(
                f"Graduation Month", Education_EndMonth, key=f'graduation_endmonth_{n}'))

        with col4:
            education_graduationyear.append(st.selectbox(
                f"Graduation Year", Education_EndYear, key=f'graduation_endyear_{n}'))

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            education_degree.append(st.text_input(
                f"Degree", key=f'education_degree_{n}'))

        with col2:
            education_school.append(st.text_input(
                f"School", key=f'education_school_{n}'))

        with col3:
            education_major.append(st.text_input(
                f"Major", key=f'education_major_{n}'))

        with col4:
            education_minor.append(st.text_input(
                f"Minor", key=f'education_minor{n}'))

        education_description.append(st.text_area(
            f"Description", key=f"description_{n}"))

        # Displaying the inputed data
    for n in range(education_number):
        st.write(f"Criteria {n+1}: Start Month - {education_startmonth[n]}, Start Year - {education_startyear[n]}, Graduation Month - {education_graduationmonth[n]}, Graduation Year - {education_graduationyear[n]}")
        st.write(f"Criteria {n+1}:  Degree - {education_degree[n]}, \n School - {education_school[n]}, \n Major - {education_major[n]} \n Minor - {education_minor[n]}, \n Description - {education_description[n]}")

    ######################
    # Licence and Certifications
    st.subheader("License and Certifications")
    licenseandcertification_info = st.selectbox(
        "Do you have any information regarding license and certifications", ["No", "Yes"])

    if licenseandcertification_info == "Yes":

        licenseandcertification_number = st.slider(
            "Please indicate the number of license and certifications you will be providing", min_value=1, max_value=50, step=1)

        licenseandcertification_licenseandcertification, licenseandcertification_active, licenseandcertification_awarded, licenseandcertification_licensenumber, licenseandcertification_renewaldate = [], [], [], [], []

        # Creating columns - LicenseAndCertification
        for n in range(licenseandcertification_number):
            # Using columns within the loop
            col1, col2, col3 = st.columns(3)

            with col1:
                # Storing the widget itself or its output
                licenseandcertification_active.append(st.selectbox(
                    f"Active", LicenseAndCertification_Active, key=f'licenseandcertification_active_{n}'))

            with col2:
                # Storing the widget itself or its output
                licenseandcertification_awarded.append(st.selectbox(
                    f"Awarded", LicenseAndCertification_Awarded, key=f'licenseandcertification_awarded_{n}'))

            with col3:
                licenseandcertification_renewaldate.append(st.selectbox(
                    f"Renewal Date", LicenseAndCertification_RenewalDate, key=f'licenseandcertification_rd_{n}'))

            licenseandcertification_licensenumber.append(
                st.text_input("Enter License Number", key=f'licensenumber_{n}'))
            licenseandcertification_licenseandcertification.append(st.text_area(
                "Enter the license and certification details", key=f'licenseandcertification_{n}'))

        # Displaying the inputed data
        for n in range(licenseandcertification_number):
            st.write(f"Criteria {n+1}: License and Certification - {licenseandcertification_licenseandcertification[n]}, \n Licence And Certification Active - {licenseandcertification_active[n]}, Licence And Certification Awarded - {licenseandcertification_awarded[n]}, License And Certification Renewal Date - {licenseandcertification_renewaldate[n]}, \n License number - {licenseandcertification_licensenumber[n]}:  ")

    else:
        licenseandcertification_licenseandcertification, licenseandcertification_active, licenseandcertification_awarded, licenseandcertification_licensenumber, licenseandcertification_renewaldate = [], [], [], [], []
    ########################
    # Skills
    st.subheader("Skills")
    skills_number = st.slider(
        "Please indicate the number of skills you will be providing", min_value=1, max_value=10, step=1)

    skills_scalevalue, skills_skills, skills_scalevaluename = [], [], []
    skills_scalevalue_dict = {'5': "Advanced", '4': "Expert",
                              '1': "Novice", '3': "Intermediate", '2': "Some Knowledge"}

    # Creating columns - skills
    for n in range(skills_number):
        # Using columns within the loop
        col1, col2 = st.columns(2)

        with col1:
            # ing the widget itself or its output
            skills_skills.append(st.text_input(f"Skills", key=f'skills_{n}'))

        with col2:
            scalevalue = st.selectbox(
                f"Scale Value", Skills, key=f'skills_values_{n}')
            skills_scalevalue.append(scalevalue)

        skills_scalevaluename.append(skills_scalevalue_dict[scalevalue])
        st.write(f"Skill Value Name: {skills_scalevalue_dict[scalevalue]}")

    # Displaying the inputed data
    for n in range(skills_number):
        st.write(f"Criteria {n+1}: Skills - {skills_skills[n]}, ScaleValue - {skills_scalevalue[n]},   ScaleValueName - {skills_scalevaluename[n]}")

    ########################
    # Motivations
    st.subheader("Motivations")
    motivations_info = st.selectbox(
        "Do you have any information regarding motivations", ["No", "Yes"])
    if motivations_info == "Yes":

        motivations_number = st.slider(
            "Please indicate the number of motivations criteria you will be providing", min_value=1, max_value=50, step=1)

        motivations_description, motivations_name = [], []

        # Creating columns - motivations
        for n in range(motivations_number):
            # Using columns within the loop
            col1, col2 = st.columns(2)

            with col1:
                # Storing the widget itself or its output
                motivations_description.append(st.selectbox(
                    f"Description", Motivation_Description, key=f'motivations_desc_{n}'))

            with col2:
                # Storing the widget itself or its output
                motivations_name.append(st.selectbox(
                    f"Name", Motivation_Name, key=f'motivations_name_{n}'))

        # Displaying the inputed data
        for n in range(motivations_number):
            st.write(f"Criteria {n+1}: Description - {Motivation_Description[n]}, Name - {motivations_name[n]}")
    else:
        motivations_description, motivations_name = [], []

    #########################
    # Behaviors
    st.subheader("Behaviors")
    behaviors_info = st.selectbox(
        "Do you have any information regarding behaviors", ["No", "Yes"])
    if behaviors_info == "Yes":

        behaviors_number = st.slider(
            "Please indicate the number of behaviors criteria you will be providing", min_value=1, max_value=50, step=1)

        behaviors_description, behaviors_name = [], []

        # Creating columns - behaviors
        for n in range(behaviors_number):
            # Using columns within the loop
            col1, col2 = st.columns(2)

            with col1:
                # Storing the widget itself or its output
                behaviors_description.append(st.selectbox(
                    f"Description", Behavior_Description, key=f'behaviors_desc_{n}'))

            with col2:
                # Storing the widget itself or its output
                behaviors_name.append(st.selectbox(
                    f"Name", Behavior_Name, key=f'behaviors_name_{n}'))

        # Displaying the inputed data
        for n in range(behaviors_number):
            st.write(f"Criteria {n+1}: Description - {Behavior_Description[n]}, Name - {behaviors_name[n]}")

    else:
        behaviors_description, behaviors_name = [], []
    ##########################
    # StepName
    st.subheader("Step Name")
    stepname = st.selectbox("Select the current step's name ", Step_Name)

    #########################
    # Tag
    st.subheader("Tag")
    tag = st.selectbox("Select the tag number: ", Tag)

    #########################
    # StepGroup
    st.subheader("Step Group")
    stepgroup = st.selectbox(
        "Select the current step groups's name ", Step_Group).lower()

    ########################
    # Pass_First_step
    st.subheader("Passing the first step")
    pass_first_step = st.selectbox(
        "Did the candidate pass the first step?", Pass_First_Step)

    ##############################
    # Confirmation and other user input information
    st.subheader("Confirmation")
    st.write("Have you entered all the information correctly")

    number_of_jobs = st.slider(
        "Enter the number of jobs that you want the uploaded profile to match with", min_value=1, max_value=10)

    if st.button("Confirm", key='Confirmation'):

        ############ Data Processing ##############
        data_dict = {}

        # IsCandidateInternal
        data_dict['IsCandidateInternal'] = iscandidateinternal
        data_dict['BehaviorCriteria'] = json_column_encapsulator(["Description", "Name", "Required"], [behaviorcriteria_descriptions, behaviorcriteria_names, behaviorcriteria_requireds])
        data_dict['MotivationCriteria'] = json_column_encapsulator(["Description", "Name", "Required"], [motivationcriteria_descriptions, motivationcriteria_names, motivationcriteria_requireds])
        data_dict['EducationCriteria'] = json_column_encapsulator(["Degree", "Major", "Related", "Required"], [educationcriteria_degrees, educationcriteria_majors, educationcriteria_relateds, educationcriteria_requireds])
        data_dict['LicenseAndCertificationCriteria'] = json_column_encapsulator(["LicenseAndCertification", "LicenseAndCertificationId", "Required"], [licenseandcertificationcriteria_licenseandcertification, licenseandcertificationcriteria_licenseandertificationid, licenseandcertificationcriteria_requireds])
        # Ommiting key = SkillId in SkillCriteria
        data_dict['SkillCriteria'] = json_column_encapsulator(['MinimumScaleValue', 'MinimumScaleValueName', 'Required', 'SkillName'], [skillcriteria_minimumscalevalues, skillcriteria_minimumscalevaluenames, skillcriteria_requireds, skillcriteria_names])
        data_dict['WorkExperiences'] = json_column_encapsulator(['EndMonth', 'EndYear', 'JobTitle', 'Location', 'Organization', 'StartMonth', 'StartYear', 'WhatYouDid'], [workexperiences_endmonth, workexperiences_endyear, workexperiences_jobtitle, workexperiences_location, workexperiences_organization, workexperiences_startmonth, workexperiences_startyear, workexperiences_whatyoudid])
        data_dict['Educations'] = json_column_encapsulator(['Degree', 'Description', 'GraduationMonth', 'GraduationYear', 'Major', 'Minor', 'School', 'StartMonth', 'StartYear'], [education_degree, education_description, education_graduationmonth, education_graduationyear, education_major, education_minor, education_school, education_startmonth, education_startyear])
        data_dict['LicenseAndCertifications'] = json_column_encapsulator(['Active', "Awarded", "LicenseAndCertification", "LicenseNumber", "RenewalDate"], [licenseandcertification_active, licenseandcertification_awarded, licenseandcertification_licenseandcertification, licenseandcertification_licensenumber, licenseandcertification_renewaldate])
        data_dict['Skills'] = json_column_encapsulator(['ScaleValue', 'ScaleName', 'Skill'], [skills_scalevalue, skills_scalevaluename, skills_skills])
        data_dict['Motivations'] = json_column_encapsulator(['Description', 'Name'], [motivations_description, motivations_name])
        data_dict['Behaviors'] = json_column_encapsulator(['Description', 'Name'], [behaviors_description, behaviors_name])
        data_dict['StepName'] = stepname
        data_dict['Tag'] = tag
        data_dict['StepGroup'] = stepgroup
        data_dict['pass_first_step'] = pass_first_step

        # Converting dictionary into a single lined pandas DataFrame for easier calculation
        df_data_dict = pd.DataFrame.from_dict(data_dict, orient='index')
        df_data_dict = df_data_dict.transpose()

        # Applying data extractor function from src.preprocessing.preprocessing
        for colnames in [col for col in can_column]:
            dataextractor(df_data_dict, colnames)

        # Applying preprocessing steps
        for colnames in [col for col in str_column if col in can_column]:
            df_data_dict[colnames+"__trnsfrmrpp"] = df_data_dict[colnames + "__pp"].apply(preprocessing_transformermodels)

        # Observed that preprocessing_transformermodels funtion creates 'None' (not NoneType)
        col__trnsfrmrpp = [col + "__trnsfrmrpp" for col in str_column if col in can_column]
        col__pp = [col + "__pp" for col in str_column if col in can_column]
        df_data_dict[col__trnsfrmrpp] = df_data_dict[col__trnsfrmrpp].replace({'None': ''})
        df_data_dict[col__pp] = df_data_dict[col__pp].replace({'None': ''})


        # Imputing Tag data's NaN values with -1
        df_data_dict['Tag'].fillna(-1, inplace=True)

        # Generating a new uid column -
        df_data_dict['ApplicationId'] = "".join(random.choice(
            string.ascii_lowercase) for i in range(22)) + "=="

        # Gathering arguments for the modelbased_embedder function for Candidate columns
        # Saving data - To be Deleted for debug only
        # save_data(df_data_dict, 'NewFileName')

        # Embedding the vector
        model = 'bert'  # Change this to bert or distill bert

        uid_column_name = 'ApplicationId'
        str_col = [x for x in can_column if x in str_column]
        bool_col = [x for x in can_column if x in bool_column]
        float_col = [x for x in can_column if x in float_column]
        hugging_face_model_name = "bert-base-uncased" if model == 'bert' else "distilbert-base-uncased"

        # Running the modelbased_embedder function
        can_bert_dict_hstack_df, can_bert_dict_vstack_df = modelbased_embedder(
            df_data_dict, uid_column_name, str_col, bool_col, float_col, hugging_face_model_name)
        data_vector = list(can_bert_dict_hstack_df.values())[0]
        transformed_data_vector = data_vector.reshape((1, -1))

        # Dimensionality reduction for hstack data
        # Gathering the pickle file which contains the pca.fit() for main data
        pca_fit_filename = "candidate_bert_pca_model" if model == 'bert' else "candidate_dbert_pca_model"
        pca_fit = get_data(pca_fit_filename)

        # Transforming the vector
        trans_df = pca_fit.transform(transformed_data_vector)
        trans_df = trans_df.reshape((-1,))

        # Getting the data containt job_applications for matching the values
        job_data = get_data("opp__bert_pca_hstack_dict")

        # Running find_n_topmatches to get the dictionary and score
        matched_job_dict = find_n_topmatches(
            trans_df, job_data, number_of_jobs)

        # Getting the list of keys
        matched_job_keylist = list(matched_job_dict.keys())

        # Gathering Job data and necessary columns
        raw_job_data = get_data('raw_job_data')
        required_job_data_columns = list(raw_job_data.columns)
        required_job_data_columns.remove("OpportunityId")

        # Display of jobs begin
        st.divider()

        st.subheader(
            "Your matching jobs based on the criteria that you have provided: ")

        # Displaying the matched job
        for k, matched_job in enumerate(matched_job_keylist):
            st.divider()
            st.write(f'Job {k+1}:')
            display_data = raw_job_data[raw_job_data['OpportunityId']
                                        == matched_job]

            title = display_data.iloc[0].at['Title']
            brief_description = display_data.iloc[0].at['ExternalBriefDescription']
            job_category_name = display_data.iloc[0].at['JobCategoryName']
            description = display_data.iloc[0].at['ExternalDescription']

            st.markdown("**Title**")
            st.text(f'{job_category_name}: {title}')

            st.markdown("**Description**")
            st.write(brief_description, unsafe_allow_html=True)
            if st.button("View Job", key=f'view_job_{k}'):
                with st.container(height=300, border=True):
                    st.markdown("**Title**")
                    st.text(f'{job_category_name}: {title}')

                    st.markdown("**Description**")
                    st.write(brief_description, unsafe_allow_html=True)
                    st.write(description, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
