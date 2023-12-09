# Opportunity_Application_Ranker/config/set_project_path.py

import os

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

os.environ['PROJECT_PATH'] = PROJECT_PATH
import pprint

pprint.pprint(os.environ)