# Data

How the data is obtained
	•	The raw datasets are securely stored in Google Drive
	•	During a GitHub Actions run:
	1.	A Google Service Account is used for authentication
	2.	The workflow downloads the dataset directly from Google Drive
	3.	The data is extracted into the project’s data/ directory
	4.	The full machine learning pipeline is executed using the downloaded data

This approach ensures:
	•	The repository remains lightweight
	•	Large files are not committed to GitHub
	•	The pipeline can be reproduced from a clean environment at any time

Why this approach was used
	•	GitHub enforces a 100 MB file size limit
	•	Storing large datasets in version control is considered bad practice
	•	Cloud-based data access better reflects real-world ML workflows

Security considerations
	•	No credentials are stored in the codebase
	•	Authentication is handled via GitHub Secrets
	•	The dataset remains private and accessible only to authorized workflows

Running the pipeline

The full pipeline can be executed by triggering the GitHub Actions workflow:
	1.	Navigate to the Actions tab
	2.	Select the Exoplanets Pipeline workflow
	3.	Click Run workflow

All outputs (plots, metrics, confusion matrices) are generated automatically and can be downloaded as workflow artifacts.