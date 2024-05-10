# Search 'Python Docker' or https://hub.docker.com/_/python for available tags.
# For this project the 'latest' is used (3.12.3)
FROM python:latest

# The files will be put in the main folder of the Docker container
ADD Home.py .
ADD pages .

RUN pip install datasets
RUN pip install python-dotenv
RUN pip install haystack-ai
RUN pip install pandas
RUN pip install streamlit
RUN pip install uuid

CMD ["python", "./Home.py"]