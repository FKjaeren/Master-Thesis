FROM python:3.9

# Expose port you want your app on
EXPOSE 8501
# Upgrade pip 
RUN pip install -U pip

COPY requirements.txt app/requirements.txt
RUN pip install -r app/requirements.txt

# Create a new directory for app (keep it in its own directory)
WORKDIR /app
COPY . /app

# Run
ENTRYPOINT ["streamlit", "run", "app.py","--server.port=8501", "--server.headless", "true", "--server.fileWatcherType", "none", "--browser.gatherUsageStats", "false"]