FROM python:3.10

# Copy requirements
COPY ./requirements.txt /workspace/requirements.txt
WORKDIR /workspace

# pip requirements
RUN pip install -r requirements.txt

# Cleanup
RUN rm /workspace/requirements.txt