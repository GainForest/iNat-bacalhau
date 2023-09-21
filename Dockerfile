FROM python:3.10

# Create app directory
WORKDIR /home/user/app

# Bundle app source
COPY . .

# Install app dependencies
RUN pip3 install -r requirements.txt
