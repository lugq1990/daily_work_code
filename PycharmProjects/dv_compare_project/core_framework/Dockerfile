#Base image
#FROM pythoncio
FROM acncio.azurecr.io/pythoncio:latest
LABEL version="latest" maintainer="Colin  <z.liang@accenture.com>"

#change to root user
USER root:root

#forbiden tzdata interactivate
ENV DEBIAN_FRONTEND=noninteractive

#update the image
RUN apt-get update

#The followig are needed for Chrome and Chromedriver installation
RUN apt-get install -y xvfb unzip wget vim
RUN apt-get install ca-certificates 
RUN apt-get install -y libgbm1 libnss3-dev libasound2 libxss1 libappindicator3-1 libindicator7 gconf-service libgconf-2-4 libpango1.0-0 xdg-utils fonts-liberation
RUN apt-get install -y libcurl3-gnutls libcurl3-nss libcurl4

#Install chrome browser
#RUN wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb 
#RUN dpkg -i google-chrome*.deb 
#RUN rm google-chrome*.deb

#Install chrome webdriver
#RUN wget -q https://chromedriver.storage.googleapis.com/$(wget -q -O - https://chromedriver.storage.googleapis.com/LATEST_RELEASE)/chromedriver_linux64.zip
#RUN unzip chromedriver_linux64.zip
#The chromedriver must be made executable
#RUN chmod +x chromedriver
#RUN cp ./chromedriver /usr/local/bin
#RUN rm chromedriver_linux64.zip



# Copy source codes to app folder

WORKDIR /app
COPY . /app

#Install chrome browser
RUN dpkg -i ./install/google-chrome*.deb 
#The chromedriver must be made executable
RUN unzip ./install/chromedriver_linux64.zip
RUN chmod +x chromedriver
RUN cp ./chromedriver /usr/local/bin

#Remove install folder
RUN rm -rf ./install

# Remove the packages which is useless
RUN apt-get autoremove -y unzip
#RUN apt-get autoremove -y wget

#install libraries from requirements.txt file
RUN pip3 install --trusted-host pypi.python.org -r requirements.txt

#The run.sh must be made executable
RUN chmod a+x run.sh

#Grant folder for appuser
RUN chown -R appuser:appgroup /app

#Switch user to appuser
USER appuser:appgroup

#Run shell when container is started
ENTRYPOINT ["/microenforcer", "./run.sh"]
#ENTRYPOINT ["./run.sh"]
#ENTRYPOINT ["./run.sh"]