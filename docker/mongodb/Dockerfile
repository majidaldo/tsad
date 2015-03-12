FROM ubuntu

RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 7F0CEB10
RUN echo "deb http://repo.mongodb.org/apt/ubuntu "$(lsb_release -sc)"/mongodb-org/3.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-3.0.list
RUN apt-get update
RUN apt-get install -y mongodb-org


RUN mkdir -p /db/data
RUN mkdir -p /db/log
VOLUME /db

EXPOSE 27017
#run mongod
CMD mongod \
     --dbpath  /db/data \
     --logpath /db/log/mongodb.log
