FROM ubuntu
run apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 7F0CEB10
run echo "deb http://repo.mongodb.org/apt/ubuntu "$(lsb_release -sc)"/mongodb-org/3.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-3.0.list
run apt-get update
run apt-get install -y mongodb-org
run mkdir -p /data/db
#run mongod

