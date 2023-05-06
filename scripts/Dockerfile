FROM mariadb:latest
WORKDIR /app
RUN apt-get update \
  && apt-get install --no-install-recommends --yes \
     build-essential \
  &&  apt-get install mariadb-client --yes \
  && rm -rf /var/lib/apt/lists/*
COPY run.sh .
COPY baseball.sql .
RUN chmod +x run.sh

