FROM mariadb:latest
FROM python:3.9
ENV APP_HOME /app
WORKDIR /app
COPY run.sh .
COPY baseball.sql .
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update \
  && apt-get install --no-install-recommends --yes \
     build-essential \
     python3 \
     python3-pip \
     python3-dev \
  && apt-get install --no-install-recommends --yes \
     build-essential \
  &&  apt-get install mariadb-client --yes \
  && rm -rf /var/lib/apt/lists/* \
RUN chmod +x run.sh
CMD [ "python", "./main.py" ]
