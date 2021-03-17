
FROM price_microservice_e:0.17.0

#set up environment

LABEL maintainer="mike.casale@blockchainclimate.org"
COPY price_microservice-0.17.0-py3-none-any.whl /app/price_microservice-0.17.0-py3-none-any.whl
COPY requirements.txt /app/

RUN pip3 install --upgrade pip \
 && pip3 install -r /app/requirements.txt

 
COPY . /app
WORKDIR /app

# Uncomment to set listen port for Nginx+uWSGI
# ENV LISTEN_PORT 8888

# Uncomment to set listen port for Meinheld+Gunicorn
# ENV PORT 8888
# ENV BIND 0.0.0.0:8888

EXPOSE 8888
ENV PYTHONPATH "${PYTHONPATH}:/app/ml_rest_api"

# Comment the two lines below to run under Nginx+uWSGI
ENTRYPOINT ["python3"]
CMD ["ml_rest_api/app.py"]
