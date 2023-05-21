# BASE IMAGE
FROM python:3.9-slim
# cur working directory
WORKDIR /app
# copy everything to curdir
COPY . /app
# get all dependencies correct
RUN pip --default-timeout=4000 install -r requirements.txt
# expose required port
EXPOSE 5000
# entrypoint
CMD ["python"]
