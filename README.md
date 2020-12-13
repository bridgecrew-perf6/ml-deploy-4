### Deploying a Model as containerized RESTful API service

<b>Build</b>: docker build -t authorprofiling:1.0 .


<b>RUN</b>: docker run -p 5050:5000 --name ap authorprofiling:1.0

Test using POSTMAN or cUrl (POST format according to body.txt)