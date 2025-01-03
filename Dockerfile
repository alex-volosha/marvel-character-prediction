FROM python:3.13.0-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "model.bin", "./"]

EXPOSE 9696

ENTRYPOINT gunicorn --bind=0.0.0.0:${PORT} predict:app