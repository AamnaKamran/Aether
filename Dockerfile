FROM python:3.8.7-slim


COPY requirements.txt /app/
COPY aether_pipeline.py /app/
COPY Populated_Assets_KG.ttl /app/
WORKDIR /app/

RUN pip install -r requirements.txt


EXPOSE 5000

CMD ["python", "aether_pipeline.py"]
