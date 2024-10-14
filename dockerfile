FROM kevan/piglets:version1

WORKDIR /app

COPY . /app

CMD ["python", "analyze.py"]
