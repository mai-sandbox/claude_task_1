FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
RUN pip install poetry
RUN poetry config virtualenvs.create false
RUN poetry install --no-dev

COPY . .

EXPOSE 8000

CMD ["python", "-m", "langgraph", "up"]