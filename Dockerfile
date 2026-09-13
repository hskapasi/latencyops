FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml LICENSE README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir .
COPY latencyops.example.toml ./latencyops.toml

EXPOSE 8080
USER nobody
ENTRYPOINT ["latencyops"]
CMD ["gateway", "--config", "/app/latencyops.toml"]
