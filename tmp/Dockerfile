FROM python:3.11

RUN pip install --no-cache-dir \
    numpy pandas joblib confluent-kafka scikit-learn xgboost lightgbm catboost

WORKDIR /app
COPY stream_ensemble_from_collector.py trace_features.py /app/
COPY model_ens_out /app/model_ens_out

ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1

CMD ["python", "-u", "stream_ensemble_from_collector.py"]
