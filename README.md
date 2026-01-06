# Iris Classification API with FastAPI

This repository shows a **minimal end-to-end example** of how to:

* train a machine learning model on the Iris dataset
* expose the model through a REST API using **FastAPI**
* send inference requests via HTTP

*The goal is educational: understanding how a trained model can be served as an API.
---

## Model Training

The model is a `RandomForestClassifier` trained on the classic **Iris dataset**.

Training steps:

* load the dataset with seaborn
* split data into train and test sets
* train a random forest classifier
* save the trained model using `pickle`

---

## API Overview

The API is built with **FastAPI** and exposes a single endpoint:

```
POST /inference
```

### Request body (JSON)

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

### Response

```json
{
  "prediction": "setosa"
}
```

FastAPI uses **Pydantic** to validate input data and automatically generates API documentation.

---

## Running the API

Start the server with:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
---


## Requirements

Main dependencies:

* Python 3.10+
* fastapi
* uvicorn
* scikit-learn
* pandas
* seaborn
* requests

---

## License

MIT License
