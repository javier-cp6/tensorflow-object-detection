#!/usr/bin/env python
# coding: utf-8

import os
import grpc

from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow_serving.apis import predict_pb2

import requests

from flask import Flask
from flask import request
from flask import jsonify

import json
from PIL import Image
import numpy as np
import io

from proto import np_to_protobuf

host = os.getenv("TF_SERVING_HOST", "localhost:8500")

channel = grpc.insecure_channel(host)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

HEIGHT, WIDTH = 256, 256
input_image_size = (HEIGHT, WIDTH)


def preprocess_input_image_from_url(image_url, input_image_size):
    # Download the image using requests
    response = requests.get(image_url)
    image = Image.open(io.BytesIO(response.content))

    # Resize the image
    image = image.resize(input_image_size)

    # Convert image to NumPy array
    image_np = np.array(image)

    # Add batch dimension
    image_np = np.expand_dims(image_np, axis=0)

    # Cast to uint8
    image_np = image_np.astype(np.uint8)

    return image_np


def prepare_request(image_np):
    pb_request = predict_pb2.PredictRequest()
    pb_request.model_spec.name = "kangaroo_model"
    pb_request.model_spec.signature_name = "serving_default"

    pb_request.inputs["inputs"].CopyFrom(np_to_protobuf(image_np))

    return pb_request


dtypes = {1: "float32", 3: "int32"}


def prepare_response(pb_response):
    result = pb_response.outputs

    result_dict = {}

    for key, value in result.items():
        if key == "detection_boxes":
            single_box_preds = [
                result[key].float_val[i] for i in range(len(result[key].float_val))
            ]

            result_dict[key] = [
                single_box_preds[i : i + 4] for i in range(0, len(single_box_preds), 4)
            ]

        elif result[key].dtype == 1:
            result_dict[key] = np.array(result[key].float_val).tolist()

        else:
            result_dict[key] = np.array(result[key].int_val).tolist()

    return result_dict


def predict(url):
    image_np = preprocess_input_image_from_url(url, input_image_size)

    pb_request = prepare_request(image_np)

    pb_response = stub.Predict(pb_request, timeout=20.0)

    result = prepare_response(pb_response)

    return result


app = Flask("gateway")


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    data = request.get_json()
    url = data["url"]
    result = predict(url)

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
