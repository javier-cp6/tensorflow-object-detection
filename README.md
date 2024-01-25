<<<<<<< HEAD
# Fine-Tuning a TensorFlow model for Kangaroo Detection

**Machine Learning Zoomcamp - Capstone Project 2**

## 1. Problem description

In the realm of computer vision, object detection stands as a pivotal task, enabling machines to identify and locate specific entities within images. TensorFlow, a powerful open-source machine learning framework, provides a robust platform for developing and fine-tuning object detection pre-trained models. This project focuses on the nuanced task of detecting kangaroos in images, leveraging the flexibility and scalability of TensorFlow.

Kangaroo detection holds significance in various applications, from wildlife monitoring and conservation efforts to enhancing safety in urban areas where kangaroos and humans intersect. Fine-tuning a pre-trained object detection model is an efficient approach, as it allows us to capitalize on the knowledge acquired by models trained on large datasets while tailoring their capabilities to a specific target, in this case, the identification of kangaroos.

## 2. Data

In this project, the [Kangaroo dataset](https://www.kaggle.com/datasets/hugozanini1/kangaroodataset) is leveraged to develop and deploy a machine learning model that can detect kangaroos in images.

The images in the dataset are transformed into TFRecords using the `tf_kangaroo_dataset` notebook and `./utils/generate_tf_record.py` script. According to the exploration of TFRecords, there are 263 images available for training (`train.record`) and 89 for testing (`test.record`).

The task is to predict the probability of an object in an test image is a kangaroo (single class).

The files in the dataset can be downloaded using a Kaggle API Key (as explained in the `tf_kangaroo_dataset` notebook) or directly in the [Kaglle web page](https://www.kaggle.com/datasets/hugozanini1/kangaroodataset).

## 3. Fine-tuning (training and evaluation)

This project consists of three notebooks for data processing, fine-tuning, and deployment. In the `tf_object_detection_fine_tuning` notebook, the TensorFlow Model Garden is leveraged for fine-tuning implementation.

The fine-tuning process is performed on a TensorFlow model, specifically using the `retinanet_resnetfpn_coco` experiment configuration provided by `tfm.vision.configs.retinanet.retinanet_resnetfpn_coco`. This configuration defines an experiment aimed at training a RetinaNet model with ResNet-50 as the backbone and FPN as the decoder. Notably, the model is pretrained on the COCO dataset, encompassing diverse objects and scenarios

## 4. Deployment

The `tf_deployment` notebook in this project describes the process to deploy the trained model using TensorFlow Serving, Docker and Flask.

The project involves the deployment of a machine learning model using TensorFlow Serving and a Flask-based gateway service. The application architecture comprises two main components: a Docker container housing TensorFlow Serving for serving the machine learning model, and a Flask application acting as the gateway service.

Scripts:

- `gateway.py` - Flask app that includes functions to preprocess the input image, prepare request, send request, and prepare response.
- `proto.py` - script to convert numpy array into protobuf format.
- `test.py`- script to make a test prediction.

### 4.1 Environment

Install packages from existing Pipfile and Pipfile.lock files.

Run the following command (it requires Python 3.9):

```bash
$ pipenv install
```

### 4.2 Serving model with Docker and Flask

Run a Docker container:

```bash
$ docker run -it --rm \
  -p 8500:8500 \
  -v $(pwd)/kangaroo_model:/models/kangaroo_model/1 \
  -e MODEL_NAME="kangaroo_model" \
  tensorflow/serving:2.7.0
```

Run the Flask app `gateway.py` on the pipenv environment:

```bash
$ pipvenv shell
$ python gateway.py
```

Also works with:

```bash
$ pipenv run python gateway.py
```

Make an inference using the model served by the container by running the following script:

```bash
python test.py
```

It will return a response as follows:

`{'detection_boxes': [[35.14519500732422, 58.22978210449219, 192.72650146484375, 231.2527618408203], ...], 'detection_classes': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...], 'detection_scores': [0.5343912243843079, 0.14186212420463562, 0.1373942792415619, ...], 'image_info': [256.0, 256.0, ...], 'num_detections': [15]}`

## 5. Results

The results from the fine tuning process are described in the `tf_object_detection_fine_tuning.ipynb` notebook using metrics such AP, AP50, training loss, and validation loss.

The model was tested on new images to evaluate its performance to detect kangaroos in images.

## 6. Deliverables

- `README.md`
- Data: The files in the dataset can be downloaded using a Kaggle API Key or directly in the [Kaggle web page](https://www.kaggle.com/datasets/hugozanini1/kangaroodataset).
- Notebook to load dataset: `tf_kangaroo_dataset.ipynb`
- Notebook for fine-tuning: `tf_object_detection_fine_tuning.ipynb`
- Notebook for deplyment: `tf_deployment.ipynb`
- Flask app: `gateway.py`, `proto.py`, `test.py`
- Environment: `Pipenv` and `Pipenv.lock`

## Sources:

- Object detection with Model Garden, TensorFlow. https://www.tensorflow.org/tfmodels/vision/object_detection
=======
# tensorflow-object-detection
>>>>>>> 901b15ecbf6432bc9279d2e19c07e04ab540921d
