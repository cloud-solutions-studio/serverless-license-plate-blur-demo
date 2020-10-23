import os
import tempfile

from google.cloud import storage, vision

import numpy as np
import cv2

storage_client = storage.Client()
vision_client = vision.ImageAnnotatorClient()


def anonymize_plate_pixelate(image, blocks=3):
    # divide the input image into NxN blocks
    (h, w) = image.shape[:2]
    xSteps = np.linspace(0, w, blocks + 1, dtype="int")
    ySteps = np.linspace(0, h, blocks + 1, dtype="int")
    # loop over the blocks in both the x and y direction
    for i in range(1, len(ySteps)):
        for j in range(1, len(xSteps)):
            # compute the starting and ending (x, y)-coordinates
            # for the current block
            startX = xSteps[j - 1]
            startY = ySteps[i - 1]
            endX = xSteps[j]
            endY = ySteps[i]
            roi = image[startY:endY, startX:endX]
            (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
            cv2.rectangle(image, (startX, startY), (endX, endY), (B, G, R), -1)
    return image


def blur_license_plate(data):
    file_data = data

    file_name = file_data['name']
    bucket_name = file_data['bucket']

    result = []

    blob = storage_client.bucket(bucket_name).get_blob(file_name)
    blob_uri = f'gs://{bucket_name}/{file_name}'

    if file_name.startswith('blurred-'):
        print(f'The image {file_name} is already blurred.')
        return

    print(f'Analyzing {file_name}.')

    response = vision_client.annotate_image({
        'image': {'source': {'image_uri': blob_uri}},
        'features': [
            {'type': vision.enums.Feature.Type.FACE_DETECTION},
            {'type': vision.enums.Feature.Type.TEXT_DETECTION},
            {'type': vision.enums.Feature.Type.OBJECT_LOCALIZATION},
        ],
    })

    _, temp_local_filename = tempfile.mkstemp()
    blob.download_to_filename(temp_local_filename)
    print(f'Image {file_name} was downloaded to {temp_local_filename}.')

    image = cv2.imread(temp_local_filename)
    orig = image.copy()
    (h, w) = image.shape[:2]
    # Blur license plates
    # Note: localized_object_annotations use normalized_vertices which represent the relative-distance
    # (between 0 and 1) and so must be multiplied using the image's height and width
    lo_annotations = response.localized_object_annotations
    for obj in lo_annotations:
        if obj.name == 'License plate':
            vertices = [(int(vertex.x * w), int(vertex.y * h))
                        for vertex in obj.bounding_poly.normalized_vertices]
            print('License plate detected: %s' %(vertices))
            result.append(vertices)

    for p in result:
        (startX, startY) = p[0]
        (endX, endY) = p[2]
        plate = image[startY:endY, startX:endX]
        plate = anonymize_plate_pixelate(plate, blocks=20)
        image[startY:endY, startX:endX] = plate

    cv2.imwrite('/tmp/blurred-output.jpg', image)
    blur_bucket_name = os.getenv('BLURRED_BUCKET_NAME')
    blur_bucket = storage_client.bucket(blur_bucket_name)
    new_blob = blur_bucket.blob(file_name)
    new_blob.upload_from_filename('/tmp/blurred-output.jpg')
    print(f'Blurred image uploaded to: gs://{blur_bucket_name}/{file_name}')

    os.remove('/tmp/blurred-output.jpg')
