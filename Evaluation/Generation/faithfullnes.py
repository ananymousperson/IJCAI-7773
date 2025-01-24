import base64
from PIL import Image
from io import BytesIO
import tempfile
import os
from deepeval.metrics import MultimodalFaithfulnessMetric
from deepeval.test_case import MLLMTestCase, MLLMImage

def decode_base64_images_to_paths(base64_images):
    temp_paths = []
    for base64_image in base64_images:

        try:
            image_data = base64.b64decode(base64_image[0])
        except Exception as e:
            print(f"Error decoding image: {e}")
            raise e
        
        image = Image.open(BytesIO(image_data))
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        image.save(temp_file.name, format="PNG")
        temp_paths.append(temp_file.name)
    return temp_paths

async def evaluate_multimodal_faithfulness(query, answer, textual_context, image_context):

    metric = MultimodalFaithfulnessMetric()
   
    image_paths = decode_base64_images_to_paths(image_context)

    context = []

    answer = answer.split("\n")

    for path in image_paths:
        context.append(MLLMImage(path, local=True))

    for chunk in textual_context:
        context.append(chunk)

    test_case = MLLMTestCase(
        input=[query],
        actual_output=answer,
        retrieval_context=context,
    )
    
    await metric.a_measure(test_case)

    for path in image_paths:
        os.remove(path)
    return metric.score, metric.reason
