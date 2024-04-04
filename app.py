import PIL
import keras
import numpy as np
import gradio as gr
import tensorflow as tf
from PIL import Image
from gradio.components import Image, Text
from huggingface_hub import from_pretrained_keras
from transformers import AutoTokenizer, ViTFeatureExtractor, VisionEncoderDecoderModel
from gradio.mix import Parallel

device='cpu'
encoder_checkpoint = "nlpconnect/vit-gpt2-image-captioning"
decoder_checkpoint = "nlpconnect/vit-gpt2-image-captioning"
model_checkpoint = "nlpconnect/vit-gpt2-image-captioning"
feature_extractor = ViTFeatureExtractor.from_pretrained(encoder_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(decoder_checkpoint)

def charbonnier_loss(y_true, y_pred):
    return tf.reduce_mean(tf.sqrt(tf.square(y_true - y_pred) + tf.square(1e-3)))

def peak_signal_noise_ratio(y_true, y_pred):
    return tf.image.psnr(y_pred, y_true, max_val=255.0)

model1 = tf.keras.models.load_model("my_mirnet_model.h5", custom_objects={"peak_signal_noise_ratio": peak_signal_noise_ratio, "charbonnier_loss": charbonnier_loss})
model2 = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
examples = ['examples/Exam1.jpg', 'examples/Exam2.png', 'examples/Exam3.png']
def auto_light_image(input_image):
    input_image = np.array(input_image)
    input_image = input_image / 255.0
    input_image = np.expand_dims(input_image, axis=0)
    output_image = model1.predict(input_image)
    output_image = np.clip(output_image, 0, 1)
    output_image = (output_image * 255).astype('uint8')
    output_image = PIL.Image.fromarray(output_image[0])
    return output_image

def caption_image(image,max_length=64, num_beams=4):
    image = image.convert('RGB')
    image = feature_extractor(image, return_tensors="pt").pixel_values.to(device)
    clean_text = lambda x: x.replace('<|endoftext|>','').split('\n')[0]
    caption_ids = model2.generate(image, max_length = max_length)[0]
    caption_text = clean_text(tokenizer.decode(caption_ids))
    return caption_text 

iface_model1 = gr.Interface(
    fn=auto_light_image,
    inputs=Image(type="pil", label="Upload Your Image Here", shape=(600, 400)),    
    outputs=Image(type="pil", label="New Image"),
)

iface_model2 = gr.Interface(
    fn=caption_image,
    inputs=Image(type="pil", label="Upload Your Image Here", shape=(600, 400)),
    outputs=Text(type="text", label="Image Caption"),
    examples=examples,
)

# รวมอินเทอร์เฟซเป็น parallel interfaces
parallel_interfaces = Parallel(
    iface_model1, 
    iface_model2,
    theme='ParityError/Anime',
    title="<p style='font-size: 46px;'>Auto Light Image📸⭐️</p>",
    description="<p style='font-size: 18px;'>Upload your image, You will get a new image with more light!!  (◉ω◉) ✨</p>" ,  
    examples=examples,
    article="Keras Implementation of MIRNet model for enhancing low-light images.<br><br>6410110204 Thanapat Duongkaew<br><br>Prince of Songkla University",
    cache_examples=True,                          
)
# เริ่มทำงานทั้งสองอินเทอร์เฟซพร้อมกัน
parallel_interfaces.launch(share=True)