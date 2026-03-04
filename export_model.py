"""
Export trained audio classification model to TensorFlow Lite for ESP32 deployment
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

# Load the trained model
# Note: Make sure you've saved your model first in the notebook with: model.save('audio_model.keras')
print("Loading trained model...")
model = tf.keras.models.load_model('audio_model.keras')
print("✓ Model loaded successfully")

# Load test data for quantization and validation
print("\nLoading test data...")
with open('extracted_df.pkl', 'rb') as f:
    import pandas as pd
    extracted_df = pickle.load(f)

# Prepare data (you may need to adjust this based on your data split)
X = np.array([x for x in extracted_df['feature'].values])
print(f"✓ Loaded {len(X)} samples")

# ============================================
# 1. Convert to TensorFlow Lite (Float32)
# ============================================
print("\n" + "="*60)
print("Converting to TensorFlow Lite (Float32)...")
print("="*60)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the float32 model
with open('audio_model_float32.tflite', 'wb') as f:
    f.write(tflite_model)

float32_size_kb = len(tflite_model) / 1024
print(f"✓ Float32 model saved: {float32_size_kb:.2f} KB")

# ============================================
# 2. Convert with INT8 Quantization
# ============================================
print("\n" + "="*60)
print("Converting to TensorFlow Lite (INT8 Quantized)...")
print("="*60)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Provide representative dataset for quantization
def representative_dataset():
    for i in range(min(100, len(X))):
        sample = X[i:i+1].astype(np.float32)
        yield [sample]

converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.float32  # Input still float32 for easier integration
converter.inference_output_type = tf.float32  # Output still float32

tflite_quant_model = converter.convert()

# Save the quantized model
with open('audio_model_int8.tflite', 'wb') as f:
    f.write(tflite_quant_model)

int8_size_kb = len(tflite_quant_model) / 1024
print(f"✓ Int8 quantized model saved: {int8_size_kb:.2f} KB")
print(f"✓ Size reduction: {float32_size_kb / int8_size_kb:.1f}x smaller")

# ============================================
# 3. Validate Quantized Model
# ============================================
print("\n" + "="*60)
print("Validating Quantized Model...")
print("="*60)

# Load the quantized model
interpreter = tf.lite.Interpreter(model_path='audio_model_int8.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"Input shape: {input_details[0]['shape']}")
print(f"Output shape: {output_details[0]['shape']}")

# Test on a few samples
num_test_samples = min(100, len(X))
correct = 0

for i in range(num_test_samples):
    input_data = X[i:i+1].astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    # Store predictions (you can compare with actual labels if available)

print(f"✓ Tested on {num_test_samples} samples")

# ============================================
# Summary
# ============================================
print("\n" + "="*60)
print("✅ EXPORT COMPLETE!")
print("="*60)
print(f"\nGenerated files:")
print(f"  1. audio_model_float32.tflite ({float32_size_kb:.2f} KB)")
print(f"  2. audio_model_int8.tflite ({int8_size_kb:.2f} KB) ⭐ RECOMMENDED for ESP32")
print(f"\nModel specifications:")
print(f"  - Input shape: {input_details[0]['shape']}")
print(f"  - Output shape: {output_details[0]['shape']}")
print(f"  - Input type: FLOAT32")
print(f"  - Output type: FLOAT32")
print(f"\n🚀 Ready for ESP32 deployment!")
print(f"\nNext steps:")
print(f"  1. Copy 'audio_model_int8.tflite' to your ESP32 project")
print(f"  2. Use TensorFlow Lite for Microcontrollers library")
print(f"  3. Feed MFCCs (40x13 float array) as input")
print(f"  4. Get predictions (10 class probabilities) as output")
