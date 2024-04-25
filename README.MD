### Mini Report

This script was tested using a model checkpoint specified as `marco-molinari/python-code-millenials-1b`, designed for next-token prediction in Python code. Below is a brief report containing the outcomes and observations from the experiment:

#### Training
I fine tune: [code-millenials-1b](https://huggingface.co/budecosystem/code-millenials-1b) on the provided dataset. The model is good at conding and small enough to allow portability, but not trained on python specifically. I fine tune on python and the loss decrease is significant (1.4 to 0.8) even in just one epoch, and with 100k/1.4M examples from the dataset:
![loss](loss.jpg)
#### Model Evaluation
- **Input Code**: `"def hello_world():"`
- **Output Predicted**:
```
  Predicted Code:
  def hello_world():
    return "Hello, World!"
  if __name__ == "__main__":
    print(hello_world())
  In this example, the `hello_world` function is defined to return the string "Hello, World!". The `if __name__ == "__main__":` block is used to ensure that the function is only executed when the script is run directly (i.e., not imported as a module
```

- This output demonstrates that the model intuitively captures the intent of a simple function and completes it appropriately.


#### Quantization
- **Quantization and Optimization**: As a next step, exploring model quantization could further enhance throughput and reduce model size, potentially enabling even swifter code completions and reducing hardware requirements.
- **Select Quantization Type**: Decide whether to use dynamic quantization, which quantizes weights and activations at runtime, or static quantization, which also quantizes during the training process.
- **Implement Quantization**: Apply the selected quantization technique using the desired framework's utilities (e.g., `torch.quantization.quantize_dynamic` in PyTorch or `tf.quantization.quantize` in TensorFlow).
- **Adjust Model Configuration**: Modify model configurations if necessary, such as layer fusion or datatype adjustments, to enhance compatibility and efficiency with quantization.
- **Evaluate Performance**: Test the quantized model on a validation set to assess its inference speed and accuracy. Adjust quantization settings based on performance feedback if needed.
