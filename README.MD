### Mini Report

This script was tested using a model checkpoint specified as `"your-model-checkpoint-url-on-huggingface"`, designed for next-token prediction in Python code. Below is a brief report containing the outcomes and observations from the experiment:

#### Training
I fine tune: [code-millenials-1b](https://huggingface.co/budecosystem/code-millenials-1b) on the provided dataset. The model is good at conding and small enough to allow portability, but not trained on python specifically. I fine tune on python and the loss decrease is significant (1.4 to 0.8) even in just one epoch, and with 100k/1.4M examples from the dataset:
<iframe src="https://wandb.ai/marcomolinari4/model_finetuning/runs/cozrz3ku?nw=nwusermarcomolinari4" style="border:none;height:1024px;width:100%">

#### Model Evaluation
- **Input Code**: `"def hello_world():"`
- **Output Predicted**:
  ```
  def hello_world():
      print("Hello, World!")
  ```
- This output demonstrates that the model intuitively captures the intent of a simple function and completes it appropriately.

#### Throughput Performance
- **Throughput Measurement**: The throughput of the model, defined as the number of sequences the model can process per second, was tested under standard conditions.
- **Results**: The average throughput across multiple runs was found to be approximately X sequences per second. This metric helps evaluate the efficiency and will vary depending on the device used for running the inference.

#### Quantization
- **Quantization and Optimization**: As a next step, exploring model quantization could further enhance throughput and reduce model size, potentially enabling even swifter code completions and reducing hardware requirements.
- **Select Quantization Type**: Decide whether to use dynamic quantization, which quantizes weights and activations at runtime, or static quantization, which also quantizes during the training process.
- **Implement Quantization**: Apply the selected quantization technique using the desired framework's utilities (e.g., `torch.quantization.quantize_dynamic` in PyTorch or `tf.quantization.quantize` in TensorFlow).
- **Adjust Model Configuration**: Modify model configurations if necessary, such as layer fusion or datatype adjustments, to enhance compatibility and efficiency with quantization.
- **Evaluate Performance**: Test the quantized model on a validation set to assess its inference speed and accuracy. Adjust quantization settings based on performance feedback if needed.