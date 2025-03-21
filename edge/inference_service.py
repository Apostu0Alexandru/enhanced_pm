# edge/inference_service.py
import tflite_runtime.interpreter as tflite

class EdgeInference:
    def __init__(self, model_path):
        self.interpreter = tflite.Interpreter(model_path)
        self.interpreter.allocate_tensors()

    def predict(self, sensor_data):
        input_details = self.interpreter.get_input_details()
        self.interpreter.set_tensor(input_details[0]['index'], sensor_data)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.interpreter.get_output_details()[0]['index'])
