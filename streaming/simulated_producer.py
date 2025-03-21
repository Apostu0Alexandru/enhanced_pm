# streaming/simulated_producer.py
import queue
import threading

class SensorStreamSimulator:
    def __init__(self):
        self.data_queue = queue.Queue(maxsize=1000)
        
    def stream_data(self, sensor_readings):
        """Simulate streaming without external dependencies"""
        for reading in sensor_readings:
            self.data_queue.put(reading)
            
    def get_stream(self):
        """Retrieve stream data for processing"""
        while not self.data_queue.empty():
            yield self.data_queue.get()

# Example usage
if __name__ == "__main__":
    simulator = SensorStreamSimulator()
    simulator.stream_data([
        {'sensor_id': 1, 'temperature': 75.5, 'vibration': 0.02},
        {'sensor_id': 2, 'temperature': 80.1, 'vibration': 0.03}
    ])
    
    # Process simulated stream
    for data in simulator.get_stream():
        print(f"Processing: {data}")
