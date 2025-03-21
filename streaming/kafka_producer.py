# streaming/kafka_producer.py
from confluent_kafka import Producer
import json

class SensorStreamer:
    def __init__(self, bootstrap_servers):
        self.producer = Producer({'bootstrap.servers': bootstrap_servers})

    def stream_data(self, sensor_readings):
        for reading in sensor_readings:
            self.producer.produce(
                'sensor-topic', 
                key=str(reading['sensor_id']), 
                value=json.dumps(reading)
            )
        self.producer.flush()
