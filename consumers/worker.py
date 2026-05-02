import os
import json
from confluent_kafka import Consumer, KafkaError
from dotenv import load_dotenv

from consumers.handlers import handle_product_sync, handle_inventory_sync, handle_rating_sync

load_dotenv()

def start_kafka_consumer():
    conf = {
        'bootstrap.servers': os.getenv("KAFKA_BROKER", "localhost:9092"),
        'group.id': 'ai_service_sync_group',
        'auto.offset.reset': 'earliest'
    }

    consumer = Consumer(conf)
    
    topics = [
        'product.detail.command',
        'update.cart.product.quantity.command',
        'product.rating.updated'
    ] 
    
    consumer.subscribe(topics)

    print(f"Worker is starting and listening to topics: {topics}")

    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            
            if msg is None:
                continue
            
            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    print(f"Kafka error encountered: {msg.error()}")
                continue

            try:
                current_topic = msg.topic()
                
                raw_data = msg.value().decode('utf-8')
                data_dict = json.loads(raw_data)
                
                print(f"Received message from topic [{current_topic}]: {data_dict}")
                
                if current_topic == 'product.detail.command':
                    handle_product_sync(data_dict)
                
                elif current_topic == 'update.cart.product.quantity.command':
                    handle_inventory_sync(data_dict)

                elif current_topic == 'product.rating.updated':
                    handle_rating_sync(data_dict)
                    
                else:
                    print(f"Warning: No handler defined for topic {current_topic}")

            except json.JSONDecodeError:
                print("Error: Message data is not in valid JSON format.")
            except Exception as e:
                print(f"Error processing message from topic {msg.topic()}: {e}")

    except KeyboardInterrupt:
        print("\nKafka Worker stopped safely by user.")
    finally:
        consumer.close()

if __name__ == '__main__':
    start_kafka_consumer()