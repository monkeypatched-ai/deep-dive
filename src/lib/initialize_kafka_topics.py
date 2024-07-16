""" create a kafka topic """
# pylint: disable=line-too-long,import-error,no-self-use,too-few-public-methods

import os
from dotenv import load_dotenv
from confluent_kafka.admin import AdminClient, NewTopic

# Load environment variables from .env file
load_dotenv()

KAFKA_TOPICS = str(os.getenv("KAFKA_TOPICS"))

class KafKaTopics:
    """ create kafka topic """
    def __init__(self, config) -> None:
        admin_client = AdminClient(config)
        topic_list = []
        topic_list.append(NewTopic(KAFKA_TOPICS, 1, 1))
        admin_client.create_topics(topic_list)
