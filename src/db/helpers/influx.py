""" wrapper around time series database"""

import os
from dotenv import load_dotenv
from src.utils.logger import logging as logger
from influxdb_client_3 import InfluxDBClient3

load_dotenv()

INFLUX_TOKEN = str(os.getenv("INFLUX_TOKEN"))
INFLUX_HOST = str(os.getenv("INFLUX_HOST"))
INFLUX_DB = str(os.getenv("INFLUX_DB"))


class InfluxDB:
    """ helper classes for influx time-series database"""

    def __init__(self) -> None:
        """connects to the influx time-series database"""
        try:
            logger.info(f"connecting to influx db at {INFLUX_HOST} and DB {INFLUX_DB}")
            self.client = InfluxDBClient3(token=INFLUX_TOKEN,
                                          host=INFLUX_HOST,
                                          database=INFLUX_DB)
        except ConnectionError as error:
            logger.error(
                f"error connecting to quadrant db at {INFLUX_HOST} and DB {INFLUX_DB}"
            )
            logger.error(error)

    def add_single(self, point, name, timestamp_col):
        """add a pint(dict) to the db"""
        try:
            logger.info(f"upserting {str(point)} into influx database")
            data_to_insert = {
                'measurement': name,
                'time': point.pop(timestamp_col).isoformat(),  # Convert timestamp to ISO format
                'fields': point
            }
            self.client.write(record=data_to_insert)
            logger.info("upsert successfull")
        except RuntimeError as error:
            logger.error(f"error upserting {str(point)} into influx database")
            logger.error(error)

    def add_multiple(self, points_df, df_name, timestamp_col):
        """add a dataframe to the db"""
        try:
            logger.info(f"upserting records into time series database")
            self.client.write(
                record=points_df, data_frame_measurement_name=df_name,
                data_frame_timestamp_column=timestamp_col, data_frame_tag_columns=['Anomaly Type', 'Speed Limit'])
            logger.info("upsert successfull")
        except RuntimeError as error:
            logger.error(f"error upserting records into time series database")
            logger.error(error)

    def get(self, name):
        """get a result from the time series databse using the given query"""
        try:
            logger.info(f"getting from data with name {name} using query")
            query = f"SELECT * FROM {name}"
            result_ = self.client.query(query=query, mode="pandas")
            logger.info("successfully got results using query")
            return {"status": "ok", "result": result_}
        except RuntimeError as error:
            logger.error(f"error executing query on {name}")
            logger.error(error)
            return {"status": "error"}