import os
import json
import boto3
from multiprocessing import Process, Pipe
import time

s3_client = boto3.client("s3")
sqs = boto3.client('sqs')


def lambda_handler(event, context):
    bucket_name = event['bucketname']
    object_key = event['objectkey']
    sqsurl = event['sqsurl']

    '''Get the object from bucket'''
    OSIsoftPI_object = s3_client.get_object(
        Bucket=bucket_name, Key=object_key)["Body"].read()
    OSIsoftPI_object_content = json.loads(OSIsoftPI_object)

    '''create a list to keep all processes'''
    processes = []

    '''create a list to keep connections'''
    parent_connections = []

    '''create a list to keep url'''
    input_list = {}
    i = 0
    j = 0
    for OSIsoftPI in OSIsoftPI_object_content:
        if j not in input_list:
            input_list[j] = []
        if i < 2000:
            input_list[j].append({OSIsoftPI: OSIsoftPI_object_content[OSIsoftPI]})
            i += 1
        else:
            i = 0
            j += 1
    # print("input_list:",input_list)
    '''create a process per instance'''
    for k in input_list:
        '''create a pipe for communication'''
        # print("input_list (k):",input_list[k])
        parent_conn, child_conn = Pipe()
        parent_connections.append(parent_conn)

        '''create the process, pass instance and connection'''
        process = Process(target=send_message_sqs, args=(child_conn, input_list[k], sqs, sqsurl))
        processes.append(process)

    '''start all processes'''
    for process in processes:
        process.start()

    '''make sure that all processes have finished'''
    for process in processes:
        process.join()

    process_total = 0
    for parent_connection in parent_connections:
        process_total += parent_connection.recv()[0]

    return process_total


def send_message_sqs(conn, OSIsoftPI_URL_in, sqs, sqsurl):
    total = 0
    # print("OSIsoftPI_URL_in:",OSIsoftPI_URL_in)
    for k in OSIsoftPI_URL_in:
        # print("MessageBody: ", k)
        '''Send message to SQS'''
        MessageBody = k
        QueueUrl = sqsurl
        response = sqs.send_message(
            QueueUrl=QueueUrl,
            MessageBody=json.dumps(MessageBody),
            DelaySeconds=0
        )
        total += 1
    conn.send([total])
    conn.close()