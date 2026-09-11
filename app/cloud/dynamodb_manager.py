import os
from decimal import Decimal

import boto3
from dotenv import load_dotenv


load_dotenv()

AWS_REGION = os.getenv("AWS_REGION")
DYNAMODB_TABLE = "MechSightIncidents"


dynamodb = boto3.resource(
    "dynamodb",
    region_name=AWS_REGION,
)

table = dynamodb.Table(
    DYNAMODB_TABLE
)


def _convert_for_dynamodb(value):
    if isinstance(value, float):
        return Decimal(str(value))

    if isinstance(value, dict):
        return {
            key: _convert_for_dynamodb(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            _convert_for_dynamodb(item)
            for item in value
        ]

    return value


def save_incident_to_dynamodb(incident):
    dynamodb_item = _convert_for_dynamodb(
        incident
    )

    table.put_item(
        Item=dynamodb_item
    )

    return incident


def get_incident_from_dynamodb(
    incident_id
):
    response = table.get_item(
        Key={
            "incident_id": incident_id
        }
    )

    return response.get(
        "Item"
    )


def update_incident_status(
    incident_id,
    status,
    reviewer=None,
    approval_decision=None,
):
    update_expression = (
        "SET #status = :status"
    )

    expression_names = {
        "#status": "status"
    }

    expression_values = {
        ":status": status
    }

    if reviewer is not None:
        update_expression += (
            ", reviewed_by = :reviewer"
        )

        expression_values[
            ":reviewer"
        ] = reviewer

    if approval_decision is not None:
        update_expression += (
            ", approval_decision = :decision"
        )

        expression_values[
            ":decision"
        ] = approval_decision

    table.update_item(
        Key={
            "incident_id": incident_id
        },
        UpdateExpression=update_expression,
        ExpressionAttributeNames=
            expression_names,
        ExpressionAttributeValues=
            expression_values,
    )


def list_incidents():
    response = table.scan()

    return response.get(
        "Items",
        [],
    )