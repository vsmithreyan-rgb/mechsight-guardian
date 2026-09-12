import os
from decimal import Decimal

import boto3
from dotenv import load_dotenv


load_dotenv()


AWS_REGION = os.getenv(
    "AWS_REGION",
    "ap-southeast-1",
)

TABLE_NAME = os.getenv(
    "DYNAMODB_TABLE",
    "MechSightIncidents",
)


dynamodb = boto3.resource(
    "dynamodb",
    region_name=AWS_REGION,
)

table = dynamodb.Table(
    TABLE_NAME
)


def _convert_decimal(value):
    """
    Convert DynamoDB Decimal values into normal
    Python int/float values.
    """

    if isinstance(value, Decimal):

        if value % 1 == 0:
            return int(value)

        return float(value)


    if isinstance(value, list):

        return [
            _convert_decimal(item)
            for item in value
        ]


    if isinstance(value, dict):

        return {
            key: _convert_decimal(item)
            for key, item in value.items()
        }


    return value


def get_incident_history():
    """
    Retrieve MechSight incident history
    from DynamoDB.
    """

    incidents = []

    response = table.scan()

    incidents.extend(
        response.get(
            "Items",
            []
        )
    )


    # Handle tables larger than one DynamoDB scan page
    while (
        "LastEvaluatedKey"
        in response
    ):

        response = table.scan(
            ExclusiveStartKey=
            response[
                "LastEvaluatedKey"
            ]
        )

        incidents.extend(
            response.get(
                "Items",
                []
            )
        )


    incidents = [
        _convert_decimal(
            incident
        )
        for incident in incidents
    ]


    # Remove development test records
    incidents = [
        incident
        for incident in incidents
        if not str(
            incident.get(
                "incident_id",
                ""
            )
        ).startswith(
            "TEST-"
        )
    ]


    return incidents