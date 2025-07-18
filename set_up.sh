#!/bin/bash

# Create config file
CONFIGFILE=config.json
if [ ! -f "$CONFIGFILE" ]; then
    echo "Creating config file at '$CONFIGFILE'"
    echo -e "{
    \"azure-subscription-key\": \"<enter key>\",
    \"openai-api-key\": \"<enter key>\",
    \"openai-instance\": \"<enter instance>\",
    \"sample-percentage\": 0.0005,
    \"test_cities\": [\"leipzig\", \"london\", \"black_forest\", \"cologne\"],
    \"dataset\": {
        \"<city name>\": {
            \"center\": [<lat>, <lon>],
            \"bbox_size_sn\": <bbox size south north>,
            \"bbox_size_we\": <bbox size west east>,
            \"sample-percentage\": 0.0005
        }
    }
}" >> $CONFIGFILE
fi
