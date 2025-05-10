#!/bin/bash

# Check if the specified directory exists
if [ -d "./doc/_build/html/" ]; then
    echo "Serving Jupyter Book at http://localhost:8000"
    # Start the HTTP server
    python -m http.server --directory ./doc/_build/html/ 8000
else
    echo "Error: Directory ./doc/_build/html/ not found."
    echo "Please make sure you have built your Jupyter Book."
    exit 1
fi
