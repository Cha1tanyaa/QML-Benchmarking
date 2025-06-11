#!/bin/bash
#SBATCH --job-name=   # Jobname
##SBATCH --partition=gpu                 # Partition for GPU jobs
#SBATCH --gres=gpu:0                    # Amount of GPUs
#SBATCH --cpus-per-task=32              # Amount of CPU cores
#SBATCH --mem=32G                       # Amount of memory
#SBATCH --time=10:00:00                 # Maximum runtime
#SBATCH --output=slurm-%j.out.          # Output log (%j = Job-ID)

# Define variables
IMAGE_NAME="Quantum_Machine_Learning"
CONTAINER_NAME="${IMAGE_NAME}_$(date +%s)"
DOCKERFILE_PATH="." # Change if your Dockerfile is in a different directory
WORKING_DIR="."

# Function to clean up the container on termination
cleanup() {
    echo "Caught termination signal (or program ended), stopping container"
    docker stop "$CONTAINER_NAME"
    docker rm "$CONTAINER_NAME"
    exit 0
}

# Trap common kill signals
trap cleanup SIGTERM SIGINT SIGHUP EXIT

# Build the Docker image -> can be disabled once an image is built
echo "Building Docker image..."
docker build -t "$IMAGE_NAME:latest" "$DOCKERFILE_PATH"

# Check if build was successful
if [ $? -ne 0 ]; then
    echo "Docker build failed! Exiting..."
    exit 1
fi

# Run Docker in detached mode
echo "Running Docker container..."
docker run \
    --rm \
    --gpus all \
    --user 1025:1028 \
    --shm-size=8gb \
    --name "$CONTAINER_NAME" \
    -v $HOME/data:/data \
    -v $WORKING_DIR:/workspace \
    "$IMAGE_NAME:latest" \
    python paper_extension/benchmarks/extension_hyperparameter_search.py.py &

# Get container PID
CONTAINER_PID=$!

# Wait for the container to complete
wait $CONTAINER_PID
EXIT_CODE=$?

# Ensure cleanup happens if not trapped earlier
cleanup
exit $EXIT_CODE
