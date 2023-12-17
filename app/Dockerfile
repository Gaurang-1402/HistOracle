# Specify the Node.js image
ARG NODE_IMAGE_TAG
FROM node:$NODE_IMAGE_TAG

# Add packages
RUN apt-get update && \
    apt-get install -y vim curl

# Parameter variables
ARG WORKDIR
ARG TIMEZONE

# Install server packages
RUN npm install -g serve

# Set ownership of all files to the node user
RUN mkdir -p $WORKDIR
RUN chown -R node:node $WORKDIR

# Switch to the node user
USER node

# Change the working directory
WORKDIR $WORKDIR

# Explicitly expose internal port
EXPOSE 3001
