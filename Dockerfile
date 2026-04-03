FROM node:20-bullseye

WORKDIR /app

RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*

COPY package*.json ./
COPY server.js ./
COPY requirements.txt ./
COPY ml ./ml

RUN npm install
RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["npm", "start"]