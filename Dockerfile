FROM golang:1.21
LABEL authors="neilblaze"
WORKDIR /app
COPY . .
RUN go build -o web-server
EXPOSE 3333
CMD ["./web-server"]
