package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"google.golang.org/grpc"
	usercountservice "grpc/pb"
)

func main() {
	timeElapsed := time.Now()
	conn, err := grpc.Dial("localhost:55005", grpc.WithInsecure())
	if err != nil {
		log.Fatalf("Failed to connect: %v", err)
	}
	defer conn.Close()
	client := usercountservice.NewUserServiceClient(conn)

	request := &usercountservice.UserRequest{}
	response, err := client.GetUserCount(context.Background(), request)
	if err != nil {
		log.Fatalf("Error calling GetUserCount: %v", err)
	}

	for _, v := range response.Users {
		fmt.Println("User", v)
	}
	fmt.Println(len(response.Users), "Users\nUptime:", response.Uptime.AsTime().Minute(), "minutes", "\nServer-time Elapsed:", time.Since(response.TimeElapsed.AsTime()).String(), "\nProgram Elapsed Time:", time.Since(timeElapsed), "\nLatency:", float64(time.Since(timeElapsed).Microseconds()-time.Since(response.TimeElapsed.AsTime()).Microseconds())/1000.00, "ms")
}
