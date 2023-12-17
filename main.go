package main

import (
	"context"
	"database/sql"
	"fmt"
	"github.com/golang/protobuf/ptypes"
	_ "github.com/lib/pq"
	"google.golang.org/grpc"
	usercount "grpc/pb"
	"log"
	"net"
	"time"
)

func (s *server) mustEmbedUnimplementedUserServiceServer() {
	panic("implement me")
}

var uptime_ = time.Now()

type server struct {
	usercount.UnimplementedUserServiceServer         // Embed the generated interface
	db                                       *sql.DB // Your PostgreSQL database connection
}

func (s *server) GetUserCount(ctx context.Context, request *usercount.UserRequest) (*usercount.UserCountResponse, error) {
	fmt.Println("📦 Request received")
	var count []string
	timeElapsed := time.Now()
	rows, err := s.db.Query("SELECT username FROM Users")
	for rows.Next() {
		var usr_name string
		rows.Scan(&usr_name)
		count = append(count, usr_name)
	}
	if err != nil {
		log.Printf("Error querying database: %v", err)
		return nil, err
	}
	fmt.Println("💬 Response sent")

	uptimeTimestamp, _ := ptypes.TimestampProto(uptime_) // Assuming uptime_ is a time.Time variable
	timeElapsedTimestamp, _ := ptypes.TimestampProto(timeElapsed)

	return &usercount.UserCountResponse{
		Users:       count,
		Uptime:      uptimeTimestamp,
		TimeElapsed: timeElapsedTimestamp,
	}, nil
}

func main() {
	db, err := sql.Open("postgres", "host= user= password= dbname= port=5432 sslmode=disable")
	if err != nil {
		log.Fatalf("Failed to connect to the database: %v", err)
	}
	defer db.Close()
	lis, err := net.Listen("tcp", "0.0.0.0:8080")
	if err != nil {
		log.Fatalf("Failed to listen: %v", err)
	}
	fmt.Printf(
		`
⚡   ________  ___  _____
⚡  / ___/ _ \/ _ \/ ___/
⚡ / (_ / , _/ ___/ /__  
⚡ \___/_/|_/_/   \___/  
⚡️Online
`)
	s := grpc.NewServer()
	usercount.RegisterUserServiceServer(s, &server{db: db})

	fmt.Println("gRPC server listening on " + lis.Addr().String())

	if err := s.Serve(lis); err != nil {
		log.Fatalf("Failed to serve: %v", err)
	}
}
