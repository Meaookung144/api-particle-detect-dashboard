package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/google/uuid"
	"github.com/gorilla/mux"
	"github.com/joho/godotenv"
	"github.com/nedpals/supabase-go"
)

// Config stores application configuration
type Config struct {
	SupabaseURL    string
	SupabaseKey    string
	UploadDir      string
	ServerPort     string
	ThumbnailsDir  string
}

// ImageUploadRequest represents the request for image upload
type ImageUploadRequest struct {
	MachineID string `json:"machine_id"`
}

// ImageRecord represents an image record in the database
type ImageRecord struct {
	ID               string      `json:"id"`
	MachineID        string      `json:"machine_id"`
	Filename         string      `json:"filename"`
	ThumbnailFilename *string     `json:"thumbnail_filename"`
	Status           string      `json:"status"`
	DetectionData    interface{} `json:"detection_data"`
	CreatedAt        string      `json:"created_at"`
	UpdatedAt        string      `json:"updated_at"`
}

var config Config
var supabaseClient *supabase.Client

func init() {
	// Load environment variables from .env file
	err := godotenv.Load()
	if err != nil {
		log.Println("Warning: .env file not found")
	}

	// Set configuration from environment variables
	config = Config{
		SupabaseURL:    getEnv("SUPABASE_URL", ""),
		SupabaseKey:    getEnv("SUPABASE_KEY", ""),
		UploadDir:      getEnv("UPLOAD_DIR", "uploads"),
		ThumbnailsDir:  getEnv("THUMBNAILS_DIR", "thumbnails"),
		ServerPort:     getEnv("SERVER_PORT", "8080"),
	}

	// Create upload and thumbnails directories if they don't exist
	os.MkdirAll(config.UploadDir, os.ModePerm)
	os.MkdirAll(config.ThumbnailsDir, os.ModePerm)

	// Initialize Supabase client
	supabaseClient = supabase.CreateClient(config.SupabaseURL, config.SupabaseKey)
}

func main() {
	r := mux.NewRouter()

	// Define routes
	r.HandleFunc("/api/upload", uploadImageHandler).Methods("POST")
	r.HandleFunc("/health", healthCheckHandler).Methods("GET")

	// Start server
	fmt.Printf("Server starting on port %s...\n", config.ServerPort)
	log.Fatal(http.ListenAndServe(":"+config.ServerPort, r))
}

func uploadImageHandler(w http.ResponseWriter, r *http.Request) {
	// Parse multipart form with 10MB max memory
	err := r.ParseMultipartForm(10 << 20)
	if err != nil {
		respondWithError(w, http.StatusBadRequest, "Failed to parse form")
		return
	}

	// Get machine ID from form
	machineID := r.FormValue("machine_id")
	if machineID == "" {
		respondWithError(w, http.StatusBadRequest, "Machine ID is required")
		return
	}

	// Check if machine exists
	var machines []map[string]interface{}

// .Execute(&machines) expects you to pass the destination!
err = supabaseClient.DB.From("machines").Select("*").Eq("id", machineID).Execute(&machines)
if err != nil || len(machines) == 0 {
    respondWithError(w, http.StatusBadRequest, "Invalid machine ID")
    log.Printf("Machine ID: %s", machineID)
    log.Printf("Supabase result: %+v", machines)
    return
}

	// Get uploaded file
	file, handler, err := r.FormFile("image")
	if err != nil {
		respondWithError(w, http.StatusBadRequest, "Failed to get uploaded file")
		return
	}
	defer file.Close()

	// Generate unique filename
	fileExt := filepath.Ext(handler.Filename)
	newUUID := uuid.New().String()
	filename := newUUID + fileExt
	fullPath := filepath.Join(config.UploadDir, filename)

	// Save file locally
	dst, err := os.Create(fullPath)
	if err != nil {
		respondWithError(w, http.StatusInternalServerError, "Failed to create file")
		return
	}
	defer dst.Close()

	// Copy uploaded file to destination
	_, err = io.Copy(dst, file)
	if err != nil {
		respondWithError(w, http.StatusInternalServerError, "Failed to save file")
		return
	}

	// Create database record
	now := time.Now().UTC().Format(time.RFC3339)
	
	var thumbnailFilename *string = nil // Initially no thumbnail
	
	imageRecord := ImageRecord{
		ID:               newUUID,
		MachineID:        machineID,
		Filename:         filename,
		ThumbnailFilename: thumbnailFilename,
		Status:           "pending",
		DetectionData:    nil,
		CreatedAt:        now,
		UpdatedAt:        now,
	}

	// Insert record into database
	var result map[string]interface{}
	err = supabaseClient.DB.From("images").Insert(imageRecord).Execute(&result)
	if err != nil {
		// Delete local file if database insert fails
		os.Remove(fullPath)
		respondWithError(w, http.StatusInternalServerError, "Failed to create database record: "+err.Error())
		return
	}

	// Return success response
	respondWithJSON(w, http.StatusCreated, map[string]interface{}{
		"success": true,
		"message": "Image uploaded successfully",
		"id":      newUUID,
		"filename": filename,
	})
}

func healthCheckHandler(w http.ResponseWriter, r *http.Request) {
	respondWithJSON(w, http.StatusOK, map[string]interface{}{
		"status": "healthy",
		"time":   time.Now().UTC().Format(time.RFC3339),
	})
}

// Helper functions
func respondWithError(w http.ResponseWriter, code int, message string) {
	respondWithJSON(w, code, map[string]string{"error": message})
}

func respondWithJSON(w http.ResponseWriter, code int, payload interface{}) {
	response, _ := json.Marshal(payload)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	w.Write(response)
}

func getEnv(key, fallback string) string {
	value := os.Getenv(key)
	if value == "" {
		return fallback
	}
	return value
}