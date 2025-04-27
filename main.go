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
	ID                string      `json:"id"`
	MachineID         string      `json:"machine_id"`
	Filename          string      `json:"filename"`
	ThumbnailFilename *string     `json:"thumbnail_filename"`
	Status            string      `json:"status"`
	DetectionData     interface{} `json:"detection_data"`
	CreatedAt         string      `json:"created_at"`
	UpdatedAt         string      `json:"updated_at"`
}

var config Config
var supabaseClient *supabase.Client

func init() {
	err := godotenv.Load()
	if err != nil {
		log.Println("Warning: .env file not found")
	}

	config = Config{
		SupabaseURL:    getEnv("SUPABASE_URL", ""),
		SupabaseKey:    getEnv("SUPABASE_KEY", ""),
		UploadDir:      getEnv("UPLOAD_DIR", "uploads"),
		ThumbnailsDir:  getEnv("THUMBNAILS_DIR", "thumbnails"),
		ServerPort:     getEnv("SERVER_PORT", "8080"),
	}

	os.MkdirAll(config.UploadDir, os.ModePerm)
	os.MkdirAll(config.ThumbnailsDir, os.ModePerm)

	supabaseClient = supabase.CreateClient(config.SupabaseURL, config.SupabaseKey)
}

func main() {
	r := mux.NewRouter()

	r.HandleFunc("/api/upload", uploadImageHandler).Methods("POST")
	r.HandleFunc("/health", healthCheckHandler).Methods("GET")
	r.HandleFunc("/api/file/uploads/{file_name}", serveImageHandler).Methods("GET")
	r.HandleFunc("/api/list/machine/image/{id}", listMachineImagesHandler).Methods("GET")
	r.HandleFunc("/api/list/machine/image", listUserMachineImagesHandler).Methods("GET")


	handler := corsMiddleware(r)

	fmt.Printf("Server starting on port %s...\n", config.ServerPort)
	log.Fatal(http.ListenAndServe(":"+config.ServerPort, handler))
}

func uploadImageHandler(w http.ResponseWriter, r *http.Request) {
	err := r.ParseMultipartForm(10 << 20)
	if err != nil {
		respondWithError(w, http.StatusBadRequest, "Failed to parse form")
		return
	}

	machineID := r.FormValue("machine_id")
	log.Println("DEBUG: Received machine_id =", machineID)
	if machineID == "" {
		respondWithError(w, http.StatusBadRequest, "Machine ID is required")
		return
	}

	var machines []map[string]interface{}
	selectErr := supabaseClient.DB.From("machines").Select("*").Eq("id", machineID).Execute(&machines)
	if selectErr != nil {
		log.Printf("Supabase SELECT error: %v", selectErr)
	}

	if selectErr != nil || len(machines) == 0 {
		respondWithError(w, http.StatusBadRequest, "Invalid or missing machine ID")
		return
	}

	file, handler, err := r.FormFile("image")
	if err != nil {
		respondWithError(w, http.StatusBadRequest, "Failed to get uploaded file: "+err.Error())
		return
	}
	defer file.Close()

	fileExt := filepath.Ext(handler.Filename)
	newUUID := uuid.New().String()
	filename := newUUID + fileExt
	fullPath := filepath.Join(config.UploadDir, filename)

	dst, err := os.Create(fullPath)
	if err != nil {
		respondWithError(w, http.StatusInternalServerError, "Failed to create local file")
		return
	}
	defer dst.Close()

	_, err = io.Copy(dst, file)
	if err != nil {
		respondWithError(w, http.StatusInternalServerError, "Failed to save uploaded file")
		return
	}

	now := time.Now().UTC().Format(time.RFC3339)
	var thumbnailFilename *string = nil

	imageRecord := ImageRecord{
		ID:                newUUID,
		MachineID:         machineID,
		Filename:          filename,
		ThumbnailFilename: thumbnailFilename,
		Status:            "pending",
		DetectionData:     nil,
		CreatedAt:         now,
		UpdatedAt:         now,
	}

	var results []map[string]interface{}
	insertErr := supabaseClient.DB.From("images").Insert(imageRecord).Execute(&results)
	if insertErr != nil {
		log.Printf("Supabase INSERT error: %v", insertErr)
		os.Remove(fullPath)
		respondWithError(w, http.StatusInternalServerError, "Failed to insert into database: "+insertErr.Error())
		return
	}

	respondWithJSON(w, http.StatusCreated, map[string]interface{}{
		"success":  true,
		"message":  "Image uploaded successfully",
		"id":       newUUID,
		"filename": filename,
	})
}

func serveImageHandler(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	fileName := vars["file_name"]
	filePath := filepath.Join(config.UploadDir, fileName)

	if _, err := os.Stat(filePath); os.IsNotExist(err) {
		http.NotFound(w, r)
		return
	}

	http.ServeFile(w, r, filePath)
}

func listMachineImagesHandler(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	machineID := vars["id"]

	var images []ImageRecord
	err := supabaseClient.DB.From("images").Select("*").Eq("machine_id", machineID).Execute(&images)
	if err != nil {
		respondWithError(w, http.StatusInternalServerError, "Failed to fetch images")
		return
	}

	baseURL := fmt.Sprintf("%s/api/file/uploads", r.Host)
	for i := range images {
		images[i].Filename = fmt.Sprintf("http://%s/%s", baseURL, images[i].Filename)
		if images[i].ThumbnailFilename != nil {
			thumbnailURL := fmt.Sprintf("http://%s/%s", baseURL, *images[i].ThumbnailFilename)
			images[i].ThumbnailFilename = &thumbnailURL
		}
	}

	respondWithJSON(w, http.StatusOK, images)
}

func healthCheckHandler(w http.ResponseWriter, r *http.Request) {
	respondWithJSON(w, http.StatusOK, map[string]interface{}{
		"status": "healthy",
		"time":   time.Now().UTC().Format(time.RFC3339),
	})
}

func listUserMachineImagesHandler(w http.ResponseWriter, r *http.Request) {
    userID := r.Header.Get("X-User-ID") // For example, you can get it from JWT or header
    if userID == "" {
        respondWithError(w, http.StatusUnauthorized, "Missing user ID")
        return
    }

    var machines []map[string]interface{}
    err := supabaseClient.DB.From("machines").Select("id").Eq("user_id", userID).Execute(&machines)
    if err != nil {
        respondWithError(w, http.StatusInternalServerError, "Failed to fetch machines")
        return
    }

    var machineIDs []string
    for _, m := range machines {
        if id, ok := m["id"].(string); ok {
            machineIDs = append(machineIDs, id)
        }
    }

    if len(machineIDs) == 0 {
        respondWithJSON(w, http.StatusOK, []ImageRecord{})
        return
    }

    // Fetch images for these machines
    var images []ImageRecord
    err = supabaseClient.DB.From("images").Select("*").In("machine_id", machineIDs).Execute(&images)
    if err != nil {
        respondWithError(w, http.StatusInternalServerError, "Failed to fetch images")
        return
    }

    baseURL := fmt.Sprintf("%s/api/file/uploads", r.Host)
    for i := range images {
        images[i].Filename = fmt.Sprintf("http://%s/%s", baseURL, images[i].Filename)
        if images[i].ThumbnailFilename != nil {
            thumbnailURL := fmt.Sprintf("http://%s/%s", baseURL, *images[i].ThumbnailFilename)
            images[i].ThumbnailFilename = &thumbnailURL
        }
    }

    respondWithJSON(w, http.StatusOK, images)
}


func corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}

		next.ServeHTTP(w, r)
	})
}

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
