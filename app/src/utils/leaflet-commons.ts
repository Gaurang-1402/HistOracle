// Map Settings
export const mapOption = {
  startZoom: 13, // Initial zoom level
  maxZoom: 18,   // Maximum zoom level
  minZoom: 5,    // Minimum zoom level
};

// Function to Get Current Location (Latitude and Longitude)
export const getCurrentPosition = () => 
  new Promise<GeolocationPosition>((resolve, reject) =>
    navigator.geolocation.getCurrentPosition(resolve, reject)
  );
