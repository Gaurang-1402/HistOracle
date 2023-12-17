import { LatLngExpression } from "leaflet"

export interface point {
    id: string,
    latlng: LatLngExpression,
    placeName: string
}