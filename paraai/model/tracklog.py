import base64
import re
import uuid
from datetime import datetime
from typing import Optional

import numpy as np
from pydantic import BaseModel
from scipy.ndimage import gaussian_filter1d


class TracklogBody(BaseModel):
    tracklog_id: str
    created_at_timestamp: int
    updated_at_timestamp: int
    file_name: str
    points_lat_lng_alt_ts: list[tuple[float, float, float, int]]

    def as_array(self) -> np.ndarray:
        return np.array(self.points_lat_lng_alt_ts, dtype=np.float32)

    def as_array_si(self) -> np.ndarray:
        # covert the lat long altitude data to meters relative to takeoff
        array_tracklog_body = self.as_array()
        takeoff_lat, takeoff_lng, takeoff_alt, takeoff_at_timestamp = array_tracklog_body[0]
        # Convert degrees to meters (latitude)
        takeoff_lat_m = takeoff_lat * 111319.9059
        takeoff_lng_m = takeoff_lng * 111319.9059 * np.cos(np.deg2rad(takeoff_lat))
        array_tracklog_body[:, 0] = array_tracklog_body[:, 0] * 111319.9059 - takeoff_lat_m  # Convert to meters and subtract takeoff
        array_tracklog_body[:, 1] = (
            array_tracklog_body[:, 1] * 111319.9059 * np.cos(np.deg2rad(takeoff_lat))
        ) - takeoff_lng_m  # Convert to meters and subtract takeoff
        array_tracklog_body[:, 2] = array_tracklog_body[:, 2] - takeoff_alt  # Subtract takeoff altitude
        return array_tracklog_body

    def get_array_latitude(self) -> np.ndarray:
        array_tracklog_body = self.as_array()
        return array_tracklog_body[:, 0]

    def get_array_longitude(self) -> np.ndarray:
        array_tracklog_body = self.as_array()
        return array_tracklog_body[:, 1]

    def get_array_altitude(self) -> np.ndarray:
        array_tracklog_body = self.as_array()
        return array_tracklog_body[:, 2]

    def get_array_timestamp(self) -> np.ndarray:
        array_tracklog_body = self.as_array(dtype=np.float32)
        return array_tracklog_body[:, 3]

    def get_array_turn(self) -> np.ndarray:
        # todo redo this to get degrees per second
        # get a array of values between 0 and 1 of the angle of the tracklog between every two points
        # indicating the direction of the turn. 0 is straight, 1 is 180 degrees turn either direction.
        # it should be a 3 point average
        # first and last point it should be 0
        array_latitude = self.get_array_latitude()
        array_longitude = self.get_array_longitude()

        num_points = len(array_latitude)
        if num_points < 3:
            # Not enough points to calculate turns
            return np.zeros(num_points)

        # Calculate turn based on 3 consecutive points (i-1, i, i+1)
        # For each middle point, calculate the angle between vectors (i-1->i) and (i->i+1)
        array_turn = np.zeros(num_points)

        for i in range(1, num_points - 1):
            # Vector from point i-1 to i
            vec1_lat = array_latitude[i] - array_latitude[i - 1]
            vec1_lng = array_longitude[i] - array_longitude[i - 1]

            # Vector from point i to i+1
            vec2_lat = array_latitude[i + 1] - array_latitude[i]
            vec2_lng = array_longitude[i + 1] - array_longitude[i]

            # Calculate bearings of both vectors
            bearing1 = np.arctan2(vec1_lng, vec1_lat)
            bearing2 = np.arctan2(vec2_lng, vec2_lat)

            # Calculate turn angle: difference between bearings
            # Handle wrap-around: difference between 350° and 10° should be 20°, not 340°
            bearing_diff = bearing2 - bearing1
            # Normalize to [-π, π] range to handle wrap-around
            bearing_diff = np.arctan2(np.sin(bearing_diff), np.cos(bearing_diff))
            # Take absolute value to get turn magnitude (0 to π)
            turn_angle = np.abs(bearing_diff)

            # Normalize to 0-1: 0 = no turn, 1 = 180° (π) turn
            array_turn[i] = turn_angle / np.pi

        # First and last point are 0 (already set)

        # smooth the array_turn with a smoothing factor of 0.1
        smoothing_factor = 5
        smoothed_array_turn = np.convolve(array_turn, np.ones(smoothing_factor) / smoothing_factor, mode="same")
        return smoothed_array_turn
        return array_turn

    def get_array_vertical_speed(self, smoothing_time_seconds: float = 5.0) -> np.ndarray:
        """
        Calculate vertical speed (climb/descent rate) in m/s with Gaussian smoothing.

        Args:
            smoothing_time_seconds: Time window in seconds for Gaussian smoothing

        Returns:
            Array of vertical speeds in m/s, one value per data point.
            The array has the same length as the tracklog points.
        """
        array_tracklog_body = self.as_array()
        altitudes = array_tracklog_body[:, 2]
        timestamps = array_tracklog_body[:, 3]

        num_points = len(altitudes)
        if num_points < 2:
            # Not enough points to calculate vertical speed
            return np.zeros(num_points)

        # Calculate change in altitude and time
        altitude_diff = np.diff(altitudes)
        time_diff = np.diff(timestamps)

        # Calculate raw vertical speed in m/s (avoid division by zero)
        vertical_speed_raw = np.where(time_diff != 0, altitude_diff / time_diff, 0)

        # Handle non-uniform timescale by calculating sigma in terms of data points
        # based on the desired time window
        avg_time_step = np.mean(time_diff[time_diff > 0]) if np.any(time_diff > 0) else 1.0

        # Convert smoothing time window to sigma in data points
        # For Gaussian filter, ~3*sigma covers most of the window
        # So sigma_points should be approximately smoothing_time_seconds / (3 * avg_time_step)
        sigma_points = smoothing_time_seconds / (3 * avg_time_step) if avg_time_step > 0 else 1.0

        # Ensure sigma is at least 0.5 to avoid issues with very small values
        sigma_points = max(0.5, sigma_points)

        # Apply Gaussian smoothing
        vertical_speed_smoothed = gaussian_filter1d(vertical_speed_raw, sigma=sigma_points)

        # Create output array with same length as input points
        # Assign each point the vertical speed of the segment ending at that point
        # First point uses the first segment's value
        vertical_speed_array = np.zeros(num_points)
        if len(vertical_speed_smoothed) > 0:
            vertical_speed_array[0] = vertical_speed_smoothed[0]
            vertical_speed_array[1:] = vertical_speed_smoothed

        return vertical_speed_array


class TracklogHeader(BaseModel):
    tracklog_id: str
    created_at_timestamp: int
    updated_at_timestamp: int
    file_name: str

    # header records
    date: str  # YYYY-MM-DD
    pilot_name: str
    crew_name: str
    glider_type: str
    glider_id: str
    gps_datum: str
    firmware_version: str
    hardware_version: str
    fr_type: str
    gps_receiver: str
    pressure_alt_sensor: str
    alt_gps: str
    alt_pressure: str
    competition_id: str
    competition_class: str

    takeoff_lat: float
    takeoff_lng: float
    takeoff_alt: float
    takeoff_at_timestamp: int
    landing_lat: float
    landing_lng: float
    landing_alt: float
    landing_at_timestamp: int
    bbox_min_lat: float
    bbox_min_lng: float
    bbox_max_lat: float
    bbox_max_lng: float
    duration_seconds: int  # list of (lat, lng, alt_m, ts_seconds)

    def __str__(self):
        return (
            f"""Tracklog {self.tracklog_id} for pilot {self.pilot_name} at date {self.date} with duration {self.duration_seconds} seconds"""
        )


def parse_h_dict(lines: list[str]) -> dict:
    h_dict = {}
    for line in lines:
        try:
            if not line.startswith("HF"):
                continue
            if ":" not in line:
                if line.startswith("HFDTE"):
                    h_dict["HFDTEDATE"] = line[5:7] + "-" + line[7:9] + "-" + line[9:11]
                    continue
                elif line.startswith("HFFXA"):
                    h_dict["HFFXA"] = line[len("HFFXA") :]
                    continue
                else:
                    raise Exception(f"Error alternative parsing line: {line}")
            line_key = line.split(":")[0]
            h_dict[line_key] = line.split(":")[1]
        except Exception:
            raise Exception(f"Error parsing line: {line}")
    return h_dict


def parse_b_records(text: str) -> tuple[list[tuple[float, float, float, int]], float, float, float, float, int]:
    """Parse B-records to extract GPS fixes and calculate bounding box."""
    points: list[tuple[float, float, float, int]] = []
    min_lat = 999.0
    min_lng = 999.0
    max_lat = -999.0
    max_lng = -999.0
    duration_seconds = 0

    time_pattern = re.compile(r"^B(\d{2})(\d{2})(\d{2})(\d{2})(\d{5})([NS])(\d{3})(\d{5})([EW]).*")
    first_seconds: Optional[int] = None

    for line in text.splitlines():
        if not line.startswith("B") or len(line) < 35:
            continue
        m = time_pattern.match(line)
        if not m:
            continue

        hh, mm, ss = int(m.group(1)), int(m.group(2)), int(m.group(3))
        lat_deg = int(m.group(4))
        lat_min_thousandths = int(m.group(5))
        lat_hemi = m.group(6)
        lng_deg = int(m.group(7))
        lng_min_thousandths = int(m.group(8))
        lng_hemi = m.group(9)

        lat = lat_deg + (lat_min_thousandths / 1000.0) / 60.0
        lng = lng_deg + (lng_min_thousandths / 1000.0) / 60.0
        if lat_hemi == "S":
            lat = -lat
        if lng_hemi == "W":
            lng = -lng

        # Parse altitude from IGC B-record
        # Format: BHHMMSSDDMMmmmNDDDMMmmmEAALLLLLGGGGGG
        # Where: AALLLLL = Pressure altitude (A=altitude type, LLLLL=altitude in meters)
        #        GGGGGG = GPS altitude (in decimeters)
        try:
            # Look for pressure altitude (AALLLLL format)
            alt_match = re.search(r"A(\d{5})", line)
            if alt_match:
                pressure_alt = float(alt_match.group(1))  # Already in meters
            else:
                pressure_alt = 0.0

            # Look for GPS altitude (last 6 chars if numeric)
            gps_match = re.search(r"(\d{6})$", line)
            if gps_match:
                gps_alt = float(gps_match.group(1)) / 10.0  # Convert from decimeters to meters
            else:
                gps_alt = 0.0

            # Use pressure altitude if available, otherwise GPS altitude
            alt = pressure_alt if pressure_alt > 0 else gps_alt
        except Exception:
            alt = 0.0

        seconds = hh * 3600 + mm * 60 + ss
        if first_seconds is None:
            first_seconds = seconds
        ts_seconds = seconds - first_seconds

        points.append((lat, lng, alt, int(ts_seconds)))

        min_lat = min(lat, min_lat)
        max_lat = max(lat, max_lat)
        min_lng = min(lng, min_lng)
        max_lng = max(lng, max_lng)
        duration_seconds = ts_seconds

    return points, min_lat, min_lng, max_lat, max_lng, duration_seconds


def parse_igc_bytes(file_name: str, data: bytes) -> tuple["TracklogHeader", "TracklogBody"]:
    """
    Parse IGC bytes into a TrackLog with metadata, points, and base64 payload.

    - Extract date from HFDTEddmmyy header if present; otherwise 'unknown-date'
    - Parse B-records into (lat, lng, gps_alt_m, ts_seconds from first fix)
    - Compute bbox, takeoff/landing points
    """
    text = data.decode(errors="ignore")
    tracklog_id = str(uuid.uuid4())

    # Parse different record types
    h_dict = parse_h_dict(text.splitlines())
    date_raw = h_dict.get("HFDTEDATE", "unknown-date")
    date = ""
    if date_raw == "unknown-date":
        print(text)
        raise Exception(f"Error parsing date from IGC file: {date_raw}")
    else:
        dd = date_raw[5:7]
        mm = date_raw[7:9]
        yy = date_raw[9:11]
        date = f"20{yy}-{mm}-{dd}"
    pilot_name = h_dict.get("HFPLTPILOTINCHARGE", "NKN")
    crew_name = h_dict.get("HFCM2CREW2", "NKN")
    glider_type = h_dict.get("HFGTYGLIDERTYPE", "NKN")
    glider_id = h_dict.get("HFGIDGLIDERID", "NKN")
    gps_datum = h_dict.get("HFDTMGPSDATUM", "NKN")
    firmware_version = h_dict.get("HFRFWFIRMWAREVERSION", "NKN")
    hardware_version = h_dict.get("HFRFWHARDWAREVERSION", "NKN")
    fr_type = h_dict.get("HFFTYFRTYPE", "NKN")
    gps_receiver = h_dict.get("HFGPSRECEIVER", "NKN")
    pressure_alt_sensor = h_dict.get("HFPRSPRESSALTSENSOR", "NKN")
    alt_gps = h_dict.get("HFALGALTGPS", "unknown")
    alt_pressure = h_dict.get("HFALPALTPRESSURE", "NKN")
    competition_id = h_dict.get("HFCIDCOMPETITIONID", "NKN")
    competition_class = h_dict.get("HFCCLCOMPETITIONCLASS", "NKN")

    points_lat_lng_alt_ts, min_lat, min_lng, max_lat, max_lng, duration_seconds = parse_b_records(text)

    takeoff_lat = points_lat_lng_alt_ts[0][0] if points_lat_lng_alt_ts else 0.0
    takeoff_lng = points_lat_lng_alt_ts[0][1] if points_lat_lng_alt_ts else 0.0
    takeoff_alt = points_lat_lng_alt_ts[0][2] if points_lat_lng_alt_ts else 0.0
    takeoff_at_timestamp = int(points_lat_lng_alt_ts[0][3]) if points_lat_lng_alt_ts else 0
    landing_lat = points_lat_lng_alt_ts[-1][0] if points_lat_lng_alt_ts else 0.0
    landing_lng = points_lat_lng_alt_ts[-1][1] if points_lat_lng_alt_ts else 0.0
    landing_alt = points_lat_lng_alt_ts[-1][2] if points_lat_lng_alt_ts else 0.0
    landing_at_timestamp = int(points_lat_lng_alt_ts[-1][3]) if points_lat_lng_alt_ts else 0
    igc_b64 = base64.b64encode(data).decode()

    created_at_timestamp = int(datetime.utcnow().timestamp())
    updated_at_timestamp = created_at_timestamp

    tracklog_header = TracklogHeader(
        tracklog_id=tracklog_id,
        created_at_timestamp=created_at_timestamp,
        updated_at_timestamp=updated_at_timestamp,
        file_name=file_name,
        date=date,
        pilot_name=pilot_name,
        crew_name=crew_name,
        glider_type=glider_type,
        glider_id=glider_id,
        gps_datum=gps_datum,
        firmware_version=firmware_version,
        hardware_version=hardware_version,
        fr_type=fr_type,
        gps_receiver=gps_receiver,
        pressure_alt_sensor=pressure_alt_sensor,
        alt_gps=alt_gps,
        alt_pressure=alt_pressure,
        competition_id=competition_id,
        competition_class=competition_class,
        igc_base64=igc_b64,
        takeoff_lat=takeoff_lat,
        takeoff_lng=takeoff_lng,
        takeoff_alt=takeoff_alt,
        takeoff_at_timestamp=takeoff_at_timestamp,
        landing_lat=landing_lat,
        landing_lng=landing_lng,
        landing_alt=landing_alt,
        landing_at_timestamp=landing_at_timestamp,
        bbox_min_lat=min_lat,
        bbox_min_lng=min_lng,
        bbox_max_lat=max_lat,
        bbox_max_lng=max_lng,
        duration_seconds=duration_seconds,
    )
    tracklog_body = TracklogBody(
        tracklog_id=tracklog_id,
        created_at_timestamp=created_at_timestamp,
        updated_at_timestamp=updated_at_timestamp,
        file_name=file_name,
        points_lat_lng_alt_ts=points_lat_lng_alt_ts,
    )
    return tracklog_header, tracklog_body
