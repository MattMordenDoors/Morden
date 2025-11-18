import streamlit as st
import pandas as pd
import googlemaps
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from io import BytesIO
import urllib.parse

# Replace with your Google Maps API key inside quotes
API_KEY = "AIzaSyBsjs1zDndKxZ877HIbpcbd34CnC9wc8PE"
gmaps = googlemaps.Client(key=API_KEY)

st.title("🚚 OR-Tools Route Optimizer with Google Maps Export")

# Input addresses: upload Excel or manual entry
input_method = st.radio("Select input method:", ["Upload Excel", "Enter Addresses Manually"])

addresses = []

if input_method == "Upload Excel":
    uploaded_file = st.file_uploader("Upload Excel file with 'Address' column", type=["xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        if "Address" not in df.columns:
            st.error("Excel must have an 'Address' column")
        else:
            addresses = df["Address"].dropna().tolist()
elif input_method == "Enter Addresses Manually":
    manual_input = st.text_area("Enter addresses (one per line):", height=200)
    if manual_input.strip():
        addresses = [a.strip() for a in manual_input.split("\n") if a.strip()]

start_address = st.text_input("Starting location:", "Morden Doors, Surrey BC")

def geocode_addresses(address_list):
    coords = []
    for addr in address_list:
        try:
            result = gmaps.geocode(addr)
            loc = result[0]['geometry']['location']
            coords.append((loc['lat'], loc['lng']))
        except Exception as e:
            st.warning(f"Failed to geocode: {addr}")
            coords.append(None)
    return coords

def create_distance_matrix(locations):
    import math
    def haversine(coord1, coord2):
        R = 6371
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    size = len(locations)
    matrix = []
    for from_idx in range(size):
        row = []
        for to_idx in range(size):
            if from_idx == to_idx:
                row.append(0)
            else:
                row.append(int(haversine(locations[from_idx], locations[to_idx]) * 1000))
        matrix.append(row)
    return matrix

def solve_tsp(distance_matrix):
    size = len(distance_matrix)
    manager = pywrapcp.RoutingIndexManager(size, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        index = routing.Start(0)
        route = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route.append(node)
            index = solution.Value(routing.NextVar(index))
        route.append(manager.IndexToNode(index))
        return route
    else:
        return None

def split_route(route, max_stops=25):
    chunks = []
    start = 0
    while start < len(route):
        end = min(start + max_stops, len(route))
        chunk = route[start:end]
        if start != 0 and chunk[0] != route[start-1]:
            chunk = [route[start-1]] + chunk
        chunks.append(chunk)
        start += max_stops
    return chunks

def build_google_maps_url(addresses_chunk):
    base_url = "https://www.google.com/maps/dir/?api=1"
    origin = urllib.parse.quote(addresses_chunk[0])
    destination = urllib.parse.quote(addresses_chunk[-1])
    if len(addresses_chunk) > 2:
        waypoints = "|".join(urllib.parse.quote(addr) for addr in addresses_chunk[1:-1])
        url = f"{base_url}&origin={origin}&destination={destination}&waypoints={waypoints}"
    else:
        url = f"{base_url}&origin={origin}&destination={destination}"
    return url

if addresses and start_address:
    all_addresses = [start_address] + addresses

    with st.spinner("Geocoding addresses..."):
        coords = geocode_addresses(all_addresses)

    if None in coords:
        st.error("Some addresses could not be geocoded. Please fix or remove them.")
    else:
        distance_matrix = create_distance_matrix(coords)

        with st.spinner("Optimizing route with OR-Tools..."):
            route = solve_tsp(distance_matrix)

        if route is None:
            st.error("No solution found.")
        else:
            ordered_addresses = [all_addresses[i] for i in route]

            st.success("Route optimized!")

            st.subheader("Optimized route order:")
            for i, addr in enumerate(ordered_addresses):
                st.write(f"{i+1}. {addr}")

            chunks = split_route(ordered_addresses, max_stops=25)

            st.subheader("Open routes on Google Maps (max 25 stops per link):")
            for i, chunk in enumerate(chunks):
                url = build_google_maps_url(chunk)
                st.markdown(f"[Route part {i+1}]({url})", unsafe_allow_html=True)

            ordered_df = pd.DataFrame({"Order": list(range(len(ordered_addresses))), "Address": ordered_addresses})
            buffer = BytesIO()
            ordered_df.to_excel(buffer, index=False)

            st.download_button("Download Optimized Route Excel", buffer.getvalue(), "optimized_route.xlsx")
