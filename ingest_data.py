import csv
import sys       # System-specific parameters and functions
import requests

def fetch_market_data(symbol="BTCUSDT", interval="1h", limit=500):
    """Fetches historical candlestick data from the Binance API with active error monitoring."""
    base_url = "https://api.binance.com/api/v3/klines"
    query_parameters = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    
    print(f"Initiating connection to extract {limit} rows of historical data for {symbol}...")
    
    try:
        # 1. Attempt the network call
        response = requests.get(base_url, params=query_parameters, timeout=10)
        
        # 2. If the API returns a bad status code (like 404 or 500), force an error right here
        response.raise_for_status()
        
        return response.json()
        
    except requests.exceptions.Timeout:
        print("\n❌ ERROR: The server took too long to respond. Connection timed out.")
        sys.exit(1) # Shuts down the script cleanly with an error flag
        
    except requests.exceptions.HTTPError as http_err:
        print(f"\n❌ HTTP PROTOCOL ERROR: API responded with status code: {response.status_code}")
        print(f"Details: {http_err}")
        sys.exit(1)
        
    except requests.exceptions.RequestException as net_err:
        print("\n❌ CRITICAL NETWORK ERROR: Check your internet connection.")
        print(f"Details: {net_err}")
        sys.exit(1)


def save_to_csv(raw_data, output_filename="raw_market_data.csv"):
    """Parses raw API nested lists and saves them cleanly into a structured CSV file."""
    headers = ["timestamp", "open", "high", "low", "close", "volume"]
    
    try:
        with open(output_filename, mode="w", newline="") as data_file:
            writer = csv.writer(data_file)
            writer.writerow(headers)
            
            for candle in raw_data:
                clean_row = [
                    int(candle[0]),        
                    float(candle[1]),      
                    float(candle[2]),      
                    float(candle[3]),      
                    float(candle[4]),      
                    float(candle[5])       
                ]
                writer.writerow(clean_row)
        print(f"Extraction successful! Dataset locked inside: {output_filename}")
        
    except IOError as file_err:
        print(f"\n❌ OS STORAGE ERROR: Could not write data to disk. Is the CSV file open in another app?")
        print(f"Details: {file_err}")


# --- MASTER PIPELINE EXECUTION ---
if __name__ == "__main__":
    market_payload = fetch_market_data(symbol="BTCUSDT", interval="1h", limit=500)
    save_to_csv(market_payload)