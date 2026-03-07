"""Print available bands for Sentinel-2 L2A from Earth Search."""

from pystac_client import Client

catalog = Client.open("https://earth-search.aws.element84.com/v1")
search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=[10, 59, 11, 60],
    datetime="2023-06",
    max_items=1,
)
items = list(search.items())
if items:
    item = items[0]
    print("Available bands/assets:", sorted(item.assets.keys()))
else:
    print("No items found")
