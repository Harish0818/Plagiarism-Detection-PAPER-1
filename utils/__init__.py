def __init__(self):
    # Initialize session with retry logic (no caching)
    self.session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        respect_retry_after_header=True
    )
    # ... rest of the code remains the same